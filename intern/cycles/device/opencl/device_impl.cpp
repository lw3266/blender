/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#ifdef WITH_OPENCL

#  include "device/opencl/device_impl.h"
#  include "device/opencl/queue.h"
#include "device/kernel.h"   // for device_kernel_as_string()
#  include "util/log.h"
#  include "util/path.h"
#  include "util/string.h"

#  include <vector>
#include <unordered_set>

CCL_NAMESPACE_BEGIN

bool device_opencl_init()
{
  cl_uint num_platforms = 0;
  if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
    return false;
  }
  return true;
}

void device_opencl_info(vector<DeviceInfo> &devices)
{
  cl_uint num_platforms = 0;
  if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
    return;
  }

  vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), NULL);

  for (cl_platform_id platform : platforms) {
    cl_uint num_devices = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &num_devices) != CL_SUCCESS) {
      continue;
    }

    vector<cl_device_id> cl_devices(num_devices);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, num_devices, cl_devices.data(), NULL);

    for (cl_device_id dev : cl_devices) {
      char name[256];
      clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);

      DeviceInfo info;
      info.type = DEVICE_OPENCL;
      info.description = string(name);
      info.num = devices.size();
      info.id = string_printf("OPENCL_%p", dev);
      info.has_peer_memory = false;
      devices.push_back(info);
    }
  }
}

unique_ptr<Device> device_opencl_create(const DeviceInfo &info,
                                        Stats &stats,
                                        Profiler &profiler,
                                        bool headless)
{
  return make_unique<OpenCLDevice>(info, stats, profiler, headless);
}

OpenCLDevice::OpenCLDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless)
    : GPUDevice(info, stats, profiler, headless), cl_prog(NULL)
{
  cl_uint num_platforms = 0;
  clGetPlatformIDs(0, NULL, &num_platforms);
  vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), NULL);

  cl_device = NULL;
  for (cl_platform_id platform : platforms) {
    cl_uint num_devices = 0;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &num_devices) == CL_SUCCESS) {
      vector<cl_device_id> cl_devices(num_devices);
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, num_devices, cl_devices.data(), NULL);
      for (cl_device_id dev : cl_devices) {
        char name[256];
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
        if (info.description == name) {
          cl_device = dev;
          cl_platform = platform;
          break;
        }
      }
    }
  }

  cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cl_platform, 0};
  cl_int err;
  cl_context_id = clCreateContext(props, 1, &cl_device, NULL, NULL, &err);

  memset(kernels, 0, sizeof(kernels));
}

OpenCLDevice::~OpenCLDevice()
{
  for (int i = 0; i < DEVICE_KERNEL_NUM; i++) {
    if (kernels[i]) clReleaseKernel(kernels[i]);
  }
  if (cl_prog) clReleaseProgram(cl_prog);
  if (cl_context_id) clReleaseContext(cl_context_id);
}

BVHLayoutMask OpenCLDevice::get_bvh_layout_mask(const uint /*kernel_features*/) const
{
  return BVH_LAYOUT_BVH2;
}

bool OpenCLDevice::compile_opencl_cpp_program()
{
  string kernel_path = path_get("scripts/addons_core/cycles/source/kernel/device/opencl/kernel.clcpp");
  string source_code;
  if (!path_read_text(kernel_path, source_code)) {
    return false;
  }
  const char *src = source_code.c_str();

  cl_int err;
  cl_prog = clCreateProgramWithSource(cl_context_id, 1, &src, NULL, &err);
  if (err != CL_SUCCESS) return false;

  /* Use Clang OpenCL C++ / SPIR-V flags */
  const char *build_options = "-cl-std=CLC++ -I";
  err = clBuildProgram(cl_prog, 1, &cl_device, build_options, NULL, NULL);

  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(cl_prog, cl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    vector<char> log(log_size);
    clGetProgramBuildInfo(cl_prog, cl_device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
    LOG_ERROR << "OpenCL C++ Compilation Error:\n" << log.data();
    return false;
  }
  return true;
}

bool OpenCLDevice::load_kernels(const uint /*kernel_features*/)
{
  if (!compile_opencl_cpp_program()) {
    return false;
  }

  /* Kernels currently implemented in kernel.clcpp. Anything not listed here has
   * no OpenCL entry point yet; its slot in kernels[] stays null and
   * OpenCLDeviceQueue::enqueue() will fail for it explicitly.
   * NOTE: "opencl_integrator_shade_light" in kernel.clcpp predates the NEE/
   * forward split (DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE/_FORWARD) and has
   * no 1:1 match under the current naming convention, so it's intentionally
   * left out here rather than silently bound to the wrong kernel. */
  static const std::unordered_set<int> supported_kernels = {
      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA,
      DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE,
      DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST,
      DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW,
      DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE,
      DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
      DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND,
      DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME,
      DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW,
  };

  bool all_ok = true;

  for (int i = 0; i < DEVICE_KERNEL_NUM; i++) {
    const DeviceKernel kernel = (DeviceKernel)i;
    if (!supported_kernels.count(i)) {
      continue;
    }

    const std::string function_name = std::string("opencl_") + device_kernel_as_string(kernel);

    cl_int err = CL_SUCCESS;
    kernels[i] = clCreateKernel(cl_prog, function_name.c_str(), &err);

    if (err != CL_SUCCESS || !kernels[i]) {
      LOG_ERROR << "Unable to create OpenCL kernel \"" << function_name << "\", error " << err;
      all_ok = false;
    }
  }

  return all_ok;
}

void OpenCLDevice::mem_alloc(device_memory &mem)
{
  cl_int err;
  cl_mem cl_buf = clCreateBuffer(
      cl_context_id, CL_MEM_READ_WRITE, mem.memory_size(), NULL, &err);
  mem.device_pointer = (device_ptr)cl_buf;
}

void OpenCLDevice::mem_copy_to(device_memory &mem)
{
  if (!mem.device_pointer) mem_alloc(mem);
  cl_mem cl_buf = (cl_mem)mem.device_pointer;
  clEnqueueWriteBuffer(
      ((OpenCLDeviceQueue *)gpu_queue_create().get())->cl_queue,
      cl_buf, CL_TRUE, 0, mem.memory_size(), mem.host_pointer, 0, NULL, NULL);
}

void OpenCLDevice::mem_copy_from(device_memory &mem, size_t, size_t, size_t, size_t)
{
  cl_mem cl_buf = (cl_mem)mem.device_pointer;
  clEnqueueReadBuffer(
      ((OpenCLDeviceQueue *)gpu_queue_create().get())->cl_queue,
      cl_buf, CL_TRUE, 0, mem.memory_size(), mem.host_pointer, 0, NULL, NULL);
}

void OpenCLDevice::mem_zero(device_memory &mem)
{
  if (!mem.device_pointer) mem_alloc(mem);
  cl_mem cl_buf = (cl_mem)mem.device_pointer;
  cl_uchar pattern = 0;
  clEnqueueFillBuffer(
      ((OpenCLDeviceQueue *)gpu_queue_create().get())->cl_queue,
      cl_buf, &pattern, 1, 0, mem.memory_size(), 0, NULL, NULL);
}

void OpenCLDevice::mem_free(device_memory &mem)
{
  if (mem.device_pointer) {
    clReleaseMemObject((cl_mem)mem.device_pointer);
    mem.device_pointer = 0;
  }
}

void OpenCLDevice::mem_move_to_host(device_memory & /*mem*/)
{
}

/* GPUDevice interface implementations */

void OpenCLDevice::get_device_memory_info(size_t &total, size_t &free)
{
  cl_ulong mem_size = 0;
  clGetDeviceInfo(cl_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
  total = (size_t)mem_size;
  free = (device_mem_in_use < total) ? (total - device_mem_in_use) : 0;
}

bool OpenCLDevice::alloc_device(void *&device_pointer, const size_t size)
{
  cl_int err;
  cl_mem cl_buf = clCreateBuffer(cl_context_id, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS || !cl_buf) {
    device_pointer = nullptr;
    return false;
  }
  device_pointer = (void *)cl_buf;
  return true;
}

void OpenCLDevice::free_device(void *device_pointer)
{
  if (device_pointer) {
    clReleaseMemObject((cl_mem)device_pointer);
  }
}

bool OpenCLDevice::shared_alloc(void *&shared_pointer, const size_t size)
{
  shared_pointer = host_alloc(MEM_READ_WRITE, size);
  return (shared_pointer != nullptr);
}

void OpenCLDevice::shared_free(void *shared_pointer)
{
  if (shared_pointer) {
    host_free(MEM_READ_WRITE, shared_pointer, 0);
  }
}

void *OpenCLDevice::shared_to_device_pointer(const void *shared_pointer)
{
  return (void *)shared_pointer;
}

void OpenCLDevice::copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size)
{
  cl_mem cl_buf = (cl_mem)device_pointer;
  clEnqueueWriteBuffer(
      ((OpenCLDeviceQueue *)gpu_queue_create().get())->cl_queue,
      cl_buf, CL_TRUE, 0, size, host_pointer, 0, NULL, NULL);
}

void OpenCLDevice::const_copy_to(const char *name, void *host, size_t size)
{
  /* If you have a specific buffer or constant memory allocation logic
   * for OpenCL constants, handle it here.
   *
   * For now, a stub/pass-through implementation allows the project to link: */
  (void)name;
  (void)host;
  (void)size;
}

unique_ptr<DeviceQueue> OpenCLDevice::gpu_queue_create()
{
  return make_unique<OpenCLDeviceQueue>(this);
}

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */