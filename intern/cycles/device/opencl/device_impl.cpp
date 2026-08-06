/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#ifdef WITH_OPENCL

#  include "device/opencl/device_impl.h"
#  include "device/opencl/queue.h"
#  include "util/log.h"
#  include "util/path.h"
#  include "util/string.h"

#  include <vector>

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
  string kernel_path = path_get("scripts/addons/cycles/kernel/device/opencl/kernel.clcpp");
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

bool OpenCLDevice::load_kernels(const uint64_t /*kernel_features*/)
{
  if (!compile_opencl_cpp_program()) return false;

  static const char *kernel_names[DEVICE_KERNEL_NUM] = {
      "opencl_integrator_init_from_camera",
      "opencl_integrator_intersect_closest",
      "opencl_integrator_shade_surface",
      "opencl_integrator_shade_volume",
      "opencl_integrator_shade_shadow",
  };

  for (int i = 0; i < DEVICE_KERNEL_NUM; i++) {
    if (kernel_names[i]) {
      cl_int err;
      kernels[i] = clCreateKernel(cl_prog, kernel_names[i], &err);
    }
  }
  return true;
}

void OpenCLDevice::const_copy_to(const char * /*name*/, void * /*host*/, const size_t /*size*/)
{
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

unique_ptr<DeviceQueue> OpenCLDevice::gpu_queue_create()
{
  return make_unique<OpenCLDeviceQueue>(this);
}

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */