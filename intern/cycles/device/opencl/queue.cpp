/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#ifdef WITH_OPENCL

#  include "device/opencl/queue.h"
#  include "device/opencl/device_impl.h"
#  include "util/log.h"

CCL_NAMESPACE_BEGIN

OpenCLDeviceQueue::OpenCLDeviceQueue(OpenCLDevice *device)
    : DeviceQueue(device), opencl_device(device)
{
  cl_int err;
#if CL_TARGET_OPENCL_VERSION >= 200
  cl_queue = clCreateCommandQueueWithProperties(
      opencl_device->cl_context_id, opencl_device->cl_device, NULL, &err);
#else
  cl_queue = clCreateCommandQueue(
      opencl_device->cl_context_id, opencl_device->cl_device, 0, &err);
#endif
}

OpenCLDeviceQueue::~OpenCLDeviceQueue()
{
  if (cl_queue) {
    clReleaseCommandQueue(cl_queue);
  }
}

int OpenCLDeviceQueue::num_concurrent_states(const size_t /*state_size*/) const
{
  return 65536;
}

int OpenCLDeviceQueue::num_concurrent_busy_states(const size_t /*state_size*/) const
{
  return 65536;
}

void OpenCLDeviceQueue::init_execution()
{
}

void OpenCLDeviceQueue::load_image_info()
{
}

bool OpenCLDeviceQueue::synchronize()
{
  if (cl_queue) {
    return (clFinish(cl_queue) == CL_SUCCESS);
  }
  return false;
}

bool OpenCLDeviceQueue::enqueue(DeviceKernel kernel, int work_size, const DeviceKernelArguments &args)
{
  cl_kernel cl_kern = opencl_device->kernels[kernel];
  if (!cl_kern) {
    return false;
  }

  /* Bind device kernel arguments dynamically using values, sizes, and count */
  for (int i = 0; i < args.count; i++) {
    cl_int err = clSetKernelArg(cl_kern, (cl_uint)i, args.sizes[i], args.values[i]);
    if (err != CL_SUCCESS) {
      LOG_ERROR << "OpenCL clSetKernelArg failed for argument " << i << " with error " << err;
      return false;
    }
  }

  size_t global_size = work_size;
  cl_int err = clEnqueueNDRangeKernel(cl_queue, cl_kern, 1, NULL, &global_size, NULL, 0, NULL, NULL);
  return (err == CL_SUCCESS);
}

void OpenCLDeviceQueue::zero_to_device(device_memory &mem)
{
  opencl_device->mem_zero(mem);
}

void OpenCLDeviceQueue::copy_to_device(device_memory &mem)
{
  opencl_device->mem_copy_to(mem);
}

void OpenCLDeviceQueue::copy_from_device(device_memory &mem)
{
  opencl_device->mem_copy_from(mem, 0, mem.data_width, mem.data_height, sizeof(uint8_t));
}

void *OpenCLDeviceQueue::copy_from_device_synchronized(device_memory &mem, vector<uint8_t> &storage)
{
  storage.resize(mem.memory_size());
  cl_mem cl_buf = (cl_mem)mem.device_pointer;
  clEnqueueReadBuffer(cl_queue, cl_buf, CL_TRUE, 0, mem.memory_size(), storage.data(), 0, NULL, NULL);
  return storage.data();
}

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */