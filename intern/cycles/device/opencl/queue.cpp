/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#ifdef WITH_OPENCL

#  include "device/opencl/queue.h"
#  include "device/opencl/device_impl.h"

CCL_NAMESPACE_BEGIN

OpenCLDeviceQueue::OpenCLDeviceQueue(OpenCLDevice *device)
    : DeviceQueue(device), opencl_device(device)
{
  cl_int err;
  cl_queue = clCreateCommandQueue(opencl_device->cl_context_id, opencl_device->cl_device, 0, &err);
}

OpenCLDeviceQueue::~OpenCLDeviceQueue()
{
  if (cl_queue) clReleaseCommandQueue(cl_queue);
}

void OpenCLDeviceQueue::synchronize()
{
  if (cl_queue) clFinish(cl_queue);
}

void OpenCLDeviceQueue::enqueue(DeviceKernel kernel, int work_size, DeviceKernelArguments args)
{
  cl_kernel cl_kern = opencl_device->kernels[kernel];
  if (!cl_kern) return;

  /* Bind device kernel arguments dynamically */
  int arg_idx = 0;
  for (const auto &arg : args.values) {
    if (arg.type == DeviceKernelArguments::POINTER) {
      cl_mem ptr = (cl_mem)arg.pointer;
      clSetKernelArg(cl_kern, arg_idx++, sizeof(cl_mem), &ptr);
    }
    else if (arg.type == DeviceKernelArguments::INT) {
      int val = arg.int_value;
      clSetKernelArg(cl_kern, arg_idx++, sizeof(int), &val);
    }
  }

  size_t global_size = work_size;
  clEnqueueNDRangeKernel(cl_queue, cl_kern, 1, NULL, &global_size, NULL, 0, NULL, NULL);
}

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */