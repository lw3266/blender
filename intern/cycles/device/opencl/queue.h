/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#pragma once

#ifdef WITH_OPENCL

#  include <CL/cl.h>
#  include "device/queue.h"

CCL_NAMESPACE_BEGIN

class OpenCLDevice;

class OpenCLDeviceQueue : public DeviceQueue {
 public:
  OpenCLDevice *opencl_device;
  cl_command_queue cl_queue;

  OpenCLDeviceQueue(OpenCLDevice *device);
  ~OpenCLDeviceQueue() override;

  void synchronize() override;
  void enqueue(DeviceKernel kernel, int work_size, DeviceKernelArguments args) override;
};

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */