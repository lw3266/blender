/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#pragma once

#ifdef WITH_OPENCL

#  define CL_TARGET_OPENCL_VERSION 300
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

  void load_image_info() override;
  bool synchronize() override;
  bool enqueue(DeviceKernel kernel, int work_size, const DeviceKernelArguments &args) override;

  void zero_to_device(device_memory &mem) override;
  void copy_to_device(device_memory &mem) override;
  void copy_from_device(device_memory &mem) override;
  void *copy_from_device_synchronized(device_memory &mem, vector<uint8_t> &storage) override;
};

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */