/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#pragma once

#ifdef WITH_OPENCL

#  define CL_TARGET_OPENCL_VERSION 300
#  include <CL/cl.h>

#  include "device/device.h"
#  include "device/opencl/device.h"
#  include "kernel/types.h"

CCL_NAMESPACE_BEGIN

class OpenCLDevice : public Device {
 public:
  cl_platform_id cl_platform;
  cl_device_id cl_device;
  cl_context cl_context_id;
  cl_program cl_prog;
  cl_kernel kernels[DEVICE_KERNEL_NUM];

  OpenCLDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless);
  ~OpenCLDevice() override;

  bool load_kernels(const DeviceRequestedFeatures &requested_features) override;

  void mem_alloc(device_memory &mem) override;
  void mem_copy_to(device_memory &mem) override;
  void mem_copy_from(device_memory &mem, size_t y, size_t w, size_t h, size_t elem) override;
  void mem_zero(device_memory &mem) override;
  void mem_free(device_memory &mem) override;

  unique_ptr<DeviceQueue> gpu_queue_create() override;

 private:
  bool compile_opencl_cpp_program();
};

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */