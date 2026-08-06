/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#pragma once

#ifdef WITH_OPENCL

#  define CL_TARGET_OPENCL_VERSION 300
#  include <CL/cl.h>

#  include "bvh/params.h"
#  include "device/device.h"
#  include "device/opencl/device.h"

CCL_NAMESPACE_BEGIN

class OpenCLDevice : public GPUDevice {
 public:
  cl_platform_id cl_platform;
  cl_device_id cl_device;
  cl_context cl_context_id;
  cl_program cl_prog;
  cl_kernel kernels[DEVICE_KERNEL_NUM];

  OpenCLDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless);
  ~OpenCLDevice() override;

  BVHLayoutMask get_bvh_layout_mask(const uint kernel_features) const override;
  bool load_kernels(const uint kernel_features) override;

  void const_copy_to(const char *name, void *host, const size_t size) override;
  void mem_alloc(device_memory &mem) override;
  void mem_copy_to(device_memory &mem) override;
  void mem_copy_from(device_memory &mem, size_t y, size_t w, size_t h, size_t elem) override;
  void mem_zero(device_memory &mem) override;
  void mem_free(device_memory &mem) override;
  void mem_move_to_host(device_memory &mem) override;

  unique_ptr<DeviceQueue> gpu_queue_create() override;

 protected:
  /* GPUDevice pure virtual interface implementations */
  void get_device_memory_info(size_t &total, size_t &free) override;
  bool alloc_device(void *&device_pointer, const size_t size) override;
  void free_device(void *device_pointer) override;
  bool shared_alloc(void *&shared_pointer, const size_t size) override;
  void shared_free(void *shared_pointer) override;
  void *shared_to_device_pointer(const void *shared_pointer) override;
  void copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size) override;

 private:
  bool compile_opencl_cpp_program();
};

CCL_NAMESPACE_END

#endif /* WITH_OPENCL */