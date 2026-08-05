/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#pragma once

#include "device/device.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;

bool device_opencl_init();
void device_opencl_info(vector<DeviceInfo> &devices);
unique_ptr<Device> device_opencl_create(const DeviceInfo &info,
                                        Stats &stats,
                                        Profiler &profiler,
                                        bool headless);

CCL_NAMESPACE_END