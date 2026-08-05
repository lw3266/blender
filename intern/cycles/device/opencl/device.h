/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2011-2024 Blender Foundation */

#pragma once

#include "device/device.h"

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;

void device_opencl_info(vector<DeviceInfo> &devices);
Device *device_opencl_create(const DeviceInfo &info, Stats &stats, Profiler &profiler);

CCL_NAMESPACE_END
