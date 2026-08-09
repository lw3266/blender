/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#define KERNEL_GPU
#define KERNEL_OPENCL

#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END

#ifndef ATTR_FALLTHROUGH
#  define ATTR_FALLTHROUGH
#endif

/* --------------------------------------------------------------------
 * Qualifiers
 * -------------------------------------------------------------------- */

#define ccl_device inline
#define ccl_device_extern
#define ccl_device_inline inline
#define ccl_device_forceinline inline
#define ccl_device_noinline
#define ccl_device_noinline_cpu ccl_device
#define ccl_device_inline_method ccl_device
#define ccl_device_template_spec template<> ccl_device_inline

#define ccl_global __global
#define ccl_private __private
#define ccl_constant __constant
#define ccl_local __local

#define ccl_gpu_shared __local
#define ccl_ray_data ccl_private

#define ccl_may_alias
#define ccl_restrict restrict

#define ccl_align(n) __attribute__((aligned(n)))

#define ccl_optional_struct_init
#define ccl_attr_maybe_unused [[maybe_unused]]

#define kernel_assert(cond)

/* The OpenCL device environment does not have a C++ `device`
 * keyword like CUDA. */
#define device