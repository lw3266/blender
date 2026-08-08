# SPDX-FileCopyrightText: 2011-2026 Blender Foundation
#
# SPDX-License-Identifier: Apache-2.0

if(NOT WITH_CYCLES_DEVICE_OPENCL)
  # Create a dummy custom target if OpenCL is disabled so CMake dependency chaining passes cleanly
  add_custom_target(cycles_kernel_opencl)
  return()
endif()

# List of OpenCL kernel source files
set(SRC_OPENCL_KERNEL_FILES
  kernel.clcpp
)

# -----------------------------------------------------------------------------
# 1. Install kernel sources for runtime JIT compilation by the OpenCL driver
# -----------------------------------------------------------------------------

delayed_install(
  ${CMAKE_CURRENT_SOURCE_DIR}
  "${SRC_OPENCL_KERNEL_FILES}"
  ${CYCLES_INSTALL_PATH}/scripts/addons/cycles/kernel/device/opencl
)

# -----------------------------------------------------------------------------
# 2. Kernel Build / Pre-compilation Target
# -----------------------------------------------------------------------------

# If building with embedded sources or pre-compiled binaries
if(WITH_CYCLES_KERNEL_EMBED)
  # Compress/embed source code using zstd_compress (same as CUDA/HIP)
  set(OPENCL_EMBED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/opencl_kernel_source.h)
  
  add_custom_command(
    OUTPUT ${OPENCL_EMBED_HEADER}
    COMMAND zstd_compress
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel.clcpp
            ${OPENCL_EMBED_HEADER}
            opencl_kernel_source
    DEPENDS zstd_compress ${CMAKE_CURRENT_SOURCE_DIR}/kernel.clcpp
    COMMENT "Embedding OpenCL kernel source"
  )
  
  add_custom_target(cycles_kernel_opencl DEPENDS ${OPENCL_EMBED_HEADER})
else()
  # Default target that triggers the delayed_install step
  add_custom_target(cycles_kernel_opencl DEPENDS ${SRC_OPENCL_KERNEL_FILES})
endif()