#pragma once

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/rvv/rvv_helper.h>
#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_RVV)
#include <ATen/cpu/vec/rvv/vec_float.h>
#include <ATen/cpu/vec/rvv/vec_qint32.h>
#include <ATen/cpu/vec/rvv/vec_qint8.h>
#include <ATen/cpu/vec/rvv/vec_quint8.h>
#include <ATen/cpu/vec/rvv/vec_bfloat16.h>
#endif
