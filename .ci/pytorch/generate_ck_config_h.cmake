# Generate ck/config.h without configuring Composable Kernel.
#
# ROCm stopped installing the CK headers system-wide, so the inductor CK
# backends compile against the headers shipped in the ck4inductor wheel. Every
# header they #include is checked into CK except one: ck/config.h is generated
# by CK's build from include/ck/config.h.in, so it ships in neither ROCm nor the
# wheel's source tree, and without it every CK choice fails to compile and is
# silently discarded. This script produces it so the wheel can carry it too.
#
# Running CK's own CMake to produce it is not an option here: CMakeLists.txt
# needs find_package(ROCM), find_package(hip REQUIRED) and enable_language(HIP),
# none of which are satisfiable in the wheel-build container -- it has hipcc but
# no ROCm CMake packages, and the TheRock SDK wheels do not ship any either.
#
# config.h.in only substitutes plain variables, so this script sets them and
# calls the same configure_file. Invoke with:
#
#   cmake -D CK_SOURCE_DIR=<ck> -D CK_OUTPUT=<out/ck/config.h> \
#         -D CK_GPU_TARGETS="gfx942;gfx950;gfx1250" \
#         -P generate_ck_config_h.cmake
#
# The rules below MIRROR composable_kernel/CMakeLists.txt (the arch blocks around
# lines 446-512, and the dtype block around 96-134). They are a copy, not a
# reference, so they can drift. Every variable below is assigned unconditionally
# -- "ON" or "" -- which lets the check at the end fail loudly when config.h.in
# grows a placeholder this script has never heard of. That is the drift this
# script can detect; a rule whose *condition* has changed upstream still needs a
# human to notice.
#
# NOTE ON MULTI-ARCH: a single header serves every arch in CK_GPU_TARGETS, so
# arch-specific macros are unioned.

cmake_minimum_required(VERSION 3.21)

foreach(required CK_SOURCE_DIR CK_OUTPUT CK_GPU_TARGETS)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "generate_ck_config_h: -D ${required}=... is required")
  endif()
endforeach()

set(config_in "${CK_SOURCE_DIR}/include/ck/config.h.in")
if(NOT EXISTS "${config_in}")
  message(FATAL_ERROR "generate_ck_config_h: ${config_in} not found")
endif()

set(SUPPORTED_GPU_TARGETS "${CK_GPU_TARGETS}")

# Set every variable to "ON" or to the empty string, never leave one undefined.
# An unset variable and a deliberately-OFF one both render as `/* #undef ... */`,
# so the drift check below cannot tell them apart -- it would have to treat every
# arch-gated OFF as drift, which fails for any target list that does not happen to
# enable everything. Assigning "" keeps the check meaningful: undefined then means
# only "this script forgot about it".
macro(ck_option name)
  if(${ARGN})
    set(${name} "ON")
  else()
    set(${name} "")
  endif()
endmacro()

# Dtypes. CK never sets CK_ENABLE_ALL_DTYPES; leaving both it and DTYPES unset is
# what selects the unquoted dtype branch in config.h.in. Setting it would select
# the other branch, whose values are quoted (`"ON"` rather than `ON`).
set(CK_ENABLE_ALL_DTYPES "")
foreach(dtype INT8 FP8 BF8 FP16 BF16 FP32 FP64)
  set(CK_ENABLE_${dtype} "ON")
endforeach()

# Instance families, keyed on the target list exactly as CK keys them.
ck_option(CK_ENABLE_DL_KERNELS
          SUPPORTED_GPU_TARGETS MATCHES "gfx101|gfx103|gfx10-1|gfx10-3")
# DPP is opt-in: CK declares DISABLE_DPP_KERNELS with a default of ON.
set(CK_ENABLE_DPP_KERNELS "")
ck_option(CK_USE_XDL SUPPORTED_GPU_TARGETS MATCHES "gfx9|gfx11|gfx12")
ck_option(CK_USE_GFX94 SUPPORTED_GPU_TARGETS MATCHES "gfx94|gfx95")
ck_option(CK_USE_WMMA SUPPORTED_GPU_TARGETS MATCHES "gfx11|gfx12")
ck_option(CK_USE_WMMA_FP8 SUPPORTED_GPU_TARGETS MATCHES "gfx12")
ck_option(CK_USE_NATIVE_MX_SUPPORT SUPPORTED_GPU_TARGETS MATCHES "gfx950|gfx1250")

# FP8 flavor: OCP on gfx12/gfx950, FNUZ on gfx90a/gfx94x. A target list spanning
# both sets both, which is why the fp8 path needs care.
#
# These are the only CK_USE_* macros tested by value rather than with defined():
# amd_ck_fp8.hpp:347 and friends use `#if CK_USE_OCP_FP8`. The value emitted here
# is the literal `ON`, an undefined identifier that `#if` evaluates to 0 -- so in
# a header-only consumer NEITHER arm is taken and f8_t falls through to fnuz.
#
# That is not a defect in this file: CK's own CMakeLists.txt pairs each of these
# with add_definitions(-DCK_USE_OCP_FP8), an *empty* definition which is what `#if`
# actually sees in a real CK build, and amd_ck_fp8.hpp:17 supplies `0` when nothing
# defined it at all. We emit the header but not the -D flags, so enabling the fp8
# path would mean passing them from compile_command.py -- not changing the value
# here, which would diverge from the header ROCm ships.
#
# Inert today: use_ck_template() admits only float16/bfloat16/float32.
ck_option(CK_USE_OCP_FP8 SUPPORTED_GPU_TARGETS MATCHES "gfx12|gfx950")
ck_option(CK_USE_FNUZ_FP8 SUPPORTED_GPU_TARGETS MATCHES "gfx90a|gfx94")

# An opt-in CK option, default OFF, and only meaningful on gfx908/gfx90a.
set(CK_USE_FP8_ON_UNSUPPORTED_ARCH "")

# Fail if config.h.in expects a variable this script never mentions. Because every
# variable above is assigned either way, "undefined here" can only mean this script
# has not been taught about it -- so it is a new placeholder upstream added, and
# configure_file would silently render it `/* #undef ... */` as though we had
# deliberately turned it off.
file(READ "${config_in}" config_in_text)
string(REGEX MATCHALL "@[A-Za-z_0-9]+@" placeholders "${config_in_text}")
list(REMOVE_DUPLICATES placeholders)
set(unknown "")
foreach(placeholder ${placeholders})
  string(REGEX REPLACE "^@(.*)@$" "\\1" name "${placeholder}")
  # DTYPES is deliberately unset: leaving it undefined is what selects the
  # CK_ENABLE_ALL_DTYPES branch in config.h.in.
  if(NOT name STREQUAL "DTYPES" AND NOT DEFINED ${name})
    list(APPEND unknown "${name}")
  endif()
endforeach()
if(unknown)
  message(FATAL_ERROR
    "generate_ck_config_h: config.h.in expects variables this script does not "
    "set: ${unknown}. CK's CMakeLists.txt has changed; update the rules above "
    "to match before regenerating.")
endif()

configure_file("${config_in}" "${CK_OUTPUT}")
message(STATUS "generate_ck_config_h: wrote ${CK_OUTPUT} for ${CK_GPU_TARGETS}")
