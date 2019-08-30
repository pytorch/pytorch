# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

LOCAL_PATH := $(call my-dir)/..

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi armeabi-v7a))
include $(CLEAR_VARS)
LOCAL_MODULE := qnnpack_aarch32_neon_ukernels
LOCAL_SRC_FILES += \
  src/q8avgpool/mp8x9p8q-neon.c \
  src/q8avgpool/up8x9-neon.c \
  src/q8avgpool/up8xm-neon.c \
  src/q8conv/4x8-aarch32-neon.S \
  src/q8dwconv/mp8x25-neon.c \
  src/q8dwconv/up8x9-aarch32-neon.S \
  src/q8gavgpool/mp8x7p7q-neon.c \
  src/q8gavgpool/up8x7-neon.c \
  src/q8gavgpool/up8xm-neon.c \
  src/q8gemm/4x-sumrows-neon.c \
  src/q8gemm/4x8-aarch32-neon.S \
  src/q8gemm/4x8c2-xzp-aarch32-neon.S \
  src/q8vadd/neon.c \
  src/u8clamp/neon.c \
  src/u8lut32norm/scalar.c \
  src/u8maxpool/16x9p8q-neon.c \
  src/u8maxpool/sub16-neon.c \
  src/u8rmax/neon.c \
  src/x8lut/scalar.c \
  src/x8zip/x2-neon.c \
  src/x8zip/x3-neon.c \
  src/x8zip/x4-neon.c \
  src/x8zip/xm-neon.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=c99 -Wall -O2 -march=armv7-a -mfloat-abi=softfp -mfpu=neon
LOCAL_STATIC_LIBRARIES := cpuinfo fxdiv
include $(BUILD_STATIC_LIBRARY)
endif # armeabi or armeabi-v7a

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),arm64-v8a))
include $(CLEAR_VARS)
LOCAL_MODULE := qnnpack_aarch64_neon_ukernels
LOCAL_SRC_FILES += \
  src/q8avgpool/mp8x9p8q-neon.c \
  src/q8avgpool/up8x9-neon.c \
  src/q8avgpool/up8xm-neon.c \
  src/q8conv/8x8-aarch64-neon.S \
  src/q8dwconv/mp8x25-neon.c \
  src/q8dwconv/up8x9-neon.c \
  src/q8gavgpool/mp8x7p7q-neon.c \
  src/q8gavgpool/up8x7-neon.c \
  src/q8gavgpool/up8xm-neon.c \
  src/q8gemm/8x8-aarch64-neon.S \
  src/q8vadd/neon.c \
  src/u8clamp/neon.c \
  src/u8lut32norm/scalar.c \
  src/u8maxpool/16x9p8q-neon.c \
  src/u8maxpool/sub16-neon.c \
  src/u8rmax/neon.c \
  src/x8lut/scalar.c \
  src/x8zip/x2-neon.c \
  src/x8zip/x3-neon.c \
  src/x8zip/x4-neon.c \
  src/x8zip/xm-neon.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=c99 -Wall -O2
LOCAL_STATIC_LIBRARIES := cpuinfo fxdiv
include $(BUILD_STATIC_LIBRARY)
endif # arm64-v8a

ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64))
include $(CLEAR_VARS)
LOCAL_MODULE := qnnpack_sse2_ukernels
LOCAL_SRC_FILES += \
  src/q8avgpool/mp8x9p8q-sse2.c \
  src/q8avgpool/up8x9-sse2.c \
  src/q8avgpool/up8xm-sse2.c \
  src/q8conv/4x4c2-sse2.c \
  src/q8dwconv/mp8x25-sse2.c \
  src/q8dwconv/up8x9-sse2.c \
  src/q8gavgpool/mp8x7p7q-sse2.c \
  src/q8gavgpool/up8x7-sse2.c \
  src/q8gavgpool/up8xm-sse2.c \
  src/q8gemm/4x4c2-sse2.c \
  src/q8vadd/sse2.c \
  src/u8clamp/sse2.c \
  src/u8lut32norm/scalar.c \
  src/u8maxpool/16x9p8q-sse2.c \
  src/u8maxpool/sub16-sse2.c \
  src/u8rmax/sse2.c \
  src/x8lut/scalar.c \
  src/x8zip/x2-sse2.c \
  src/x8zip/x3-sse2.c \
  src/x8zip/x4-sse2.c \
  src/x8zip/xm-sse2.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=c99 -Wall -O2
LOCAL_STATIC_LIBRARIES := cpuinfo FP16 fxdiv
include $(BUILD_STATIC_LIBRARY)
endif # x86 or x86_64

include $(CLEAR_VARS)
LOCAL_MODULE = qnnpack_exec
LOCAL_SRC_FILES := \
  src/indirection.c \
  src/operator-run.c
LOCAL_C_INCLUDES := $(LOCAL_PATH)/include $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=c99 -Wall -O2
ifeq ($(NDK_DEBUG),1)
LOCAL_CFLAGS += -DPYTORCH_QNNP_LOG_LEVEL=5
else
LOCAL_CFLAGS += -DPYTORCH_QNNP_LOG_LEVEL=0
endif
LOCAL_STATIC_LIBRARIES := clog cpuinfo FP16 pthreadpool_interface fxdiv
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := qnnpack
LOCAL_SRC_FILES := \
  src/init.c \
  src/add.c \
  src/average-pooling.c \
  src/channel-shuffle.c \
  src/clamp.c \
  src/convolution.c \
  src/deconvolution.c \
  src/fully-connected.c \
  src/global-average-pooling.c \
  src/leaky-relu.c \
  src/max-pooling.c \
  src/sigmoid.c \
  src/softargmax.c \
  src/operator-delete.c
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_C_INCLUDES := $(LOCAL_EXPORT_C_INCLUDES) $(LOCAL_PATH)/src
LOCAL_CFLAGS := -std=c99 -Wall
ifeq (,$(findstring 4.9,$(NDK_TOOLCHAIN)))
# Clang compiler supports -Oz
LOCAL_CFLAGS += -Oz
else
# gcc-4.9 compiler supports only -Os
LOCAL_CFLAGS += -Os
endif
ifeq ($(NDK_DEBUG),1)
LOCAL_CFLAGS += -DPYTORCH_QNNP_LOG_LEVEL=5
else
LOCAL_CFLAGS += -DPYTORCH_QNNP_LOG_LEVEL=0
endif
LOCAL_STATIC_LIBRARIES := clog cpuinfo pthreadpool_interface qnnpack_exec
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),x86 x86_64))
LOCAL_STATIC_LIBRARIES += qnnpack_sse2_ukernels
endif # x86 or x86_64
ifeq ($(TARGET_ARCH_ABI),$(filter $(TARGET_ARCH_ABI),armeabi armeabi-v7a))
LOCAL_STATIC_LIBRARIES += qnnpack_aarch32_neon_ukernels
endif # armeabi or armeabi-v7a
ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
LOCAL_STATIC_LIBRARIES += qnnpack_aarch64_neon_ukernels
endif # arm64-v8a
include $(BUILD_STATIC_LIBRARY)

$(call import-add-path,$(LOCAL_PATH)/deps)

$(call import-module,clog/jni)
$(call import-module,FP16/jni)
$(call import-module,cpuinfo/jni)
$(call import-module,pthreadpool/jni)
$(call import-module,fxdiv/jni)
