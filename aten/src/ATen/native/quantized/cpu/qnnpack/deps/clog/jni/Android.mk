# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

LOCAL_PATH := $(call my-dir)/..

include $(CLEAR_VARS)
LOCAL_MODULE := clog
LOCAL_SRC_FILES := src/clog.c
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/include
LOCAL_C_INCLUDES := $(LOCAL_EXPORT_C_INCLUDES)
LOCAL_CFLAGS := -std=c99 -Wall
ifeq (,$(findstring 4.9,$(NDK_TOOLCHAIN)))
# Clang compiler supports -Oz
LOCAL_CFLAGS += -Oz
else
# gcc-4.9 compiler supports only -Os
LOCAL_CFLAGS += -Os
endif
LOCAL_EXPORT_LDLIBS := -llog
include $(BUILD_STATIC_LIBRARY)
