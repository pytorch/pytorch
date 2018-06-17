# Copyright (c) 2005-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

export tbb_root?=$(NDK_PROJECT_PATH)

ifeq (armeabi-v7a,$(APP_ABI))
	export SYSROOT:=$(NDK_ROOT)/platforms/$(APP_PLATFORM)/arch-arm
else ifeq (arm64-v8a,$(APP_ABI))
	export SYSROOT:=$(NDK_ROOT)/platforms/$(APP_PLATFORM)/arch-arm64
else
	export SYSROOT:=$(NDK_ROOT)/platforms/$(APP_PLATFORM)/arch-$(APP_ABI)
endif

ifeq (windows,$(tbb_os))
	export CPATH_SEPARATOR :=;
else
	export CPATH_SEPARATOR :=:
endif

export ANDROID_NDK_ROOT:=$(NDK_ROOT)
export ndk_version:=$(lastword $(subst -, ,$(ANDROID_NDK_ROOT)))
ndk_version:= $(firstword $(subst /, ,$(ndk_version)))
ndk_version:= $(firstword $(subst \, ,$(ndk_version)))

ifeq (clang,$(compiler))
	ifeq (,$(findstring $(ndk_version),ifeq (,$(findstring $(ndk_version),$(foreach v, 7 8 9 10 11 12,r$(v) r$(v)b r$(v)c r$(v)d r$(v)e)))))
		TBB_RTL :=llvm-libc++
	else
		TBB_RTL :=llvm-libc++/libcxx
	endif
	TBB_RTL_LIB :=llvm-libc++
	TBB_RTL_FILE :=libc++_shared.so
else
	TBB_RTL :=gnu-libstdc++/$(NDK_TOOLCHAIN_VERSION)
	TBB_RTL_LIB :=$(TBB_RTL)
	TBB_RTL_FILE :=libgnustl_shared.so
endif

export CPATH := $(SYSROOT)/usr/include$(CPATH_SEPARATOR)$(NDK_ROOT)/sources/cxx-stl/$(TBB_RTL)/include$(CPATH_SEPARATOR)$(NDK_ROOT)/sources/cxx-stl/$(TBB_RTL)/libs/$(APP_ABI)/include$(CPATH_SEPARATOR)$(NDK_ROOT)/sources/android/support/include

LIB_STL_ANDROID_DIR := $(NDK_ROOT)/sources/cxx-stl/$(TBB_RTL_LIB)/libs/$(APP_ABI)
#LIB_STL_ANDROID is required to be set up for copying Android specific library to a device next to test
export LIB_STL_ANDROID := $(LIB_STL_ANDROID_DIR)/$(TBB_RTL_FILE)
export CPLUS_LIB_PATH := $(SYSROOT)/usr/lib -L$(LIB_STL_ANDROID_DIR)
export target_os_version:=$(APP_PLATFORM)
export tbb_tool_prefix:=$(TOOLCHAIN_PREFIX)
export TARGET_CXX
export TARGET_CC
export TARGET_CFLAGS

include $(NDK_PROJECT_PATH)/src/Makefile
