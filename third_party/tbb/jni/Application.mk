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

ifndef tbb_os

  # Windows sets environment variable OS; for other systems, ask uname
  ifeq ($(OS),)
    OS:=$(shell uname)
    ifeq ($(OS),)
      $(error "Cannot detect operating system")
    endif
    export tbb_os=$(OS)
  endif

  ifeq ($(OS), Windows_NT)
    export tbb_os=windows
  endif
  ifeq ($(OS), Linux)
    export tbb_os=linux
  endif
  ifeq ($(OS), Darwin)
    export tbb_os=macos
  endif

endif

export compiler?=clang
export arch?=ia32
export target?=android

ifeq (ia32,$(arch))
    APP_ABI:=x86
    export TRIPLE:=i686-linux-android
else ifeq (intel64,$(arch))
    APP_ABI:=x86_64
    export TRIPLE:=x86_64-linux-android
else ifeq (arm,$(arch))
    APP_ABI:=armeabi-v7a
    export TRIPLE:=arm-linux-androideabi
else ifeq (arm64,$(arch))
    APP_ABI:=arm64-v8a
    export TRIPLE:=aarch64-linux-android
else
    APP_ABI:=$(arch)
endif

api_version?=21
export API_LEVEL:=$(api_version)
APP_PLATFORM:=android-$(api_version)

ifeq (clang,$(compiler))
    NDK_TOOLCHAIN_VERSION:=clang
    APP_STL:=c++_shared
else
    NDK_TOOLCHAIN_VERSION:=4.9
endif
