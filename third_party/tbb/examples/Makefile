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

# This file should be always located and called in $(tbb_root)/examples

# Usage:
#   make [all] [clean] [release] [debug] [test]
#      executes specified targets for all examples.
#   make {example's dir}/{target}
#      calls specified example with specified target.
#      For instance: make task/tree_sum/all

tbb_root?=..
BUILDING_PHASE:=0
include ../build/common.inc

#check, if tbb_root is not absolute path (the filter keeps only /* paths)
ifeq ($(filter /% $(SLASH)%, $(subst :, ,$(tbb_root)) ),)
    # also changes related variables like work_dir
    override tbb_root := $(CWD)$(SLASH)..
    export TBBROOT := $(tbb_root)
endif

override CXXFLAGS += $(CXX_STD_FLAGS)

# give stdver priotity over autodetection
ifneq (,$(stdver))
    CXX0XFLAGS=-std=$(stdver)
endif
# empty CXX0XFLAGS exported as well, to have control over C++ autodetection
export CXX0XFLAGS

ifeq ($(tbb_os),windows)
    ifeq ($(UNIXMODE),1)
        EXAMPLE_MAKEFILE = Makefile
        EXAMPLES = $(wildcard */*/$(EXAMPLE_MAKEFILE))
        ifeq ($(compiler),gcc)
            override CXXFLAGS += -Wl,--enable-auto-import
        endif
        export CPATH := $(CPATH);$(tbb_root)/include
        export LIBRARY_PATH := $(LIBRARY_PATH);$(work_dir)_release;$(work_dir)_debug
        export RM = cmd /C del /Q /F
    else
        EXAMPLE_MAKEFILE = Makefile.windows
        EXAMPLES = $(wildcard */*/$(EXAMPLE_MAKEFILE))
        EXAMPLE_MAKEFILE := $(if $(EXAMPLES),Makefile.windows,Makefile)
        export INCLUDE := $(tbb_root)$(SLASH)include;$(INCLUDE)
        export LIB := $(work_dir)_release;$(work_dir)_debug;$(LIB)
    endif
    work_dir := $(subst /,$(SLASH),$(work_dir))
    export PATH := $(work_dir)_release;$(work_dir)_debug;$(PATH)
    export TBB_ARCH_PLATFORM = $(arch)\$(runtime)
    export TBB_TARGET_ARCH = $(arch)
else
    EXAMPLE_MAKEFILE = Makefile
    EXAMPLES := $(wildcard */*/$(EXAMPLE_MAKEFILE))
    # platform-specific settings
    ifeq ($(arch),ia64)
        override CXXFLAGS += $(PIC_KEY)
    endif
    ifneq ($(filter suncc gcc clang,$(compiler)),)
        ifeq ($(compiler),suncc)
            override CXXFLAGS += -I$(tbb_root)$(SLASH)include -library=stlport4 -M$(tbb_root)/build/suncc.map.pause -erroff=unassigned,attrskipunsup,badargtype2w,badbinaryopw,wbadasg,wvarhidemem,wbadinit
        endif
        ifeq ($(arch),intel64)
            override CXXFLAGS += -m64
        endif
        ifeq ($(arch),ia32)
            override CXXFLAGS += -m32
        endif
        ifeq ($(arch),ppc64)
            override CXXFLAGS += -m64
        endif
        ifeq ($(arch),ppc32)
            override CXXFLAGS += -m32
        endif
        ifeq ($(arch),sparc)
            override CXXFLAGS += -mcpu=v9 -m64
        endif
    endif
    ifeq ($(compiler),xl)
        # -qsuppress=1540-0198 suppresses warnings like "1540-0198 (W) The omitted keyword "private" is assumed for base class "no_copy"."
        # -qsuppress=1540-1401 suppresses warnings like "1540-1401 (I) An unknown "pragma ivdep" is specified."
        override CXXFLAGS += -I$(tbb_root)$(SLASH)include -qsuppress=1540-0198:1540-1401 -L$(work_dir)_release -L$(work_dir)_debug
        ifeq ($(arch),intel64)
            override CXXFLAGS += -q64
        endif
        ifeq ($(arch),ia32)
            override CXXFLAGS += -q32
        endif
        ifeq ($(arch),ppc64)
            override CXXFLAGS += -q64
        endif
        ifeq ($(arch),ppc32)
            override CXXFLAGS += -q32
        endif
    endif
    ifeq ($(tbb_os),macos)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(work_dir)_release:$(work_dir)_debug
        override CXXFLAGS += -Wl,-rpath,$(work_dir)_release -Wl,-rpath,$(work_dir)_debug
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(work_dir)_release:$(work_dir)_debug
        ifeq ($(findstring mic,$(offload) $(target)),mic)
            mic_path=$(tbb_build_dir)$(SLASH)$(mic_tbb_build_prefix)
            export MIC_LIBRARY_PATH := $(mic_path)_release:$(mic_path)_debug:$(MIC_LIBRARY_PATH)
            export MIC_LD_LIBRARY_PATH := $(mic_path)_release:$(mic_path)_debug:$(MIC_LD_LIBRARY_PATH)
        else
            # -L necessary for non-native compilers which don't search $LIBRARY_PATH
            override CXXFLAGS += -L$(work_dir)_release -L$(work_dir)_debug
        endif
    endif
    export LIBS
    export LIBRARY_PATH := $(LIBRARY_PATH):$(work_dir)_release:$(work_dir)_debug
    export CPATH := $(CPATH):$(tbb_root)/include
endif

export CXXFLAGS
COMMON_TARGETS := all clean release debug test perf_build perf_run
# list of directories of examples
EXAMPLES_DIRS := $(foreach T,$(EXAMPLES),$(dir $(T)))
# targets to explicitly call example have format: {example's dir}/{example's target}
EXAMPLES_TARGETS := $(foreach T,$(COMMON_TARGETS),$(addsuffix $(T),$(EXAMPLES_DIRS)))

.PHONY: $(COMMON_TARGETS) $(EXAMPLES_TARGETS)

.DEFAULT: all

# execute standard targets for all examples
$(COMMON_TARGETS):: % : $(addsuffix %,$(EXAMPLES_DIRS))

# proxy rule for calling appropriate example
$(EXAMPLES_TARGETS)::
	@echo ------------------------ $@ ------------------------
	-$(MAKE) -C $(@D)  -f $(EXAMPLE_MAKEFILE) $(notdir $@) CXX="$(CPLUS)" $(if $(run_cmd),run_cmd="$(run_cmd)",) $(if $(args),ARGS="$(args)",)  $(if $(UI),UI="$(UI)")

printenv:
ifeq ($(tbb_os),windows)
	@cmd /C set PATH
	@cmd /C set LIB
	@cmd /C set INCLUDE
else
	@env | grep PATH
endif

