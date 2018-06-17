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

# Common Makefile that builds and runs example.

# Just specify your program basename
PROG=stereo
ARGS=
PERF_RUN_ARGS=

# Trying to find if icl.exe is set
CXX1 = $(TBB_CXX)-
CXX2 = $(CXX1:icl.exe-=icl.exe)
CXX  = $(CXX2:-=cl.exe)

# TBB libs
TBBLIB=tbb.lib
TBBLIB_DEBUG=tbb_debug.lib

# OpenCL lib
OPENCL_LIB=OpenCL.lib

# The C++ compiler options
MYCXXFLAGS=/TP /EHsc /W3 /nologo /D _CONSOLE /D _MBCS /D WIN32 /D _CRT_SECURE_NO_DEPRECATE /D _CRT_SECURE_NO_WARNINGS /D _SCL_SECURE_NO_WARNINGS $(CXXFLAGS)
MYLDFLAGS=/INCREMENTAL:NO /NOLOGO /DEBUG $(LDFLAGS)

all: release test
release:
	$(CXX) *.cpp /MD /O2 /D NDEBUG $(MYCXXFLAGS) /link $(TBBLIB) $(OPENCL_LIB) $(MYLDFLAGS) /OUT:$(PROG).exe
debug:
	$(CXX) *.cpp /MDd /Od /Zi /D TBB_USE_DEBUG /D _DEBUG $(MYCXXFLAGS) /link $(TBBLIB_DEBUG) $(OPENCL_LIB) $(MYLDFLAGS) /OUT:$(PROG).exe
clean:
	@cmd.exe /C del $(PROG).exe *.obj *.?db *.manifest
test:
	$(PROG) $(ARGS)
compiler_check:
	@$(CXX) >nul 2>&1 || echo "$(CXX) command not found. Check if CXX=$(CXX) is set properly"

perf_build: release

perf_run:
	$(PROG) $(PERF_RUN_ARGS)
