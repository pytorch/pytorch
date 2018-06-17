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
PROG=convex_hull_bench
ARGS=
PERF_RUN_ARGS = silent auto 40000000
LIGHT_ARGS = 4 400

# Trying to find if icl.exe is set
CXX1 = $(TBB_CXX)-
CXX2 = $(CXX1:icl.exe-=icl.exe)
CXX  = $(CXX2:-=cl.exe)

# The C++ compiler options
# Below, _SECURE_SCL=0 is for VC9 and earlier, and _ITERATOR_DEBUG_LEVEL=0 is for VC10 and later
MYCXXFLAGS = /TP /EHsc /W3 /nologo /D _CONSOLE /D _MBCS /D WIN32 /D _ITERATOR_DEBUG_LEVEL=0 /D _SECURE_SCL=0 $(CXXFLAGS)
MYLDFLAGS =/INCREMENTAL:NO /NOLOGO /DEBUG /FIXED:NO $(LDFLAGS)

all: release test
release: compiler_check
	$(CXX) convex_hull_sample.cpp /MD /O2 /D NDEBUG $(MYCXXFLAGS) /link tbb.lib $(LIBS) $(MYLDFLAGS) /OUT:convex_hull_sample.exe
	$(CXX) convex_hull_bench.cpp /MD /O2 /D NDEBUG $(MYCXXFLAGS) /link tbb.lib $(LIBS) $(MYLDFLAGS) /OUT:convex_hull_bench.exe
debug: compiler_check
	$(CXX) convex_hull_sample.cpp /MDd /Od /Zi /D TBB_USE_DEBUG /D _DEBUG $(MYCXXFLAGS) /link tbb_debug.lib $(LIBS) $(MYLDFLAGS) /OUT:convex_hull_sample.exe
	$(CXX) convex_hull_bench.cpp /MDd /Od /Zi /D TBB_USE_DEBUG /D _DEBUG $(MYCXXFLAGS) /link tbb_debug.lib $(LIBS) $(MYLDFLAGS) /OUT:convex_hull_bench.exe
clean:
	@cmd.exe /C del convex_hull*.exe *.obj *.?db *.manifest
test:
	$(PROG) $(ARGS)
perf_build: release

perf_run:
	convex_hull_sample $(PERF_RUN_ARGS)

light_test:
	$(PROG) $(LIGHT_ARGS)

compiler_check:
	@echo compiler_test>compiler_test && @$(CXX) /E compiler_test >nul 2>&1  || echo "$(CXX) command not found. Check if CXX=$(CXX) is set properly"
	@cmd.exe /C del compiler_test
