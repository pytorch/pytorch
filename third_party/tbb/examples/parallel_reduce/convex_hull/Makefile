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

# GNU Makefile that builds and runs example.
run_cmd=
PROG=convex_hull_bench
ARGS=
PERF_RUN_ARGS = silent auto 40000000
LIGHT_ARGS = 4 400

# The C++ compiler
ifneq (,$(shell which icc 2>/dev/null))
CXX=icc
endif # icc

ifeq ($(shell uname), Linux)
ifeq ($(target), android)
LIBS+= --sysroot=$(SYSROOT)
run_cmd=../../common/android.linux.launcher.sh
else
LIBS+= -lrt 
endif
else ifeq ($(shell uname), Darwin)
override CXXFLAGS += -Wl,-rpath,$(TBBROOT)/lib
endif

all:	release test

release: *.cpp
	$(CXX) -O2 -DNDEBUG $(CXXFLAGS) -o convex_hull_sample convex_hull_sample.cpp -ltbb $(LIBS)
	$(CXX) -O2 -DNDEBUG $(CXXFLAGS) -o convex_hull_bench convex_hull_bench.cpp -ltbb $(LIBS)

debug: *.cpp
	$(CXX) -O0 -g -DTBB_USE_DEBUG $(CXXFLAGS) -o convex_hull_sample convex_hull_sample.cpp -ltbb_debug $(LIBS)
	$(CXX) -O0 -g -DTBB_USE_DEBUG $(CXXFLAGS) -o convex_hull_bench convex_hull_bench.cpp -ltbb_debug $(LIBS)

clean:
	$(RM) convex_hull_bench convex_hull_sample *.o *.d

perf_build: release

perf_run:
	$(run_cmd) ./convex_hull_sample $(PERF_RUN_ARGS)

test:
	$(run_cmd) ./$(PROG) $(ARGS)

light_test:
	$(run_cmd) ./$(PROG) $(LIGHT_ARGS)
