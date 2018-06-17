# Copyright (c) 2016-2018 Intel Corporation
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
PROG=matmult
ARGS=

# icpc by default
ifneq (,$(shell which icc 2>/dev/null))
    CXX=icpc
endif

# autodetect C++11 support
include ../../common/examples-common.inc

ifeq ($(shell uname), Linux)
LIBS+= -lrt -lpthread -lcilkrts
else ifeq ($(shell uname), Darwin)
override CXXFLAGS += -Wl,-rpath,$(TBBROOT)/lib
endif

all: release test

release:
	$(CXX) -O2 -DNDEBUG *.cpp $(CXXFLAGS) -o $(PROG) -ltbb $(LIBS) $(CXX0XFLAGS)

debug:
	$(CXX) -O0 -g -DTBB_USE_DEBUG *.cpp $(CXXFLAGS) -o $(PROG) -ltbb_debug $(LIBS)  $(CXX0XFLAGS)

clean:
	$(RM) $(PROG) *.o *.d

test:
	$(run_cmd) ./$(PROG) $(ARGS)

perf_build: release

perf_run:
	$(run_cmd) ./$(PROG)
