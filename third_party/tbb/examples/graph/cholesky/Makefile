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
PROG=cholesky
ARGS=4 2

# The C++ compiler
ifneq (,$(shell which icpc 2>/dev/null))
CXX=icpc
endif # which icpc

ifeq ($(shell uname), Darwin)
override CXXFLAGS += -Wl,-rpath,$(TBBROOT)/lib  -Wl,-rpath,$(MKLROOT)/lib
# MKL support
override MKL_LIBS += -lmkl_core -lmkl_sequential -lmkl_intel
else
ifeq ($(shell uname), Linux)
LIBS+= -lrt 
endif
# MKL support
override MKL_LIBS += -lmkl_rt
endif

# C++0x support
override CXXFLAGS += -std=c++0x

all:	release test

release: *.cpp
	$(CXX) -O2  -DNDEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbb $(MKL_LIBS) $(LIBS)

debug: *.cpp
	$(CXX) -O0 -g -DTBB_USE_DEBUG $(CXXFLAGS) -o $(PROG) $^ -ltbb_debug $(MKL_LIBS) $(LIBS)

clean:
	$(RM) $(PROG) *.o *.d

test:
	$(run_cmd) ./$(PROG) $(ARGS)
