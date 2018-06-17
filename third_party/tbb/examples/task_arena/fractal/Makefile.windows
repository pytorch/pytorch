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
PROG=Fractal
ARGS=auto
PERF_RUN_ARGS=auto 1 1000000 silent
LIGHT_ARGS=auto 1 1000

# Trying to find if icl.exe is set
CXX1 = $(TBB_CXX)-
CXX2 = $(CXX1:icl.exe-=icl.exe)
CXX  = $(CXX2:-=cl.exe)
# Uncomment one of next lines to choose user interface type (console, gdiplus, direct draw)
#UI = con
UI = gdi
#UI = dd

# Machine architecture, auto-detected from TBB_TARGET_ARCH by default
# Use XARCH variable to change it. See index.html for more information
ARCH0 = $(TBB_TARGET_ARCH)-
ARCH1 = $(ARCH0:ia32-=x86)
ARCH2 = $(ARCH1:intel64-=AMD64)
XARCH = $(ARCH2:-=x86)

MAKEINC = ../../common/gui/Makefile.win
SOURCES = fractal.cpp main.cpp 

all: release test
release: compiler_check
	@$(MAKE) -f $(MAKEINC) UI=$(UI) CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" LFLAGS="$(LDFLAGS) tbb.lib $(LIBS)" XARCH=$(XARCH) RCNAME=gui SOURCE="$(SOURCES)" EXE=$(PROG).exe build_one
debug: compiler_check
	@$(MAKE) -f $(MAKEINC) UI=$(UI) DEBUG=_debug CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS) /D TBB_USE_DEBUG" LFLAGS="$(LDFLAGS) tbb_debug.lib $(LIBS)" XARCH=$(XARCH) RCNAME=gui SOURCE="$(SOURCES)" EXE=$(PROG).exe build_one
clean:
	@cmd.exe /C del $(PROG).exe *.obj *.?db *.manifest msvs\gui.res
test:
	$(PROG) $(ARGS)
light_test:
	$(PROG) $(LIGHT_ARGS)

perf_build: compiler_check
	@$(MAKE) -f $(MAKEINC) UI=con CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" LFLAGS="$(LDFLAGS) tbb.lib $(LIBS)" XARCH=$(XARCH) RCNAME=gui SOURCE="$(SOURCES) " EXE=$(PROG).exe build_one
perf_run:
	$(PROG) $(PERF_RUN_ARGS)

compiler_check:
	@echo compiler_test>compiler_test && @$(CXX) /E compiler_test >nul 2>&1  || echo "$(CXX) command not found. Check if CXX=$(CXX) is set properly"
	@cmd.exe /C del compiler_test

