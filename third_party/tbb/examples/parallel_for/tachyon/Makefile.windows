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

# The original source for this example is
# Copyright (c) 1994-2008 John E. Stone
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

# Common Makefile that builds and runs example.

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

# The default runtime arguments
ARGS = dat\balls.dat
PERF_RUN_ARGS = silent dat\balls3.dat
LIGHT_ARGS=dat\model2.dat

# Add these for tbb/tbb1d release builds
# /GL is a workaround to prevent run-time crash when built by VS2010
CXXFLAGS_TBB_NDEBUG = $(CXXFLAGS) /GL
LIBS_TBB_NDEBUG = tbb.lib $(LIBS)

# Add these for tbb/tbb1d debug builds
CXXFLAGS_TBB_DEBUG = $(CXXFLAGS) /D TBB_USE_DEBUG
LIBS_TBB_DEBUG =  tbb_debug.lib $(LIBS)


MAKEINC = ../../common/gui/Makefile.win
# scr/main.cpp src/tachyon_video.cpp cannot be included to the SOURCE_COMMON list since it depends on UI and /subsystem which is not specified for common SOURCE build
SOURCE = src/main.cpp src/tachyon_video.cpp
SOURCE_COMMON = src/pthread.cpp  src/api.cpp src/apigeom.cpp src/apitrigeom.cpp src/bndbox.cpp src/box.cpp src/camera.cpp src/coordsys.cpp src/cylinder.cpp src/extvol.cpp src/global.cpp src/grid.cpp src/imageio.cpp src/imap.cpp src/intersect.cpp src/jpeg.cpp src/light.cpp src/objbound.cpp src/parse.cpp src/plane.cpp src/ppm.cpp src/quadric.cpp src/render.cpp src/ring.cpp src/shade.cpp src/sphere.cpp src/texture.cpp src/tgafile.cpp src/trace_rest.cpp src/triangle.cpp src/ui.cpp src/util.cpp src/vector.cpp src/vol.cpp

# Targets
all: build run
release: build
debug: build_debug
test: run

build: build_serial build_tbb1d build_tbb
build_debug: build_serial_debug build_tbb1d_debug build_tbb_debug
run: run_serial run_tbb1d run_tbb

serial: build_serial run_serial
serial_debug: build_serial_debug run_serial
tbb: build_tbb run_tbb
tbb_debug: build_tbb_debug run_tbb
tbb1d: build_tbb1d run_tbb1d
tbb1d_debug: build_tbb1d_debug run_tbb1d

CXXFLAGS_COMMON = /c /nologo /EHsc /Zc:forScope /D WIN32 /D _MBCS /D _CRT_SECURE_NO_DEPRECATE /Foobj/ /MP $(CXXFLAGS)
CXXFLAGS_NDEBUG = /MD /O2 /Ot /Gy /D NDEBUG
CXXFLAGS_DEBUG = /MDd /Od /Zi /D _DEBUG
LIB_LINK_FLAGS = /nologo /machine:$(XARCH)

tachyon_common.lib:
	@cmd.exe /C if not exist obj mkdir obj
	$(CXX) $(CXXFLAGS_NDEBUG) $(CXXFLAGS_COMMON) $(SOURCE_COMMON)
	LIB $(LIB_LINK_FLAGS) obj/*.obj /OUT:$@
	@cmd.exe /C if exist obj rmdir /S /Q obj

tachyon_common_debug.lib:
	@cmd.exe /C if not exist obj mkdir obj
	$(CXX) $(CXXFLAGS_DEBUG) $(CXXFLAGS_COMMON) $(SOURCE_COMMON)
	LIB $(LIB_LINK_FLAGS) obj/*.obj /OUT:$@
	@cmd.exe /C if exist obj rmdir /S /Q obj

build_serial: tachyon_common.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.serial.cpp $(SOURCE) tachyon_common.lib" EXE=tachyon.serial.exe RCNAME=gui UI=$(UI) CXX="$(CXX)" CXXFLAGS="/GL $(CXXFLAGS)" XARCH=$(XARCH) build_one
build_serial_debug: tachyon_common_debug.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.serial.cpp $(SOURCE) tachyon_common_debug.lib" EXE=tachyon.serial.exe RCNAME=gui UI=$(UI) DEBUG=_debug CXX="$(CXX)" CXXFLAGS=$(CXXFLAGS) XARCH=$(XARCH) build_one
run_serial:
	-.\tachyon.serial.exe $(ARGS)

build_tbb: tachyon_common.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.tbb.cpp $(SOURCE) tachyon_common.lib" EXE=tachyon.tbb.exe RCNAME=gui UI=$(UI) CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS_TBB_NDEBUG)" LFLAGS="$(LIBS_TBB_NDEBUG)" XARCH=$(XARCH) build_one
build_tbb_debug: tachyon_common_debug.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.tbb.cpp $(SOURCE) tachyon_common_debug.lib" EXE=tachyon.tbb.exe RCNAME=gui UI=$(UI) DEBUG=_debug CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS_TBB_DEBUG)" LFLAGS="$(LIBS_TBB_DEBUG)" XARCH=$(XARCH) build_one
run_tbb:
	-.\tachyon.tbb.exe $(ARGS)

build_tbb1d: tachyon_common.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.tbb1d.cpp $(SOURCE) tachyon_common.lib" EXE=tachyon.tbb1d.exe RCNAME=gui UI=$(UI) CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS_TBB_NDEBUG)" LFLAGS="$(LIBS_TBB_NDEBUG)" XARCH=$(XARCH) build_one
build_tbb1d_debug: tachyon_common_debug.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.tbb1d.cpp $(SOURCE) tachyon_common_debug.lib" EXE=tachyon.tbb1d.exe RCNAME=gui UI=$(UI) DEBUG=_debug CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS_TBB_DEBUG)" LFLAGS="$(LIBS_TBB_DEBUG)" XARCH=$(XARCH) build_one
run_tbb1d:
	-.\tachyon.tbb1d.exe $(ARGS)


clean:
	@cmd.exe /C del tachyon.* *.manifest *.obj *.lib msvs\gui.res *.?db

perf_build: tachyon_common.lib
	@$(MAKE) -f $(MAKEINC) SOURCE="src/trace.tbb.cpp $(SOURCE) tachyon_common.lib" EXE=tachyon.tbb.exe RCNAME=gui UI=con CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS_TBB_NDEBUG)" LFLAGS="$(LIBS_TBB_NDEBUG)" XARCH=$(XARCH) build_one
perf_run:
	-.\tachyon.tbb.exe $(PERF_RUN_ARGS)

light_test:
	-.\tachyon.tbb.exe $(LIGHT_ARGS)
