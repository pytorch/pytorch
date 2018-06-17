# Copyright (c) 2017-2018 Intel Corporation
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


.NOTPARALLEL:

tbb_root ?= ../..
BUILDING_PHASE=0
TEST_RESOURCE = $(RML.RES)
include $(tbb_root)/build/common.inc
DEBUG_SUFFIX=$(findstring _debug,_$(cfg))

ifneq (linux,$(target))
$(error "IPC RML is supported on Linux only")
endif

.PHONY: default rml test clean

# default target
default: rml test

RML_ROOT ?= $(tbb_root)/src/rml
RML_SERVER_ROOT = $(RML_ROOT)/server
# TODO: new API needs to be added for this server, exposing everything
RML.DEF =

VPATH = $(tbb_root)/src/tbb $(tbb_root)/src/tbb/$(ASSEMBLY_SOURCE)
VPATH += $(tbb_root)/python/rml $(RML_ROOT)/test $(tbb_root)/src/test
VPATH += $(tbb_root)/src/rml/client

include $(tbb_root)/build/common_rules.inc

#--------------------------------------------------------------------------
# Define rules for making the RML server shared library and client objects.
#--------------------------------------------------------------------------

# Object files that make up RML server
RML_SERVER.OBJ = ipc_server.$(OBJ)

# Object files that RML clients need
RML_TBB_CLIENT.OBJ ?= ipc_utils.$(OBJ)
RML.OBJ = $(RML_SERVER.OBJ) $(RML_TBB_CLIENT.OBJ)
ifeq (windows,$(tbb_os))
RML_ASM.OBJ = $(if $(findstring intel64,$(arch)),$(TBB_ASM.OBJ))
endif
ifeq (linux,$(tbb_os))
RML_ASM.OBJ = $(if $(findstring ia64,$(arch)),$(TBB_ASM.OBJ))
endif

RML_TBB_DEP= cache_aligned_allocator_rml.$(OBJ) dynamic_link_rml.$(OBJ) tbb_misc_rml.$(OBJ) tbb_misc_ex_rml.$(OBJ)
TBB_DEP_NON_RML_TEST?= cache_aligned_allocator_rml.$(OBJ) dynamic_link_rml.$(OBJ) $(RML_ASM.OBJ) tbb_misc_rml.$(OBJ) tbb_misc_ex_rml.$(OBJ)
ifeq ($(cfg),debug)
RML_TBB_DEP+= spin_mutex_rml.$(OBJ)
TBB_DEP_RML_TEST?= $(RML_ASM.OBJ) tbb_misc_rml.$(OBJ)
else
TBB_DEP_RML_TEST?= $(RML_ASM.OBJ)
endif
LIBS += $(LIBDL)
TBB_DEP_RML_TEST =  rml_tbb.$(OBJ) dynamic_link_rml.$(OBJ)

INCLUDES += $(INCLUDE_KEY)$(RML_ROOT)/include $(INCLUDE_KEY).
T_INCLUDES = $(INCLUDES) $(INCLUDE_KEY)$(tbb_root)/src/test $(INCLUDE_KEY)$(RML_SERVER_ROOT)

# Suppress superfluous warnings for RML compilation
R_CPLUS_FLAGS =  $(subst DO_ITT_NOTIFY,DO_ITT_NOTIFY=0,$(CPLUS_FLAGS)) $(WARNING_SUPPRESS) \
		 $(DEFINE_KEY)TBB_USE_THREADING_TOOLS=0 $(DEFINE_KEY)__TBB_RML_STATIC=1 $(DEFINE_KEY)__TBB_NO_IMPLICIT_LINKAGE=1

%.$(OBJ): %.cpp
	$(CPLUS) $(COMPILE_ONLY) $(R_CPLUS_FLAGS) $(PIC_KEY) $(DSE_KEY) $(INCLUDES) $<

tbb_misc_rml.$(OBJ) $(RML_SERVER.OBJ): version_string.ver

RML_TEST.OBJ = test_job_automaton.$(OBJ) test_thread_monitor.$(OBJ) test_rml_tbb.$(OBJ)

$(RML_TBB_DEP): %_rml.$(OBJ): %.cpp
	$(CPLUS) $(COMPILE_ONLY) $(OUTPUTOBJ_KEY)$@ $(R_CPLUS_FLAGS) $(PIC_KEY) $(DSE_KEY) $(INCLUDES) $<

$(RML_TEST.OBJ): %.$(OBJ): %.cpp
	$(CPLUS) $(COMPILE_ONLY) $(R_CPLUS_FLAGS) $(PIC_KEY) $(T_INCLUDES) $<

ifneq (,$(RML.DEF))
rml.def: $(RML.DEF)
	$(CPLUS) $(PREPROC_ONLY) $< $(CPLUS_FLAGS) $(INCLUDES) > $@

LIB_LINK_FLAGS += $(EXPORT_KEY)rml.def
$(RML.DLL): rml.def
endif

$(RML.DLL): CPLUS_FLAGS += $(SDL_FLAGS)
$(RML.DLL): BUILDING_LIBRARY = $(RML.DLL)
$(RML.DLL): $(RML_TBB_DEP) $(RML.OBJ) $(RML.RES) $(RML_NO_VERSION.DLL) $(RML_ASM.OBJ)
	$(LIB_LINK_CMD) $(LIB_OUTPUT_KEY)$(RML.DLL) $(RML.OBJ) $(RML_TBB_DEP) $(RML_ASM.OBJ) $(RML.RES) $(LIB_LINK_LIBS) $(LIB_LINK_FLAGS)

ifneq (,$(RML_NO_VERSION.DLL))
$(RML_NO_VERSION.DLL):
	echo "INPUT ($(RML.DLL))" > $(RML_NO_VERSION.DLL)
endif

rml: rml_dll
rml_dll: $(RML.DLL)

#------------------------------------------------------
# End of rules for making the RML server shared library
#------------------------------------------------------

#------------------------------------------------------
# Define rules for making the RML unit tests
#------------------------------------------------------

add_debug=$(basename $(1))_debug$(suffix $(1))
cross_suffix=$(if $(crosstest),$(if $(DEBUG_SUFFIX),$(subst _debug,,$(1)),$(call add_debug,$(1))),$(1))

RML_TESTS = test_job_automaton.$(TEST_EXT) test_thread_monitor.$(TEST_EXT)
RML_CUSTOM_TESTS = test_rml_tbb.$(TEST_EXT)

test_rml_tbb.$(TEST_EXT): test_rml_tbb.$(OBJ) $(RML_TBB_CLIENT.OBJ) $(TBB_DEP_RML_TEST)
	$(CPLUS) $(OUTPUT_KEY)$@ $(CPLUS_FLAGS) test_rml_tbb.$(OBJ) $(RML_TBB_CLIENT.OBJ) $(TBB_DEP_RML_TEST) $(LIBS) $(LINK_FLAGS)

$(RML_TESTS): %.$(TEST_EXT): %.$(OBJ) $(TBB_DEP_NON_RML_TEST)
	$(CPLUS) $(OUTPUT_KEY)$@ $(CPLUS_FLAGS) $< $(TBB_DEP_NON_RML_TEST) $(LIBS) $(LINK_FLAGS)

export IPC_ENABLE=1
### run_cmd is usually empty
test: $(call cross_suffix,$(RML.DLL)) $(TEST_PREREQUISITE) $(RML_TESTS) $(RML_CUSTOM_TESTS)
	$(run_cmd) ./test_job_automaton.$(TEST_EXT) $(args)
	$(run_cmd) ./test_thread_monitor.$(TEST_EXT) $(args)
#TODO:	$(run_cmd) ./test_rml_tbb.$(TEST_EXT) $(args)
#TODO:	IPC_ENABLE=1 LD_PRELOAD=$(abspath libirml.so.1) $(MAKE) -rf $(tbb_root)/src/Makefile cfg=release tbb_test_release

#------------------------------------------------------
# End of rules for making the TBBMalloc unit tests
#------------------------------------------------------

# Include automatically generated dependencies
-include *.d

clean:
	-rm -rf *.o *.so* *.d *.def version_string.ver
	-rm -rf $(work_dir)_release/libirml*
	-rm -rf $(work_dir)_debug/libirml*
