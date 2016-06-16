#
# Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
DEBUG ?= 0
BUILDDIR ?= build

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc

NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_52,code=sm_52 \
                -gencode=arch=compute_52,code=compute_52

CXXFLAGS   := -I$(CUDA_INC) -fPIC -fvisibility=hidden
NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -lineinfo -std=c++11 -maxrregcount 96
# Use addprefix so that we can specify more than one path
LDFLAGS    := $(addprefix -L,${CUDA_LIB}) -lcudart

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3
CXXFLAGS  += -O3
else
NVCUFLAGS += -O0 -G
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra
CXXFLAGS  += -Wall -Wextra
else
.SILENT:
endif


NCCL_MAJOR   := 1
NCCL_MINOR   := 5
NCCL_PATCH   := 3
CXXFLAGS  += -DNCCL_MAJOR=$(NCCL_MAJOR) -DNCCL_MINOR=$(NCCL_MINOR) -DNCCL_PATCH=$(NCCL_PATCH)

CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
CXXFLAGS  += -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR)

.PHONY : lib clean debclean test mpitest install
.DEFAULT : lib

INCEXPORTS  := nccl.h
LIBSRCFILES := libwrap.cu core.cu all_gather.cu all_reduce.cu broadcast.cu reduce.cu reduce_scatter.cu
LIBNAME     := libnccl.so

INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))
LIBSONAME  := $(patsubst %,%.$(NCCL_MAJOR),$(LIBNAME))
LIBTARGET  := $(patsubst %,%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH),$(LIBNAME))
LIBLINK    := $(patsubst lib%.so, -l%, $(LIBNAME))
LIBOBJ     := $(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(LIBSRCFILES)))
DEPFILES   := $(patsubst %.o, %.d, $(LIBOBJ)) $(patsubst %, %.d, $(TESTBINS)) $(patsubst %, %.d, $(MPITESTBINS))

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET)

-include $(DEPFILES)

$(LIBDIR)/$(LIBTARGET) : $(LIBOBJ)
	@printf "Linking   %-25s\n" $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LDFLAGS) $(LIBOBJ)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(INCDIR)/%.h : src/%.h
	@printf "Grabbing  %-25s > %-25s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(OBJDIR)/%.o : src/%.cu
	@printf "Compiling %-25s > %-25s\n" $< $@
	mkdir -p $(OBJDIR)
	$(NVCC) -c $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< -o $@
	@$(NVCC) -M $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf $(BUILDDIR)

install : lib
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/* $(PREFIX)/lib/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

#### TESTS ####

TEST_ONLY ?= 0

# Tests depend on lib, except in TEST_ONLY mode.
ifeq ($(TEST_ONLY), 0)
TSTDEP = $(INCTARGETS) $(LIBDIR)/$(LIBTARGET)
endif

NCCL_LIB ?= $(LIBDIR)
NCCL_INC ?= $(INCDIR)

MPI_HOME ?= /usr
MPI_INC ?= $(MPI_HOME)/include
MPI_LIB ?= $(MPI_HOME)/lib
MPIFLAGS   := -I$(MPI_INC) -L$(MPI_LIB) -lmpi

TESTS       := all_gather_test all_reduce_test broadcast_test reduce_test reduce_scatter_test
MPITESTS    := mpi_test

TSTINC     := -I$(NCCL_INC) -Itest/include
TSTLIB     := -L$(NCCL_LIB) $(LIBLINK) $(LDFLAGS)
TSTDIR     := $(BUILDDIR)/test/single
MPITSTDIR  := $(BUILDDIR)/test/mpi
TESTBINS   := $(patsubst %, $(TSTDIR)/%, $(TESTS))
MPITESTBINS:= $(patsubst %, $(MPITSTDIR)/%, $(MPITESTS))

test : $(TESTBINS)

$(TSTDIR)/% : test/single/%.cu $(TSTDEP)
	@printf "Building  %-25s > %-24s\n" $< $@
	mkdir -p $(TSTDIR)
	$(NVCC) $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@ $< $(TSTLIB) -lcuda -lcurand -lnvToolsExt
	@$(NVCC) -M $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< $(TSTLIB) -lcuda -lcurand -lnvToolsExt > $(@:%=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%=%.d.tmp) > $(@:%=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%=%.d)
	@rm -f $(@:%=%.d.tmp)

mpitest : $(MPITESTBINS)

$(MPITSTDIR)/% : test/mpi/%.cu $(TSTDEP)
	@printf "Building  %-25s > %-24s\n" $< $@
	mkdir -p $(MPITSTDIR)
	$(NVCC) $(MPIFLAGS) $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@ $< $(TSTLIB) -lcurand
	@$(NVCC) $(MPIFLAGS) -M $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< $(TSTLIB) -lcurand > $(@:%=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%=%.d.tmp) > $(@:%=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%=%.d)
	@rm -f $(@:%=%.d.tmp)

#### PACKAGING ####

DEB_GEN_IN := $(shell ls debian/*.in)
DEB_GEN    := $(DEB_GEN_IN:.in=)

DEB_REVISION   ?= 1
DEB_TIMESTAMP  := $(shell date -R)

deb : lib $(DEB_GEN)
	@printf "Building Debian package\n"
	debuild -eBUILDDIR -eLD_LIBRARY_PATH -uc -us -d -b
	mkdir -p $(BUILDDIR)/deb/
	mv ../libnccl*.deb $(BUILDDIR)/deb/

debclean :
	rm -f $(DEB_GEN)

debian/% : debian/%.in
	@printf "Generating %-25s > %-24s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${nccl:Debian}/$(DEB_REVISION)/g" \
	    -e "s/\$${nccl:Timestamp}/$(DEB_TIMESTAMP)/g" \
	    $< > $@
