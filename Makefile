#
# Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
KEEP ?= 0
DEBUG ?= 0
PROFAPI ?= 0
BUILDDIR ?= build
BUILDDIR := $(abspath $(BUILDDIR))

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC ?= $(CUDA_HOME)/bin/nvcc

NVCC_GENCODE ?= -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_52,code=sm_52 \
                -gencode=arch=compute_60,code=sm_60\
                -gencode=arch=compute_61,code=sm_61 \
                -gencode=arch=compute_60,code=compute_60

CXXFLAGS   := -I$(CUDA_INC) -fPIC -fvisibility=hidden
NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -lineinfo -std=c++11 -maxrregcount 96
# Use addprefix so that we can specify more than one path
LDFLAGS    := $(addprefix -L,${CUDA_LIB}) -lcudart -lrt

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

ifneq ($(KEEP), 0)
NVCUFLAGS += -keep
endif

ifneq ($(PROFAPI), 0)
CXXFLAGS += -DPROFAPI
endif

NCCL_MAJOR   := 1
NCCL_MINOR   := 3
NCCL_PATCH   := 5
CXXFLAGS  += -DNCCL_MAJOR=$(NCCL_MAJOR) -DNCCL_MINOR=$(NCCL_MINOR) -DNCCL_PATCH=$(NCCL_PATCH)

CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
CXXFLAGS  += -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR)

.PHONY : all lib staticlib clean test mpitest install deb debian debclean forlib fortest forclean
.DEFAULT : all

INCEXPORTS  := nccl.h
LIBSRCFILES := libwrap.cu core.cu all_gather.cu all_reduce.cu broadcast.cu reduce.cu reduce_scatter.cu
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a

INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj

INCTARGETS := $(patsubst %, $(INCDIR)/%, $(INCEXPORTS))
LIBSONAME  := $(patsubst %,%.$(NCCL_MAJOR),$(LIBNAME))
LIBTARGET  := $(patsubst %,%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH),$(LIBNAME))
STATICLIBTARGET := $(STATICLIBNAME)
LIBLINK    := $(patsubst lib%.so, -l%, $(LIBNAME))
LIBOBJ     := $(patsubst %.cu, $(OBJDIR)/%.o, $(filter %.cu, $(LIBSRCFILES)))
DEPFILES   := $(patsubst %.o, %.d, $(LIBOBJ)) $(patsubst %, %.d, $(TESTBINS)) $(patsubst %, %.d, $(MPITESTBINS))

all : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET)

staticlib : $(INCTARGETS) $(LIBDIR)/$(STATICLIBTARGET)

-include $(DEPFILES)

$(LIBDIR)/$(LIBTARGET) : $(LIBOBJ)
	@printf "Linking   %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LDFLAGS) $(LIBOBJ)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)
	@printf "Archiving %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	ar cr $@ $(LIBOBJ)

$(INCDIR)/%.h : src/%.h
	@printf "Grabbing  %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(OBJDIR)/%.o : src/%.cu
	@printf "Compiling %-35s > %s\n" $< $@
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

TESTS       := all_gather_test     all_gather_scan \
               all_reduce_test     all_reduce_scan \
               broadcast_test      broadcast_scan \
               reduce_test         reduce_scan \
               reduce_scatter_test reduce_scatter_scan
MPITESTS    := mpi_test

TSTINC     := -I$(NCCL_INC) -Itest/include
TSTLIB     := -L$(NCCL_LIB) $(LIBLINK) $(LDFLAGS)
TSTDIR     := $(BUILDDIR)/test/single
MPITSTDIR  := $(BUILDDIR)/test/mpi
TESTBINS   := $(patsubst %, $(TSTDIR)/%, $(TESTS))
MPITESTBINS:= $(patsubst %, $(MPITSTDIR)/%, $(MPITESTS))

test : $(TESTBINS)

$(TSTDIR)/% : test/single/%.cu test/include/*.h $(TSTDEP)
	@printf "Building  %-35s > %s\n" $< $@
	mkdir -p $(TSTDIR)
	$(NVCC) $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@ $< $(TSTLIB) -lcuda -lcurand -lnvToolsExt
	@$(NVCC) -M $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< $(TSTLIB) -lcuda -lcurand -lnvToolsExt > $(@:%=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%=%.d.tmp) > $(@:%=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%=%.d)
	@rm -f $(@:%=%.d.tmp)

mpitest : $(MPITESTBINS)

$(MPITSTDIR)/% : test/mpi/%.cu $(TSTDEP)
	@printf "Building  %-35s > %s\n" $< $@
	mkdir -p $(MPITSTDIR)
	$(NVCC) $(MPIFLAGS) $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" -o $@ $< $(TSTLIB) -lcurand
	@$(NVCC) $(MPIFLAGS) -M $(TSTINC) $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< $(TSTLIB) -lcurand > $(@:%=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%=%.d.tmp) > $(@:%=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%=%.d)
	@rm -f $(@:%=%.d.tmp)

#### PACKAGING ####

DEBIANDIR  := $(BUILDDIR)/debian

DEBGEN_IN  := $(shell (cd debian ; ls *.in))
DEBGEN     := $(DEBGEN_IN:.in=)
DEBFILES   := compat copyright libnccl-dev.install libnccl-dev.manpages nccl.7 rules $(DEBGEN)
DEBTARGETS := $(patsubst %, $(DEBIANDIR)/%, $(DEBFILES))

DEB_REVISION   ?= 1
DEB_TIMESTAMP  := $(shell date -R)
DEB_ARCH       ?= amd64

debian : $(DEBTARGETS)

deb : lib debian
	@printf "Building Debian package\n"
	(cd $(BUILDDIR); debuild -eLD_LIBRARY_PATH -uc -us -d -b)
	mkdir -p $(BUILDDIR)/deb/
	mv $(BUILDDIR)/../libnccl*.deb $(BUILDDIR)/deb/

debclean :
	rm -Rf $(DEBIANDIR)

$(DEBIANDIR)/% : debian/%.in
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${cuda:Major}/$(CUDA_MAJOR)/g" \
	    -e "s/\$${cuda:Minor}/$(CUDA_MINOR)/g" \
	    -e "s/\$${deb:Revision}/$(DEB_REVISION)/g" \
	    -e "s/\$${deb:Timestamp}/$(DEB_TIMESTAMP)/g" \
	    -e "s/\$${deb:Arch}/$(DEB_ARCH)/g" \
	    $< > $@

$(DEBIANDIR)/% : debian/%
	@printf "Grabbing  %-35s > %s\n" $< $@
	mkdir -p $(DEBIANDIR)
	cp -f $< $@

#### FORTRAN BINDINGS ####

export NCCL_MAJOR NCCL_MINOR NCCL_PATCH CUDA_MAJOR CUDA_MINOR LIBLINK CUDA_LIB BUILDDIR

forlib : lib
	$(MAKE) -C fortran lib
fortest : forlib
	$(MAKE) -C fortran test
forclean :
	$(MAKE) -C fortran clean

