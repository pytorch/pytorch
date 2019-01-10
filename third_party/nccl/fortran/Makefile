FC := gfortran
FCNAME := $(notdir $(FC))

BUILDDIR ?= ../build
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj

LIBNAME    := libncclfor.so
LIBSONAME  := $(patsubst %,%.$(NCCL_MAJOR),$(LIBNAME))
LIBTARGET  := $(patsubst %,%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH),$(LIBNAME))
LIBLINK    += $(patsubst lib%.so,-l%,$(LIBNAME))

LIBCUDAFOR := libcudafor.so

ifneq ($(filter pgf%, $(FCNAME)), )
# PGI compiler (pgfortran, pgf90, pgf95)
FCMODFLAGS  := -module $(INCDIR)
FCPREFLAGS  := -Mpreprocess
FCCUDAFLAGS := -Mcuda,cuda$(CUDA_MAJOR).$(CUDA_MINOR)
FCFLAGS     := -fast -O3
else
# non-PGI compilers do not have CUDA support, compile our own CUDA lib
CUDAFORDEP  := $(LIBDIR)/$(LIBCUDAFOR)
CUDALINK    := -L$(CUDA_LIB) -lcudart
CUDAFORLINK := -lcudafor
ifeq ($(FCNAME), gfortran)
FCMODFLAGS  := -J$(INCDIR)
FCPREFLAGS  += -cpp
FCFLAGS     += -ffree-line-length-none
else ifeq ($(FCNAME), ifort)
FCMODFLAGS  := -module $(INCDIR)
FCPREFLAGS  += -fpp
endif
endif

ifeq ($(VERBOSE), 0)
.SILENT:
endif

lib: $(CUDAFORDEP)
	$(MAKE) $(LIBDIR)/$(LIBTARGET)

$(LIBDIR)/$(LIBTARGET): $(OBJDIR)/ncclfor.o
	@printf "Linking   %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(FC) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) $< -o $(LIBDIR)/$(LIBTARGET)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(LIBCUDAFOR): $(OBJDIR)/cudafor.o
	@printf "Linking   %-35s > %s\n" $(LIBCUDAFOR) $@
	mkdir -p $(LIBDIR)
	$(FC) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBCUDAFOR) $< -o $(LIBDIR)/$(LIBCUDAFOR)

$(OBJDIR)/%.o: src/%.f90
	@printf "Building  %-35s > %s\n" $< $@
	mkdir -p $(OBJDIR)
	mkdir -p $(INCDIR)
	$(FC) -c $(FCMODFLAGS) $(FCPREFLAGS) -fPIC $(FCCUDAFLAGS) $(FCFLAGS) $< -o $@

TESTS := reduce_ptr_out allreduce_ptr_out reducescatter_ptr_out broadcast_ptr allgather_ptr_out
ifneq ($(filter pgf%, $(FCNAME)), )
TESTS += reduce_arr_out allreduce_arr_out reducescatter_arr_out broadcast_arr allgather_arr_out
endif

TESTDIR  := $(BUILDDIR)/test/fortran
TESTBINS := $(patsubst %,$(TESTDIR)/%,$(TESTS))

test: lib $(TESTBINS)

$(TESTDIR)/%: test/%.f90 lib
	@printf "Building  %-35s > %s\n" $< $@
	@mkdir -p $(TESTDIR)
	$(FC) $(FCCUDAFLAGS) $(FCFLAGS) $< $(CUDALINK) -I$(INCDIR) -L$(LIBDIR) $(CUDAFORLINK) $(LIBLINK) -o $@

clean:
	rm -f $(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	rm -f $(LIBDIR)/$(LIBCUDAFOR) $(OBJDIR)/*for.o $(INCDIR)/*.mod
	rm -rf $(TESTDIR)/

