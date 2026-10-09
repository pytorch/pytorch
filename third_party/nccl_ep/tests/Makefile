# Unit tests for nccl_ep output layout (GPU-major vs expert-major).
#
# Build:  make -C contrib/nccl_ep/tests [BUILDDIR=<path>]
# Run:    bash run_tests.sh
#   or:   NCCL_HOME=<build> NCCL_EP_BUILDDIR=<build> bash run_tests.sh

BUILDDIR ?= $(abspath ../../../build)
NCCL_HOME ?= $(BUILDDIR)
NCCL_EP_BUILDDIR ?= $(BUILDDIR)

NCCL_INCDIR := $(NCCL_HOME)/include
NCCL_LIBDIR := $(NCCL_HOME)/lib
NCCL_EP_INCDIR := $(NCCL_EP_BUILDDIR)/include
NCCL_EP_LIBDIR := $(NCCL_EP_BUILDDIR)/lib
DST_DIR := $(NCCL_EP_BUILDDIR)/test/nccl_ep

GTEST_DIR  := $(abspath ../../../test/googletest/googletest)
GTEST_OBJDIR := $(NCCL_EP_BUILDDIR)/test
GTEST_LIBS := $(GTEST_OBJDIR)/gtest.a $(GTEST_OBJDIR)/gtest_main.a

NVCC         ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
NVCC_GENCODE ?= -gencode=arch=compute_90,code=sm_90
CXXSTD       := -std=c++17

NVCUFLAGS := $(NVCC_GENCODE) \
             -I$(NCCL_EP_INCDIR) \
             -I$(NCCL_INCDIR) \
             -I$(GTEST_DIR)/include \
             --expt-relaxed-constexpr \
             --compiler-options "-fPIC -pthread" \
             -ccbin $(CXX) $(CXXSTD)

NVLDFLAGS := -L$(NCCL_EP_LIBDIR) -lnccl_ep -L$(NCCL_LIBDIR) -lnccl -lpthread -lrt -ldl -lcuda

TEST_BINARIES := $(DST_DIR)/test_output_layout $(DST_DIR)/test_handle_maps $(DST_DIR)/test_lifecycle $(DST_DIR)/test_ht_bwd $(DST_DIR)/test_tensor_create $(DST_DIR)/test_ht_stale_routing_map

.PHONY: all clean
all: $(TEST_BINARIES)

$(GTEST_OBJDIR)/gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	@mkdir -p $(GTEST_OBJDIR)
	$(CXX) $(CXXSTD) -pthread -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -c $< -o $@

$(GTEST_OBJDIR)/gtest_main.o: $(GTEST_DIR)/src/gtest_main.cc
	@mkdir -p $(GTEST_OBJDIR)
	$(CXX) $(CXXSTD) -pthread -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -c $< -o $@

$(GTEST_OBJDIR)/gtest.a: $(GTEST_OBJDIR)/gtest-all.o
	$(AR) rcs $@ $^

$(GTEST_OBJDIR)/gtest_main.a: $(GTEST_OBJDIR)/gtest-all.o $(GTEST_OBJDIR)/gtest_main.o
	$(AR) rcs $@ $^

$(DST_DIR):
	@mkdir -p $@

INTERNAL_HEADERS := $(abspath ../nccl_ep_test_internal.h) test_common.h

$(DST_DIR)/%.o: %.cu $(INTERNAL_HEADERS) | $(DST_DIR)
	@printf "Compiling  %-45s > %s\n" $< $@
	$(NVCC) $(NVCUFLAGS) -c $< -o $@

.PRECIOUS: $(DST_DIR)/%.o

$(DST_DIR)/%: $(DST_DIR)/%.o $(NCCL_EP_LIBDIR)/libnccl_ep.so $(GTEST_LIBS)
	@printf "Linking    %-45s > %s\n" $< $@
	$(NVCC) $(NVCUFLAGS) $< $(NVLDFLAGS) $(GTEST_LIBS) -o $@

clean:
	rm -f $(TEST_BINARIES) $(patsubst %,%.o,$(TEST_BINARIES))
