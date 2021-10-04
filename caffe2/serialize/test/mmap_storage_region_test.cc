#include "caffe2/serialize/mmap_storage_region.h"

#include <ATen/ATen.h>
#include <array>
#include <aten/src/ATen/Context.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>
#include <cstdio>
#include <filesystem> // remove
#include <fstream> // ofstream
#include <gtest/gtest.h>
#include <string>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <unistd.h> // close(...)

namespace caffe2 {
namespace serialize {
namespace {


TEST(MmapStorageRegionTest, Test1) {
  if (!MmapStorageRegion::isSupportedByPlatform()) {
    GTEST_SKIP() << "MmapStorageRegion not supported on this platform";
  }

  // Create file for mmap
  const std::string filename = "data.txt";
  const int N = 3;
  std::ofstream stream(filename);
  for (uint8_t i = 0; i < N; i++) {
    stream << i;
  }
  stream.close();

  const at::ScalarType type = at::ScalarType::Byte;
  const caffe2::TypeMeta dtype = at::CPU(type).typeMeta();
  const c10::TensorOptions options = at::CPU(type).options();
  std::array<at::Tensor, N> tensors;

  MmapStorageRegion* region = nullptr;

  // Simulate creating an MmapStorageRegion intrusive ptr within a function call
  // and using its DataPtrs after returning from the scope in which it was made
  const auto set_mmap_storage_region_and_make_tensors = [&]() {
    c10::intrusive_ptr<MmapStorageRegion> mmapping =
        c10::make_intrusive<MmapStorageRegion>(filename);

    region = mmapping.get();

    // There should be one "use", the intrusive ptr
    ASSERT_EQ(c10::raw::intrusive_ptr::use_count(region), 1);

    for (int i = 0; i < N; i++) {
      at::DataPtr storage_ptr = mmapping->getData(i, DeviceType::CPU);
      at::Storage storage = at::Storage(
        c10::Storage::use_byte_size_t(),
        dtype.itemsize(),
        std::move(storage_ptr),
        /*allocator=*/nullptr,
        /*resizable=*/false);
      tensors[i] = at::empty({0}, options).set_(storage);

      // There should be one "use" for each created Tensor and the intrusive ptr
      ASSERT_EQ(c10::raw::intrusive_ptr::use_count(region), 2 + i);
    }

  };

  set_mmap_storage_region_and_make_tensors();

  // One "use" for each Tensor, but not for the intrusive ptr
  ASSERT_EQ(c10::raw::intrusive_ptr::use_count(region), N);

  for (int i = 0; i < N; i++) {
    // Tensor data should still exist
    ASSERT_EQ(tensors[i].data_ptr<uint8_t>()[0], i);
  }

  for (int i = N - 1; i >= 0; i--) {
    // Remove the tensor from scope, decreasing the use count
    tensors[i] = at::empty({0});
    if (i > 0) {
      // One "use" for each Tensor that hasn't been removed from scope so far
      ASSERT_EQ(c10::raw::intrusive_ptr::use_count(region), i);
    }
  }

  // By now, the MmapStorageRegion has been deconstructed so we can't check
  // the use count anymore, but I confirmed with print statements that the
  // deconstructor has been called by this point

  std::filesystem::remove(filename);
}


} // namespace
} // namespace serialize
} // namespace caffe2
