#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <c10/util/Borrowed.h>
#include <c10/util/BorrowedTensorTraits.h>

namespace {

template <typename T>
T getSampleValue();

template <>
at::Tensor getSampleValue() {
  return at::zeros({2, 2}).to(at::kCPU);
}

template <>
at::TensorBase getSampleValue() {
  return getSampleValue<at::Tensor>();
}

template <typename T>
class BorrowedTest : public ::testing::Test {
 public:
  c10::Borrowed<T> defaultConstructed;
  c10::Borrowed<T> sample;
  c10::Borrowed<T> sampleViaRawImplPointer;
  ;
  T ownedSample;

 protected:
  void SetUp() override; // defined below helpers
  void TearDown() override {
    defaultConstructed = c10::Borrowed<T>();
    sample = c10::Borrowed<T>();
    sampleViaRawImplPointer = c10::Borrowed<T>();
    ownedSample = T();
  }

  void expectEmpty(const c10::Borrowed<T>& b) {
    EXPECT_FALSE(static_cast<bool>(b));
  }

  void expectBorrowsFrom(const c10::Borrowed<T>& b, const T& from) {
    EXPECT_TRUE(static_cast<bool>(b));
    EXPECT_EQ(b->unsafeGetTensorImpl(), from.unsafeGetTensorImpl());
    EXPECT_EQ((*b).unsafeGetTensorImpl(), from.unsafeGetTensorImpl());
    EXPECT_EQ(b.get(), b.operator->());
  }
};

template <typename T>
void BorrowedTest<T>::SetUp() {
  defaultConstructed = c10::Borrowed<T>();
  ownedSample = getSampleValue<T>();
  sample = c10::Borrowed<T>(ownedSample);
  sampleViaRawImplPointer = c10::Borrowed<T>(ownedSample.unsafeGetTensorImpl());
}

using BorrowedTypes = ::testing::Types<at::Tensor, at::TensorBase>;

TYPED_TEST_CASE(BorrowedTest, BorrowedTypes);

TYPED_TEST(BorrowedTest, Basic) {
  this->expectEmpty(this->defaultConstructed);
  this->expectBorrowsFrom(this->sample, this->ownedSample);
  this->expectBorrowsFrom(this->sampleViaRawImplPointer, this->ownedSample);
}

TYPED_TEST(BorrowedTest, CopyConstructor) {
  auto copiedDefault = this->defaultConstructed;
  auto copiedSample = this->sample;
  auto copiedSampleViaRawImplPointer = this->sampleViaRawImplPointer;

  this->expectEmpty(copiedDefault);
  this->expectBorrowsFrom(copiedSample, this->ownedSample);
  this->expectBorrowsFrom(copiedSampleViaRawImplPointer, this->ownedSample);
}

TYPED_TEST(BorrowedTest, MoveConstructor) {
  auto movedDefault = std::move(this->defaultConstructed);
  auto movedSample = std::move(this->sample);
  auto movedSampleViaRawImplPointer = this->sampleViaRawImplPointer;

  this->expectEmpty(movedDefault);
  this->expectBorrowsFrom(movedSample, this->ownedSample);
  this->expectBorrowsFrom(movedSampleViaRawImplPointer, this->ownedSample);
}

TYPED_TEST(BorrowedTest, CopyAssignment) {
  c10::Borrowed<TypeParam> copiedDefault, copiedSample,
      copiedSampleViaRawImplPointer;
  copiedDefault = this->defaultConstructed;
  copiedSample = this->sample;
  copiedSampleViaRawImplPointer = this->sampleViaRawImplPointer;

  this->expectEmpty(copiedDefault);
  this->expectBorrowsFrom(copiedSample, this->ownedSample);
  this->expectBorrowsFrom(copiedSampleViaRawImplPointer, this->ownedSample);
}

TYPED_TEST(BorrowedTest, MoveAssignment) {
  c10::Borrowed<TypeParam> movedDefault, movedSample,
      movedSampleViaRawImplPointer;
  movedDefault = std::move(this->defaultConstructed);
  movedSample = std::move(this->sample);
  movedSampleViaRawImplPointer = this->sampleViaRawImplPointer;

  this->expectEmpty(movedDefault);
  this->expectBorrowsFrom(movedSample, this->ownedSample);
  this->expectBorrowsFrom(movedSampleViaRawImplPointer, this->ownedSample);
}

TYPED_TEST(BorrowedTest, TensorAssignment) {
  auto copiedDefault = this->defaultConstructed;
  auto copiedSample = this->sample;
  auto copiedSampleViaRawImplPointer = this->sampleViaRawImplPointer;
  for (auto dest :
       {this->defaultConstructed,
        this->sample,
        this->sampleViaRawImplPointer}) {
    for (auto source :
         {this->defaultConstructed,
          this->sample,
          this->sampleViaRawImplPointer}) {
      dest = *source;
      if (source) {
        this->expectBorrowsFrom(dest, *source);
      } else {
        this->expectEmpty(dest);
      }
    }
  }
}

} // namespace
