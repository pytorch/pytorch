#include <gtest/gtest.h>

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/ATen.h>

#include <functional>

using namespace at::vec256;

bool check_equal(const at::Tensor& a, const at::Tensor& b) {
  return (a.equal(b));
}

bool check_almost_equal(
    const at::Tensor& a, const at::Tensor& b, const float tolerance) {
  double max_val = a.abs().max().item<float>();
  max_val = std::max(max_val, b.abs().max().item<float>());
  if ((a - b).abs().max().item<float>() > tolerance * max_val) {
    std::cout << "Max difference:"
      << (a - b).abs().max().item<float>() << std::endl;
    return false;
  }
  return true;
}

template<typename T>
void BlendTestHelperScalar(
    const T* a_ptr,
    const T* b_ptr,
    T* res_ptr,
    const int64_t num_els,
    const int64_t count) {
  for(auto i = 0; i < num_els; ++i) {
    for (auto j = 0; j < Vec256<float>::size(); ++j) {
      auto index = i * Vec256<float>::size() + j;
      if (j < count) {
        res_ptr[index] = b_ptr[index];
      } else {
        res_ptr[index] = a_ptr[index];
      }
    }
  }
}

namespace Impl {
float reciprocal(const float a) {
  return (1/a);
}

float rsqrt(const float a) {
  return (1/std::sqrt(a));
}

float frac(const float a) {
  return a - (static_cast<int32_t>(a));
}
}

template<typename T>
void BlendTestHelperVector(
    const T* a_ptr,
    const T* b_ptr,
    T* res_ptr,
    const int64_t num_els,
    const int64_t count) {
  for(auto i = 0; i < num_els; ++i) {
    auto a_elements = Vec256<float>::loadu(a_ptr);
    auto b_elements = Vec256<float>::loadu(b_ptr);
    a_ptr += Vec256<float>::size();
    b_ptr += Vec256<float>::size();
    auto res_elements = Vec256<float>::set(a_elements, b_elements, count);
    res_elements.store(res_ptr);
    res_ptr += Vec256<float>::size();
  }
}

#define TranscedentalTester(opnamespace, name)                    \
void TranscedentalHelper_##name(const float tolerance = 1e-6) {   \
  at::Tensor a = at::rand({23, 23});                              \
  a = a * -10;                                                    \
  a = a + 10;                                                     \
  at::Tensor ref_res = at::zeros({23, 23});                       \
  at::Tensor vec_res = at::zeros({23, 23});                       \
  float* a_ptr = a.data_ptr<float>();                             \
  float* ref_res_ptr = ref_res.data_ptr<float>();                 \
  float* vec_res_ptr = vec_res.data_ptr<float>();                 \
  size_t num_els =                                                \
    (a.numel() / Vec256<float>::size()) * Vec256<float>::size();  \
  for(auto i = 0; i < num_els; ++i) {                             \
    ref_res_ptr[i] = opnamespace::name(a_ptr[i]);                 \
  }                                                               \
  for (size_t i = 0; i < num_els; i += Vec256<float>::size()) {   \
    auto a_elements = Vec256<float>::loadu(a_ptr);                \
    a_ptr += Vec256<float>::size();                               \
    auto res = a_elements.name();                                 \
    res.store(vec_res_ptr);                                       \
    vec_res_ptr += Vec256<float>::size();                         \
  }                                                               \
  ASSERT_TRUE(check_almost_equal(ref_res, vec_res, tolerance));   \
}

#define TranscedentalTester2(name)                                \
void TranscedentalHelper_##name(const float tolerance = 1e-6) {   \
  at::Tensor a = at::rand({23, 23});                              \
  at::Tensor b = at::rand({23, 23});                              \
  a = a * -10;                                                    \
  a = a + 10;                                                     \
  at::Tensor ref_res = at::zeros({23, 23});                       \
  at::Tensor vec_res = at::zeros({23, 23});                       \
  float* a_ptr = a.data_ptr<float>();                             \
  float* b_ptr = a.data_ptr<float>();                             \
  float* ref_res_ptr = ref_res.data_ptr<float>();                 \
  float* vec_res_ptr = vec_res.data_ptr<float>();                 \
  size_t num_els =                                                \
    (a.numel() / Vec256<float>::size()) * Vec256<float>::size();  \
  for(auto i = 0; i < num_els; ++i) {                             \
    ref_res_ptr[i] = std::name(a_ptr[i], b_ptr[i]);               \
  }                                                               \
  for (size_t i = 0; i < num_els; i += Vec256<float>::size()) {   \
    auto a_elements = Vec256<float>::loadu(a_ptr);                \
    auto b_elements = Vec256<float>::loadu(b_ptr);                \
    a_ptr += Vec256<float>::size();                               \
    b_ptr += Vec256<float>::size();                               \
    auto res = a_elements.name(b_elements);                       \
    res.store(vec_res_ptr);                                       \
    vec_res_ptr += Vec256<float>::size();                         \
  }                                                               \
  ASSERT_TRUE(check_almost_equal(ref_res, vec_res, tolerance));   \
}

// Not testing all the transcendentals.
// In fact fewer than these might suffice, since current implementation
// actually just calls STL version of these.
// So what is really being checked is the logic to map a function.
TranscedentalTester(std, abs)
TranscedentalTester(std, acos)
TranscedentalTester(std, asin)
TranscedentalTester(std, atan)
TranscedentalTester(std, erf)
TranscedentalTester(std, exp)
TranscedentalTester(std, log)
TranscedentalTester(std, tan)
TranscedentalTester(std, trunc)
TranscedentalTester(std, sqrt)

TranscedentalTester2(atan2)
TranscedentalTester2(fmod)
TranscedentalTester2(pow)

TranscedentalTester(Impl, reciprocal)
TranscedentalTester(Impl, rsqrt)
TranscedentalTester(Impl, frac)

enum class OP_TYPE {
  EQ = 0,
  NE,
  GT,
  GE,
  LT,
  LE,
  MIN,
  MAX,
  ADD,
  SUB,
  MUL,
  DIV,
  AND,
  OR,
  EXOR
};

void BasicOpTestHelper(const OP_TYPE& compare_type) {
  at::Tensor a = at::rand({23, 23});
  at::Tensor b = at::rand({23, 23});
  at::Tensor ref_res = at::zeros({23, 23});
  at::Tensor vec_res = at::zeros({23, 23});

  size_t num_els =
    (a.numel() / Vec256<float>::size()) * Vec256<float>::size();
  // Vector components
  float* a_ptr = a.data_ptr<float>();
  float* b_ptr = b.data_ptr<float>();
  float* ref_res_ptr = ref_res.data_ptr<float>();
  for (size_t i = 0; i < num_els; ++i) {
    switch (compare_type) {
      case OP_TYPE::EQ:
        if (a_ptr[i] == b_ptr[i]) {
          ref_res_ptr[i] = 1.0f;
        } else {
          ref_res_ptr[i] = 0;
        }
        break;
      case OP_TYPE::NE:
        if (a_ptr[i] != b_ptr[i]) {
          ref_res_ptr[i] = 1.0f;
        } else {
          ref_res_ptr[i] = 0;
        }
        break;
      case OP_TYPE::GT:
        if (a_ptr[i] > b_ptr[i]) {
          ref_res_ptr[i] = 1.0f;
        } else {
          ref_res_ptr[i] = 0;
        }
        break;
      case OP_TYPE::GE:
        if (a_ptr[i] >= b_ptr[i]) {
          ref_res_ptr[i] = 1.0f;
        } else {
          ref_res_ptr[i] = 0;
        }
        break;
      case OP_TYPE::LT:
        if (a_ptr[i] < b_ptr[i]) {
          ref_res_ptr[i] = 1.0f;
        } else {
          ref_res_ptr[i] = 0;
        }
        break;
      case OP_TYPE::LE:
        if (a_ptr[i] <= b_ptr[i]) {
          ref_res_ptr[i] = 1.0f;
        } else {
          ref_res_ptr[i] = 0;
        }
        break;
      case OP_TYPE::MIN:
        ref_res_ptr[i] = std::min(a_ptr[i], b_ptr[i]);
        break;
      case OP_TYPE::MAX:
        ref_res_ptr[i] = std::max(a_ptr[i], b_ptr[i]);
        break;
      case OP_TYPE::ADD:
        ref_res_ptr[i] = a_ptr[i] + b_ptr[i];
        break;
      case OP_TYPE::SUB:
        ref_res_ptr[i] = a_ptr[i] - b_ptr[i];
        break;
      case OP_TYPE::MUL:
        ref_res_ptr[i] = a_ptr[i] * b_ptr[i];
        break;
      case OP_TYPE::DIV:
        ref_res_ptr[i] = a_ptr[i] / b_ptr[i];
        break;
      case OP_TYPE::OR:
        {
          uint32_t *a_val, *b_val;
          a_val = reinterpret_cast<uint32_t*>(&a_ptr[i]);
          b_val = reinterpret_cast<uint32_t*>(&b_ptr[i]);
          uint32_t c_val = (*a_val) | (*b_val);
          float* c_val_float;
          c_val_float = reinterpret_cast<float*>(&c_val);
          ref_res_ptr[i] = *c_val_float;
        }
        break;
      case OP_TYPE::AND:
        {
          uint32_t *a_val, *b_val;
          a_val = reinterpret_cast<uint32_t*>(&a_ptr[i]);
          b_val = reinterpret_cast<uint32_t*>(&b_ptr[i]);
          uint32_t c_val = (*a_val) & (*b_val);
          float* c_val_float;
          c_val_float = reinterpret_cast<float*>(&c_val);
          ref_res_ptr[i] = *c_val_float;
        }
        break;
      case OP_TYPE::EXOR:
        {
          uint32_t *a_val, *b_val;
          a_val = reinterpret_cast<uint32_t*>(&a_ptr[i]);
          b_val = reinterpret_cast<uint32_t*>(&b_ptr[i]);
          uint32_t c_val = (*a_val) ^ (*b_val);
          float* c_val_float;
          c_val_float = reinterpret_cast<float*>(&c_val);
          ref_res_ptr[i] = *c_val_float;
        }
        break;
    }
  }

  // Vectorized impl
  float* vec_res_ptr = vec_res.data_ptr<float>();
  for (size_t i = 0; i < num_els; i += Vec256<float>::size()) {
    auto a_elements = Vec256<float>::loadu(a_ptr);
    auto b_elements = Vec256<float>::loadu(b_ptr);
    a_ptr += Vec256<float>::size();
    b_ptr += Vec256<float>::size();
    Vec256<float> res_elements;
    switch (compare_type) {
      case OP_TYPE::EQ:
        res_elements = a_elements.eq(b_elements);
        break;
      case OP_TYPE::NE:
        res_elements = a_elements.ne(b_elements);
        break;
      case OP_TYPE::GT:
        res_elements = a_elements.gt(b_elements);
        break;
      case OP_TYPE::GE:
        res_elements = a_elements.ge(b_elements);
        break;
      case OP_TYPE::LT:
        res_elements = a_elements.lt(b_elements);
        break;
      case OP_TYPE::LE:
        res_elements = a_elements.le(b_elements);
        break;
      case OP_TYPE::MIN:
        res_elements = at::vec256::minimum(a_elements, b_elements);
        break;
      case OP_TYPE::MAX:
        res_elements = at::vec256::maximum(a_elements, b_elements);
        break;
      case OP_TYPE::ADD:
        res_elements = a_elements + b_elements;
        break;
      case OP_TYPE::SUB:
        res_elements = a_elements - b_elements;
        break;
      case OP_TYPE::MUL:
        res_elements = a_elements * b_elements;
        break;
      case OP_TYPE::DIV:
        res_elements = a_elements / b_elements;
        break;
      case OP_TYPE::OR:
        res_elements = a_elements | b_elements;
        break;
      case OP_TYPE::AND:
        res_elements = a_elements & b_elements;
        break;
      case OP_TYPE::EXOR:
        res_elements = a_elements ^ b_elements;
        break;
    }
    res_elements.store(vec_res_ptr);
    vec_res_ptr += Vec256<float>::size();
  }
  ASSERT_TRUE(check_equal(ref_res, vec_res));
}

// Checks both loads and stores.
TEST(Vec256TestFloat, CopyTest) {
  at::Tensor a = at::rand({23, 23});
  at::Tensor b = at::zeros({23, 23});
  // Copy goes through vec256 via tensoriterator
  b.copy_(a);
  ASSERT_TRUE(check_equal(a, b));
}

TEST(Vec256TestFloat, arangeTest) {
  at::Tensor arange_output_ref = at::zeros({8});
  at::Tensor arange_output_vectorized = at::zeros({8});
  float base = 7.f;
  float step = 5.f;
  float* ref_output_ptr = arange_output_ref.data_ptr<float>();
  for (int64_t i = 0; i < 8; ++i) {
    ref_output_ptr[i] = base + i * step;
  }
  float* vec_output_ptr = arange_output_vectorized.data_ptr<float>();
  auto arange_output = Vec256<float>::arange(base, step);
  arange_output.store(vec_output_ptr);
  ASSERT_TRUE(check_equal(arange_output_ref, arange_output_vectorized));
}

// Checks blend and blendv.
TEST(Vec256TestFloat, Blend) {
  at::Tensor a = at::rand({23, 23});
  at::Tensor b = at::rand({23, 23});
  at::Tensor ref_res = at::zeros({23, 23});
  at::Tensor vec_res = at::zeros({23, 23});

  // Check templatized blend.
  // Reference result:
  const int64_t mask = 0xC5;
  // Only check over multiple of Vec::size elements
  size_t num_els =
    (a.numel() / Vec256<float>::size()) * Vec256<float>::size();
  // Vector components
  float* a_ptr = a.data_ptr<float>();
  float* b_ptr = b.data_ptr<float>();
  float* ref_res_ptr = ref_res.data_ptr<float>();
  int64_t tmp_mask = mask;
  for (size_t i = 0; i < num_els; ++i) {
    if (i % Vec256<float>::size() == 0) {
      tmp_mask = mask;
    }
    if (tmp_mask & 0x1) {
      ref_res_ptr[i] = b_ptr[i];
    } else {
      ref_res_ptr[i] = a_ptr[i];
    }
    tmp_mask = tmp_mask >> 1;
  }

  // Vectorized impl
  float* vec_res_ptr = vec_res.data_ptr<float>();
  for (size_t i = 0; i < num_els; i += Vec256<float>::size()) {
    auto a_elements = Vec256<float>::loadu(a_ptr);
    auto b_elements = Vec256<float>::loadu(b_ptr);
    a_ptr += Vec256<float>::size();
    b_ptr += Vec256<float>::size();
    auto res_elements = Vec256<float>::blend<mask>(a_elements, b_elements);
    res_elements.store(vec_res_ptr);
    vec_res_ptr += Vec256<float>::size();
  }
  ASSERT_TRUE(check_equal(ref_res, vec_res));

  // Vector components
  a_ptr = a.data_ptr<float>();
  b_ptr = b.data_ptr<float>();
  int32_t full_int_mask = 0xFFFFFFFF;
  float* full_ptr = reinterpret_cast<float*>(&full_int_mask);
  float full_float_mask = *full_ptr;
  Vec256<float> float_mask(full_float_mask, 0.f, full_float_mask, 0.f,
                           0.f, full_float_mask, 0.f, 0.f);
  float float_mask_array[Vec256<float>::size()];
  float_mask.store(float_mask_array);
  ref_res_ptr = ref_res.data_ptr<float>();
  for (size_t i = 0; i < num_els; ++i) {
    if (float_mask_array[i % Vec256<float>::size()] != 0) {
      ref_res_ptr[i] = b_ptr[i];
    } else {
      ref_res_ptr[i] = a_ptr[i];
    }
    tmp_mask = tmp_mask >> 1;
  }

  // Vectorized impl
  vec_res_ptr = vec_res.data_ptr<float>();
  for (size_t i = 0; i < num_els; i += Vec256<float>::size()) {
    auto a_elements = Vec256<float>::loadu(a_ptr);
    auto b_elements = Vec256<float>::loadu(b_ptr);
    a_ptr += Vec256<float>::size();
    b_ptr += Vec256<float>::size();
    auto res_elements = Vec256<float>::blendv(a_elements, b_elements, float_mask);
    res_elements.store(vec_res_ptr);
    vec_res_ptr += Vec256<float>::size();
  }
  ASSERT_TRUE(check_equal(ref_res, vec_res));
}

// Checks Set
TEST(Vec256TestFloat, Set) {
  at::Tensor a = at::rand({23, 23});
  at::Tensor b = at::rand({23, 23});
  at::Tensor ref_res = at::zeros({23, 23});
  at::Tensor vec_res = at::zeros({23, 23});

  const float* a_ptr = a.data_ptr<float>();
  const float* b_ptr = b.data_ptr<float>();
  float* ref_res_ptr = ref_res.data_ptr<float>();
  float* vec_res_ptr = vec_res.data_ptr<float>();

  // Only check over multiple of Vec::size elements
  const size_t num_els = (a.numel() / Vec256<float>::size());
  BlendTestHelperScalar(a_ptr, b_ptr, ref_res_ptr, num_els, 0);
  BlendTestHelperVector(a_ptr, b_ptr, vec_res_ptr, num_els, 0);
  ASSERT_TRUE(check_equal(ref_res, vec_res));
  BlendTestHelperScalar(a_ptr, b_ptr, ref_res_ptr, num_els, 1);
  BlendTestHelperVector(a_ptr, b_ptr, vec_res_ptr, num_els, 1);
  ASSERT_TRUE(check_equal(ref_res, vec_res));
  BlendTestHelperScalar(a_ptr, b_ptr, ref_res_ptr, num_els, 4);
  BlendTestHelperVector(a_ptr, b_ptr, vec_res_ptr, num_els, 4);
  ASSERT_TRUE(check_equal(ref_res, vec_res));
  BlendTestHelperScalar(a_ptr, b_ptr, ref_res_ptr, num_els, 6);
  BlendTestHelperVector(a_ptr, b_ptr, vec_res_ptr, num_els, 6);
  ASSERT_TRUE(check_equal(ref_res, vec_res));
  BlendTestHelperScalar(a_ptr, b_ptr, ref_res_ptr, num_els, 8);
  BlendTestHelperVector(a_ptr, b_ptr, vec_res_ptr, num_els, 8);
  ASSERT_TRUE(check_equal(ref_res, vec_res));
}

TEST(Vec256TestFloat, Abs) {
  TranscedentalHelper_abs();
}

TEST(Vec256TestFloat, acos) {
  TranscedentalHelper_acos();
}

TEST(Vec256TestFloat, asin) {
  TranscedentalHelper_asin();
}

TEST(Vec256TestFloat, atan) {
  TranscedentalHelper_atan();
}

TEST(Vec256TestFloat, erf) {
  TranscedentalHelper_erf();
}

TEST(Vec256TestFloat, exp) {
  TranscedentalHelper_exp();
}

TEST(Vec256TestFloat, tan) {
  TranscedentalHelper_tan();
}

TEST(Vec256TestFloat, log) {
  TranscedentalHelper_log();
}

TEST(Vec256TestFloat, trunc) {
  TranscedentalHelper_trunc();
}

TEST(Vec256TestFloat, sqrt) {
  TranscedentalHelper_sqrt();
}

TEST(Vec256TestFloat, atan2) {
  TranscedentalHelper_atan2();
}

TEST(Vec256TestFloat, fmod) {
  TranscedentalHelper_fmod();
}

TEST(Vec256TestFloat, pow) {
  TranscedentalHelper_pow();
}

TEST(Vec256TestFloat, reciprocal) {
  TranscedentalHelper_reciprocal(1e-3);
}

TEST(Vec256TestFloat, rsqrt) {
  // rsqrt tolerance is much worse.
  // If we did not set seed even this is violated sometimes.
  TranscedentalHelper_rsqrt(5e-3);
}

TEST(Vec256TestFloat, frac) {
  TranscedentalHelper_frac();
}

TEST(Vec256TestFloat, compare_eq) {
  BasicOpTestHelper(OP_TYPE::EQ);
}

TEST(Vec256TestFloat, compare_ne) {
  BasicOpTestHelper(OP_TYPE::NE);
}

TEST(Vec256TestFloat, compare_gt) {
  BasicOpTestHelper(OP_TYPE::GT);
}

TEST(Vec256TestFloat, compare_ge) {
  BasicOpTestHelper(OP_TYPE::GE);
}

TEST(Vec256TestFloat, compare_lt) {
  BasicOpTestHelper(OP_TYPE::LT);
}

TEST(Vec256TestFloat, compare_le) {
  BasicOpTestHelper(OP_TYPE::LE);
}

TEST(Vec256TestFloat, check_min) {
  BasicOpTestHelper(OP_TYPE::MIN);
}

TEST(Vec256TestFloat, check_max) {
  BasicOpTestHelper(OP_TYPE::MAX);
}

TEST(Vec256TestFloat, compare_add) {
  BasicOpTestHelper(OP_TYPE::ADD);
}

TEST(Vec256TestFloat, compare_sub) {
  BasicOpTestHelper(OP_TYPE::SUB);
}

TEST(Vec256TestFloat, check_mul) {
  BasicOpTestHelper(OP_TYPE::MUL);
}

TEST(Vec256TestFloat, check_div) {
  BasicOpTestHelper(OP_TYPE::DIV);
}

TEST(Vec256TestFloat, compare_or) {
  BasicOpTestHelper(OP_TYPE::OR);
}

TEST(Vec256TestFloat, check_and) {
  BasicOpTestHelper(OP_TYPE::AND);
}

TEST(Vec256TestFloat, check_xor) {
  BasicOpTestHelper(OP_TYPE::EXOR);
}

TEST(Vec256TestFloat, check_convert) {
  at::Tensor a = at::rand({23, 23});
  a = a * -10;
  a = a + 10;
  at::Tensor ref_res =
    at::empty({23, 23}, at::device(at::kCPU).dtype(at::kInt));
  at::Tensor vec_res =
    at::empty({23, 23}, at::device(at::kCPU).dtype(at::kInt));
  float* a_float_ptr = a.data_ptr<float>();
  int32_t* ref_res_int_ptr = ref_res.data_ptr<int32_t>();
  int32_t* vec_res_int_ptr = vec_res.data_ptr<int32_t>();
  for(auto i = 0; i < a.numel(); ++i) {
    ref_res_int_ptr[i] = static_cast<int32_t>(a_float_ptr[i]);
  }
  at::vec256::convert(a_float_ptr, vec_res_int_ptr, a.numel());
  ASSERT_TRUE(check_almost_equal(ref_res, vec_res, 1e-6));

  a = at::randint(-100, 100, {23, 23});
  a = a.to(at::kInt);
  ref_res = at::empty({23, 23});
  vec_res = at::empty({23, 23});
  int32_t* a_int_ptr = a.data_ptr<int32_t>();
  float* ref_res_float_ptr = ref_res.data_ptr<float>();
  float* vec_res_float_ptr = vec_res.data_ptr<float>();
  for(auto i = 0; i < a.numel(); ++i) {
    ref_res_float_ptr[i] = static_cast<float>(a_int_ptr[i]);
  }
  at::vec256::convert(a_int_ptr, vec_res_float_ptr, a.numel());
  ASSERT_TRUE(check_almost_equal(ref_res, vec_res, 1e-6));
}

TEST(Vec256TestFloat, check_fmadd) {
  at::Tensor a = at::rand({23, 23});
  a = a * -10;
  a = a + 10;
  at::Tensor b = at::rand({23, 23});
  b = b * -5;
  b = b + 5;
  at::Tensor c = at::rand({23, 23});
  c = c * 20;
  at::Tensor ref_res = at::zeros({23, 23});
  at::Tensor vec_res = at::zeros({23, 23});
  float* a_ptr = a.data_ptr<float>();
  float* b_ptr = a.data_ptr<float>();
  float* c_ptr = a.data_ptr<float>();
  float* ref_res_ptr = ref_res.data_ptr<float>();
  float* vec_res_ptr = vec_res.data_ptr<float>();
  size_t num_els =
    (a.numel() / Vec256<float>::size()) * Vec256<float>::size();
  for(auto i = 0; i < num_els; ++i) {
    ref_res_ptr[i] = a_ptr[i] * b_ptr[i] + c_ptr[i];
  }
  for (size_t i = 0; i < num_els; i += Vec256<float>::size()) {
    auto a_elements = Vec256<float>::loadu(a_ptr);
    auto b_elements = Vec256<float>::loadu(b_ptr);
    auto c_elements = Vec256<float>::loadu(c_ptr);
    a_ptr += Vec256<float>::size();
    b_ptr += Vec256<float>::size();
    c_ptr += Vec256<float>::size();
    auto res_elements = at::vec256::fmadd(a_elements, b_elements, c_elements);
    res_elements.store(vec_res_ptr);
    vec_res_ptr += Vec256<float>::size();
  }
  ASSERT_TRUE(check_almost_equal(ref_res, vec_res, 1e-6));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  at::manual_seed(42);
  return RUN_ALL_TESTS();
}
