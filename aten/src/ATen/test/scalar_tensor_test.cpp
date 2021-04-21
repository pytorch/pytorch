#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <c10/util/accumulate.h>

#include <algorithm>
#include <iostream>
#include <numeric>

using namespace at;

#define TRY_CATCH_ELSE(fn, catc, els)                          \
  {                                                            \
    /* avoid mistakenly passing if els code throws exception*/ \
    bool _passed = false;                                      \
    try {                                                      \
      fn;                                                      \
      _passed = true;                                          \
      els;                                                     \
    } catch (std::exception & e) {                             \
      ASSERT_FALSE(_passed);                                   \
      catc;                                                    \
    }                                                          \
  }

void require_equal_size_dim(const Tensor &lhs, const Tensor &rhs) {
  ASSERT_EQ(lhs.dim(), rhs.dim());
  ASSERT_TRUE(lhs.sizes().equals(rhs.sizes()));
}

bool should_expand(const IntArrayRef &from_size, const IntArrayRef &to_size) {
  if (from_size.size() > to_size.size()) {
    return false;
  }
  for (auto from_dim_it = from_size.rbegin(); from_dim_it != from_size.rend();
       ++from_dim_it) {
    for (auto to_dim_it = to_size.rbegin(); to_dim_it != to_size.rend();
         ++to_dim_it) {
      if (*from_dim_it != 1 && *from_dim_it != *to_dim_it) {
        return false;
      }
    }
  }
  return true;
}

void test(DeprecatedTypeProperties &T) {
  std::vector<std::vector<int64_t>> sizes = {{}, {0}, {1}, {1, 1}, {2}};

  // single-tensor/size tests
  for (auto s = sizes.begin(); s != sizes.end(); ++s) {
    // verify that the dim, sizes, strides, etc match what was requested.
    auto t = ones(*s, T);
    ASSERT_EQ((size_t)t.dim(), s->size());
    ASSERT_EQ((size_t)t.ndimension(), s->size());
    ASSERT_TRUE(t.sizes().equals(*s));
    ASSERT_EQ(t.strides().size(), s->size());
    const auto numel = c10::multiply_integers(s->begin(), s->end());
    ASSERT_EQ(t.numel(), numel);
    // verify we can output
    std::stringstream ss;
    ASSERT_NO_THROW(ss << t << std::endl);

    // set_
    auto t2 = ones(*s, T);
    t2.set_();
    require_equal_size_dim(t2, ones({0}, T));

    // unsqueeze
    ASSERT_EQ(t.unsqueeze(0).dim(), t.dim() + 1);

    // unsqueeze_
    {
      auto t2 = ones(*s, T);
      auto r = t2.unsqueeze_(0);
      ASSERT_EQ(r.dim(), t.dim() + 1);
    }

    // squeeze (with dimension argument)
    if (t.dim() == 0 || t.sizes()[0] == 1) {
      ASSERT_EQ(t.squeeze(0).dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      // In PyTorch, it is a no-op to try to squeeze a dimension that has size
      // != 1; in NumPy this is an error.
      ASSERT_EQ(t.squeeze(0).dim(), t.dim());
    }

    // squeeze (with no dimension argument)
    {
      std::vector<int64_t> size_without_ones;
      for (auto size : *s) {
        if (size != 1) {
          size_without_ones.push_back(size);
        }
      }
      auto result = t.squeeze();
      require_equal_size_dim(result, ones(size_without_ones, T));
    }

    {
      // squeeze_ (with dimension argument)
      auto t2 = ones(*s, T);
      if (t2.dim() == 0 || t2.sizes()[0] == 1) {
        ASSERT_EQ(t2.squeeze_(0).dim(), std::max<int64_t>(t.dim() - 1, 0));
      } else {
        // In PyTorch, it is a no-op to try to squeeze a dimension that has size
        // != 1; in NumPy this is an error.
        ASSERT_EQ(t2.squeeze_(0).dim(), t.dim());
      }
    }

    // squeeze_ (with no dimension argument)
    {
      auto t2 = ones(*s, T);
      std::vector<int64_t> size_without_ones;
      for (auto size : *s) {
        if (size != 1) {
          size_without_ones.push_back(size);
        }
      }
      auto r = t2.squeeze_();
      require_equal_size_dim(t2, ones(size_without_ones, T));
    }

    // reduce (with dimension argument and with 1 return argument)
    if (t.numel() != 0) {
      ASSERT_EQ(t.sum(0).dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      ASSERT_TRUE(t.sum(0).equal(at::zeros({}, T)));
    }

    // reduce (with dimension argument and with 2 return arguments)
    if (t.numel() != 0) {
      auto ret = t.min(0);
      ASSERT_EQ(std::get<0>(ret).dim(), std::max<int64_t>(t.dim() - 1, 0));
      ASSERT_EQ(std::get<1>(ret).dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      ASSERT_ANY_THROW(t.min(0));
    }

    // simple indexing
    if (t.dim() > 0 && t.numel() != 0) {
      ASSERT_EQ(t[0].dim(), std::max<int64_t>(t.dim() - 1, 0));
    } else {
      ASSERT_ANY_THROW(t[0]);
    }

    // fill_ (argument to fill_ can only be a 0-dim tensor)
    TRY_CATCH_ELSE(
        t.fill_(t.sum(0)), ASSERT_GT(t.dim(), 1), ASSERT_LE(t.dim(), 1));
  }

  for (auto lhs_it = sizes.begin(); lhs_it != sizes.end(); ++lhs_it) {
    for (auto rhs_it = sizes.begin(); rhs_it != sizes.end(); ++rhs_it) {
      // is_same_size should only match if they are the same shape
      {
        auto lhs = ones(*lhs_it, T);
        auto rhs = ones(*rhs_it, T);
        if (*lhs_it != *rhs_it) {
          ASSERT_FALSE(lhs.is_same_size(rhs));
          ASSERT_FALSE(rhs.is_same_size(lhs));
        }
      }
      // forced size functions (resize_, resize_as, set_)
      // resize_
      {
        {
         auto lhs = ones(*lhs_it, T);
         auto rhs = ones(*rhs_it, T);
         lhs.resize_(*rhs_it);
         require_equal_size_dim(lhs, rhs);
        }
        // resize_as_
        {
          auto lhs = ones(*lhs_it, T);
          auto rhs = ones(*rhs_it, T);
          lhs.resize_as_(rhs);
          require_equal_size_dim(lhs, rhs);
        }
        // set_
        {
          {
            // with tensor
            auto lhs = ones(*lhs_it, T);
            auto rhs = ones(*rhs_it, T);
            lhs.set_(rhs);
            require_equal_size_dim(lhs, rhs);
          }
          {
            // with storage
            auto lhs = ones(*lhs_it, T);
            auto rhs = ones(*rhs_it, T);
            lhs.set_(rhs.storage());
            // should not be dim 0 because an empty storage is dim 1; all other
            // storages aren't scalars
            ASSERT_NE(lhs.dim(), 0);
          }
          {
            // with storage, offset, sizes, strides
            auto lhs = ones(*lhs_it, T);
            auto rhs = ones(*rhs_it, T);
            lhs.set_(rhs.storage(), rhs.storage_offset(), rhs.sizes(), rhs.strides());
            require_equal_size_dim(lhs, rhs);
          }
        }
      }

      // view
      {
        auto lhs = ones(*lhs_it, T);
        auto rhs = ones(*rhs_it, T);
        auto rhs_size = *rhs_it;
        TRY_CATCH_ELSE(auto result = lhs.view(rhs_size),
                       ASSERT_NE(lhs.numel(), rhs.numel()),
                       ASSERT_EQ(lhs.numel(), rhs.numel());
                       require_equal_size_dim(result, rhs););
      }

      // take
      {
        auto lhs = ones(*lhs_it, T);
        auto rhs = zeros(*rhs_it, T).toType(ScalarType::Long);
        TRY_CATCH_ELSE(auto result = lhs.take(rhs),
                       ASSERT_EQ(lhs.numel(), 0); ASSERT_NE(rhs.numel(), 0),
                       require_equal_size_dim(result, rhs));
      }

      // put
      {
        auto lhs = ones(*lhs_it, T);
        auto rhs1 = zeros(*rhs_it, T).toType(ScalarType::Long);
        auto rhs2 = zeros(*rhs_it, T);
        TRY_CATCH_ELSE(auto result = lhs.put(rhs1, rhs2),
                       ASSERT_EQ(lhs.numel(), 0); ASSERT_NE(rhs1.numel(), 0),
                       require_equal_size_dim(result, lhs));
      }

      // ger
      {
        auto lhs = ones(*lhs_it, T);
        auto rhs = ones(*rhs_it, T);
        TRY_CATCH_ELSE(auto result = lhs.ger(rhs),
                       ASSERT_TRUE(
                           (lhs.numel() == 0 || rhs.numel() == 0 ||
                            lhs.dim() != 1 || rhs.dim() != 1)),
                       [&]() {
                         int64_t dim0 = lhs.dim() == 0 ? 1 : lhs.size(0);
                         int64_t dim1 = rhs.dim() == 0 ? 1 : rhs.size(0);
                         require_equal_size_dim(
                             result, at::empty({dim0, dim1}, result.options()));
                       }(););
      }

      // expand
      {
        auto lhs = ones(*lhs_it, T);
        auto lhs_size = *lhs_it;
        auto rhs = ones(*rhs_it, T);
        auto rhs_size = *rhs_it;
        bool should_pass = should_expand(lhs_size, rhs_size);
        TRY_CATCH_ELSE(auto result = lhs.expand(rhs_size),
                       ASSERT_FALSE(should_pass),
                       ASSERT_TRUE(should_pass);
                       require_equal_size_dim(result, rhs););

        // in-place functions (would be good if we can also do a non-broadcasting
        // one, b/c broadcasting functions will always end up operating on tensors
        // of same size; is there an example of this outside of assign_ ?)
        {
          bool should_pass_inplace = should_expand(rhs_size, lhs_size);
          TRY_CATCH_ELSE(lhs.add_(rhs),
                         ASSERT_FALSE(should_pass_inplace),
                         ASSERT_TRUE(should_pass_inplace);
                         require_equal_size_dim(lhs, ones(*lhs_it, T)););
        }
      }
    }
  }
}

TEST(TestScalarTensor, TestScalarTensorCPU) {
  manual_seed(123);
  test(CPU(kFloat));
}

TEST(TestScalarTensor, TestScalarTensorCUDA) {
  manual_seed(123);

  if (at::hasCUDA()) {
    test(CUDA(kFloat));
  }
}
