#include "ATen/ATen.h"
#include "test_assert.h"
#include <iostream>
#include <numeric>

using namespace at;

void assert_equal_size_dim(const Tensor &lhs, const Tensor &rhs) {
  ASSERT(lhs.dim() == rhs.dim());
  ASSERT(lhs.sizes().equals(rhs.sizes()));
}

bool should_expand(const IntList &from_size, const IntList &to_size) {
  if(from_size.size() > to_size.size()) {
    return false;
  }
  for (auto from_dim_it = from_size.rbegin(); from_dim_it != from_size.rend(); ++from_dim_it) {
    for (auto to_dim_it = to_size.rbegin(); to_dim_it != to_size.rend(); ++to_dim_it) {
      if (*from_dim_it != 1 && *from_dim_it != *to_dim_it) {
        return false;
      }
    }
  }
  return true;
}

void test(Type &T) {
  std::vector<std::vector<int64_t> > sizes = { {}, {0}, {1}, {1, 1}, {2}};

  // single-tensor/size tests
  for (auto s = sizes.begin(); s != sizes.end(); ++s) {
    // verify that the dim, sizes, strides, etc match what was requested.
    auto t = T.ones(*s);
    ASSERT((std::size_t)t.dim() == s->size());
    ASSERT((std::size_t)t.ndimension() == s->size());
    ASSERT(t.sizes().equals(*s));
    ASSERT(t.strides().size() == s->size());
    auto numel = std::accumulate(s->begin(), s->end(), 1, std::multiplies<int64_t>());
    ASSERT(t.numel() == numel);
    // verify we can output
    std::cout << t << std::endl;

    // set_
    auto t2 = T.ones(*s);
    t2.set_();
    assert_equal_size_dim(t2, T.ones({0}));

    // unsqueeze
    if (t.numel() != 0) {
      ASSERT(t.unsqueeze(0).dim() == t.dim() + 1);
    } else {
      ASSERT_THROWS(t.unsqueeze(0));
    }

    // unsqueeze_
    {
      auto t2 = T.ones(*s);
      if (t2.numel() != 0) {
        auto r = t2.unsqueeze_(0);
        ASSERT(r.dim() == t.dim() + 1);
      } else {
        ASSERT_THROWS(t2.unsqueeze_(0));
      }
    }

    // squeeze (with dimension argument)
    if (t.dim() > 0 && t.sizes()[0] == 1) {
      ASSERT(t.squeeze(0).dim() == t.dim() - 1);
    } else if (t.dim() == 0) {
      ASSERT_THROWS(t.squeeze(0));
    } else {
      // In PyTorch, it is a no-op to try to squeeze a dimension that has size != 1;
      // in NumPy this is an error.
      ASSERT(t.squeeze(0).dim() == t.dim());
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
      assert_equal_size_dim(result, T.ones(size_without_ones));
    }

    {
      // squeeze_ (with dimension argument)
      auto t2 = T.ones(*s);
      if (t2.dim() > 0 && t2.sizes()[0] == 1) {
        ASSERT(t2.squeeze_(0).dim() == t.dim() - 1);
      } else if (t2.dim() == 0) {
        ASSERT_THROWS(t2.squeeze_(0));
      } else {
        // In PyTorch, it is a no-op to try to squeeze a dimension that has size != 1;
        // in NumPy this is an error.
        ASSERT(t2.squeeze_(0).dim() == t.dim());
      }
    }

    // squeeze_ (with no dimension argument)
    {
      auto t2 = T.ones(*s);
      std::vector<int64_t> size_without_ones;
      for (auto size : *s) {
        if (size != 1) {
          size_without_ones.push_back(size);
        }
      }
      auto r = t2.squeeze_();
      assert_equal_size_dim(t2, T.ones(size_without_ones));
    }

    // reduce (with dimension argument and with 1 return argument)
    if (t.dim() > 0 && t.numel() != 0) {
      ASSERT(t.sum(0).dim() == t.dim() - 1);
    } else if (t.dim() == 0) {
      ASSERT_THROWS(t.sum(0));
    } else {
      // FIXME: you should be able to reduce over size {0}
      ASSERT_THROWS(t.sum(0));
    }

    // reduce (with dimension argument and with 2 return arguments)
    if (t.dim() > 0 && t.numel() != 0) {
      auto ret = t.min(0);
      ASSERT(std::get<0>(ret).dim() == t.dim() - 1);
      ASSERT(std::get<1>(ret).dim() == t.dim() - 1);
    } else if (t.dim() == 0) {
      ASSERT_THROWS(t.min(0));
    } else {
      // FIXME: you should be able to reduce over size {0}
      ASSERT_THROWS(t.min(0));
    }

    // simple indexing
    if (t.dim() > 0 && t.numel() != 0) {
      ASSERT(t[0].dim() == std::max<int64_t>(t.dim() - 1, 0));
    } else if (t.dim() == 0) {
      ASSERT_THROWS(t[0]);
    }

    // fill_
    if (t.dim() > 0 && t.numel() != 0) {
      // can only fill_ 0-dim tensors
      TRY_CATCH_ELSE(t.fill_(t.sum(0)),
                     ASSERT(t.dim() != 1),
                     ASSERT(t.dim() == 1));
    }
  }

  for (auto lhs_it = sizes.begin(); lhs_it != sizes.end(); ++lhs_it) {
    for (auto rhs_it = sizes.begin(); rhs_it != sizes.end(); ++rhs_it) {
      // is_same_size should only match if they are the same shape
      {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          if(*lhs_it != *rhs_it) {
            ASSERT(!lhs.is_same_size(rhs));
            ASSERT(!rhs.is_same_size(lhs));
          }
      }
      // forced size functions (resize_, resize_as, set_)
      {
        // resize_
        {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          lhs.resize_(*rhs_it);
          assert_equal_size_dim(lhs, rhs);
        }
        // resize_as_
        {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          lhs.resize_as_(rhs);
          assert_equal_size_dim(lhs, rhs);
        }
        // set_
        {
          {
            // with tensor
            auto lhs = T.ones(*lhs_it);
            auto rhs = T.ones(*rhs_it);
            lhs.set_(rhs);
            assert_equal_size_dim(lhs, rhs);
          }
          {
            // with storage
            auto lhs = T.ones(*lhs_it);
            auto rhs = T.ones(*rhs_it);
            auto storage = T.storage(rhs.numel());
            lhs.set_(*storage);
            // should not be dim 0 because an empty storage is dim 1; all other storages aren't scalars
            ASSERT(lhs.dim() != 0);
          }
          {
            // with storage, offset, sizes, strides
            auto lhs = T.ones(*lhs_it);
            auto rhs = T.ones(*rhs_it);
            auto storage = T.storage(rhs.numel());
            lhs.set_(*storage, rhs.storage_offset(), rhs.sizes(), rhs.strides());
            assert_equal_size_dim(lhs, rhs);
          }
        }
      }

      // view
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        TRY_CATCH_ELSE(auto result = lhs.view(rhs_size),
                       ASSERT(lhs.numel() != rhs.numel()),
                       ASSERT(lhs.numel() == rhs.numel()); assert_equal_size_dim(result, rhs););
      }

      // take
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.zeros(*rhs_it).toType(ScalarType::Long);
        TRY_CATCH_ELSE(auto result = lhs.take(rhs),
                       ASSERT(lhs.numel() == 0 && rhs.numel() != 0),
                       assert_equal_size_dim(result, rhs));
      }


      // ger
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.ones(*rhs_it);
        TRY_CATCH_ELSE(auto result = lhs.ger(rhs),
                       ASSERT(lhs.numel() == 0 || rhs.numel() == 0 || lhs.dim() > 1 || rhs.dim() > 1),
                       [&]() {
                         int64_t dim0 = lhs.dim() == 0 ? 1 : lhs.size(0);
                         int64_t dim1 = rhs.dim() == 0 ? 1 : rhs.size(0);
                         assert_equal_size_dim(result, result.type().tensor({dim0, dim1}));
                       }(););
      }

      // expand
      {
        auto lhs = T.ones(*lhs_it);
        auto lhs_size = *lhs_it;
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        bool should_pass = should_expand(lhs_size, rhs_size);
        TRY_CATCH_ELSE(auto result = lhs.expand(rhs_size),
                       ASSERT(!should_pass),
                       ASSERT(should_pass); assert_equal_size_dim(result, rhs););

        // in-place functions (would be good if we can also do a non-broadcasting one, b/c
        // broadcasting functions will always end up operating on tensors of same size;
        // is there an example of this outside of assign_ ?)
        {
          bool should_pass_inplace = should_expand(rhs_size, lhs_size);
          TRY_CATCH_ELSE(lhs.add_(rhs),
                         ASSERT(!should_pass_inplace),
                         ASSERT(should_pass_inplace); assert_equal_size_dim(lhs, T.ones(*lhs_it)););
        }
      }
    }
  }
}

int main() {
  test(CPU(kFloat));

  if (at::hasCUDA()) {
    test(CUDA(kFloat));
  }

  return 0;
}
