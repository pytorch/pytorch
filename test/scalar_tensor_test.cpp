#include "ATen/ATen.h"
#include <iostream>
#include <numeric>

using namespace at;

void assert_equal_size_dim(const Tensor &lhs, const Tensor &rhs) {
  assert(lhs.dim() == rhs.dim());
  assert(lhs.sizes().equals(rhs.sizes()));
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

int main() {
  Type & T = CPU(kFloat);

  std::vector<std::vector<int64_t> > sizes = { {}, {0}, {1}, {1, 1}, {2}};

  // single-tensor/size tests
  for (auto s = sizes.begin(); s != sizes.end(); ++s) {
    // verify that the dim, sizes, strides, etc match what was requested.
    auto t = T.ones(*s);
    assert(t.dim() == s->size());
    assert(t.ndimension() == s->size());
    assert(t.sizes().equals(*s));
    assert(t.strides().size() == s->size());
    auto numel = std::accumulate(s->begin(), s->end(), 1, std::multiplies<int64_t>());
    assert(t.numel() == numel);
    // verify we can output
    std::cout << t << std::endl;

    // set_
    auto t2 = T.ones(*s);
    t2.set_();
    assert_equal_size_dim(t2, T.ones({0}));

    // unsqueeze
    if (t.numel() != 0) {
      if (t.dim() > 0) {
        assert(t.unsqueeze(0).dim() == t.dim() + 1);
      } else {
        // FIXME: should be able to remove this if/else, unsqueezing a scalar should give 1-dimension
        assert(t.unsqueeze(0).dim() == t.dim() + 2);
      }
    } else {
      try {
        // can't unsqueeze empty tensor
        t.unsqueeze(0);
        assert (false);
      } catch (std::runtime_error &e) {}
    }

    // squeeze
    if (t.dim() > 0 && t.sizes()[0] == 1) {
      // FIXME: the max should be 0, but we don't reduce down to scalars properly yet
      assert(t.squeeze(0).dim() == std::max<int64_t>(t.dim() - 1, 1));
    } else if (t.dim() == 0 || t.numel() == 0)  {
      try {
        t.squeeze(0);
        assert(false);
      } catch (std::runtime_error &e) {}
    } else {
      // In PyTorch, it is a no-op to try to squeeze a dimension that has size != 1;
      // in NumPy this is an error.
      assert(t.squeeze(0).dim() == t.dim());
    }

    // reduce
    if (t.dim() > 0 && t.numel() != 0) {
      // FIXME: the max should be 0, but we don't reduce down to scalars properly yet
      assert(t.sum(0).dim() == std::max<int64_t>(t.dim() - 1, 1));
    } else if (t.dim() == 0) {
      try {
        t.sum(0);
        assert(false);
      } catch (std::runtime_error &e) {}
    } else {
      // FIXME: you should be able to reduce over size {0}
      try {
        t.sum(0);
        assert(false);
      } catch (std::runtime_error &e) {}
    }

    // simple indexing
    if (t.dim() > 0 && t.numel() != 0) {
      assert(t[0].dim() == std::max<int64_t>(t.dim() - 1, 0));
    } else if (t.dim() == 0) {
      try {
        t[0];
        assert(false);
      } catch (std::runtime_error &e) {}
    }
  }

  for (auto lhs_it = sizes.begin(); lhs_it != sizes.end(); ++lhs_it) {
    for (auto rhs_it = sizes.begin(); rhs_it != sizes.end(); ++rhs_it) {
      // is_same_size should only match if they are the same shape
      {
          auto lhs = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          if(*lhs_it != *rhs_it) {
            assert(!lhs.is_same_size(rhs));
            assert(!rhs.is_same_size(lhs));
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
            assert(lhs.dim() != 0);
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

        // assign_
        {
          auto lhs = T.ones(*lhs_it);
          auto lhs_save = T.ones(*lhs_it);
          auto rhs = T.ones(*rhs_it);
          try {
            lhs.assign_(rhs);
            assert(lhs_save.numel() == rhs.numel());
            // ensure didn't change shape
            assert_equal_size_dim(lhs, lhs_save);
          } catch (std::runtime_error &e) {
            assert(lhs_save.numel() != rhs.numel());
          }
        }
      }

      // view
      {
        auto lhs = T.ones(*lhs_it);
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        try {
          auto result = lhs.view(rhs_size);
          assert(lhs.numel() == rhs.numel());
          assert_equal_size_dim(result, rhs);
        } catch (std::runtime_error &e) {
          assert(lhs.numel() != rhs.numel());
        }
      }

      // expand
      {
        auto lhs = T.ones(*lhs_it);
        auto lhs_size = *lhs_it;
        auto rhs = T.ones(*rhs_it);
        auto rhs_size = *rhs_it;
        bool should_pass = should_expand(lhs_size, rhs_size);
        try {
          auto result = lhs.expand(rhs_size);
          assert(should_pass);
          assert_equal_size_dim(result, rhs);
        } catch (std::runtime_error &e) {
          assert(!should_pass);
        }

        // in-place functions (would be good if we can also do a non-broadcasting one, b/c
        // broadcasting functions will always end up operating on tensors of same size;
        // is there an example of this outside of assign_ ?)
        {
          bool should_pass_inplace = should_expand(rhs_size, lhs_size);
          try {
            lhs.add_(rhs);
            assert(should_pass_inplace);
            assert_equal_size_dim(lhs, T.ones(*lhs_it));
          } catch (std::runtime_error &e) {
            assert(!should_pass_inplace);
          }
        }
      }
    }
  }

  return 0;
}
