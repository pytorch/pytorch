#include <gtest/gtest.h>

#include <ATen/TensorIndexing.h>
#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::indexing;
using namespace torch::test;

TEST(TensorIndexingTest, TestIndexedTensorSavedHistory) {
  auto t = torch::zeros({4, 4, 4, 4});
  std::vector<TensorIndex> indices = t(None, "...", 0, true, {1, None, 2}, torch::tensor({1, 2})).indices();
  ASSERT_TRUE(indices[0].is_none());
  ASSERT_TRUE(indices[1].is_ellipsis());
  ASSERT_TRUE(indices[2].is_integer());
  ASSERT_TRUE(indices[2].integer() == 0);
  ASSERT_TRUE(indices[3].is_boolean());
  ASSERT_TRUE(indices[3].boolean() == true);
  ASSERT_TRUE(indices[4].is_slice());
  ASSERT_TRUE(indices[4].slice().start() == 1);
  ASSERT_TRUE(indices[4].slice().stop() == std::numeric_limits<int64_t>::max());
  ASSERT_TRUE(indices[4].slice().step() == 2);
  ASSERT_TRUE(indices[5].is_tensor());
  assert_equal(indices[5].tensor(), torch::tensor({1, 2}));

  assert_equal(t(None, "...", 0, true, {1, None, 2}, torch::tensor({1, 2})).original_tensor(), t);
}

void check_pass_as_reference(torch::Tensor & tensor) {
  ASSERT_TRUE(tensor.item<int>() == 1);
}

void check_pass_as_const_reference(const torch::Tensor & tensor) {
  ASSERT_TRUE(tensor.item<int>() == 1);
}

void check_pass_as_value(torch::Tensor tensor) {
  ASSERT_TRUE(tensor.item<int>() == 1);
}

TEST(TensorIndexingTest, TestIndexedTensorPassAsTensor) {
  at::IndexedTensor t = torch::tensor(1, torch::kInt)("...");
  check_pass_as_reference(t);
  check_pass_as_const_reference(t);
  check_pass_as_value(t);
}

/*
class TestIndexing(TestCase):
    def test_single_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))
*/
TEST(TensorIndexingTest, TestSingleInt) {
  auto v = torch::randn({5, 7, 3});
  assert_equal(v(4).sizes(), {7, 3});
}

/*
    def test_multiple_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))
*/
TEST(TensorIndexingTest, TestMultipleInt) {
  auto v = torch::randn({5, 7, 3});
  assert_equal(v(4).sizes(), {7, 3});
  assert_equal(v(4, {}, 1).sizes(), {7});
}

/*
    def test_none(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))
*/
TEST(TensorIndexingTest, TestNone) {
  auto v = torch::randn({5, 7, 3});
  assert_equal(v(None).sizes(), {1, 5, 7, 3});
  assert_equal(v({}, None).sizes(), {5, 1, 7, 3});
  assert_equal(v({}, None, None).sizes(), {5, 1, 1, 7, 3});
  assert_equal(v("...", None).sizes(), {5, 7, 3, 1});
}

/*
    def test_step(self):
        v = torch.arange(10)
        self.assertEqual(v[::1], v)
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        self.assertEqual(v[::11].tolist(), [0])
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])
*/
TEST(TensorIndexingTest, TestStep) {
  auto v = torch::arange(10, torch::kLong);  // yf225 TODO: wait for https://github.com/pytorch/pytorch/pull/27629 to merge first
  assert_equal(v({None, None, 1}), v);
  assert_equal(v({None, None, 2}), torch::tensor({0, 2, 4, 6, 8}));
  assert_equal(v({None, None, 3}), torch::tensor({0, 3, 6, 9}));
  assert_equal(v({None, None, 11}), torch::tensor({0}));
  assert_equal(v({1, 6, 2}), torch::tensor({1, 3, 5}));
}

/*
    def test_step_assignment(self):
        v = torch.zeros(4, 4)
        v[0, 1::2] = torch.tensor([3., 4.])
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        self.assertEqual(v[1:].sum(), 0)
*/
TEST(TensorIndexingTest, TestStepAssignment) {
  auto v = torch::zeros({4, 4});
  v(0, {1, None, 2}) = torch::tensor({3., 4.});
  assert_equal(v(0), torch::tensor({0., 3., 0., 4.}));
  ASSERT_TRUE(exactly_equal(v({1, None}).sum(), 0));
}

/*
    def test_bool_indices(self):
        v = torch.randn(5, 7, 3)
        boolIndices = torch.tensor([True, False, True, True, False], dtype=torch.bool)
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([True, False, True], dtype=torch.bool)
        boolIndices = torch.tensor([True, False, False], dtype=torch.bool)
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
            self.assertEqual(v[boolIndices], v[uint8Indices])
            self.assertEqual(v[boolIndices], tensor([True], dtype=torch.bool))
            self.assertEquals(len(w), 2)
*/
TEST(TensorIndexingTest, TestBoolIndices) {
  {
    auto v = torch::randn({5, 7, 3});
    auto boolIndices = torch::tensor({true, false, true, true, false}, torch::kBool);
    assert_equal(v(boolIndices).sizes(), {3, 7, 3});
    assert_equal(v(boolIndices), torch::stack({v(0), v(2), v(3)}));
  }
  {
    auto v = torch::tensor({true, false, true}, torch::kBool);
    auto boolIndices = torch::tensor({true, false, false}, torch::kBool);
    auto uint8Indices = torch::tensor({1, 0, 0}, torch::kUInt8);

    {
      std::stringstream buffer;
      CerrRedirect cerr_redirect(buffer.rdbuf());

      assert_equal(v(boolIndices).sizes(), v(uint8Indices).sizes());
      assert_equal(v(boolIndices), v(uint8Indices));
      assert_equal(v(boolIndices), torch::tensor({true}, torch::kBool));

      ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);
    }
  }
}

/*
    def test_bool_indices_accumulate(self):
        mask = torch.zeros(size=(10, ), dtype=torch.bool)
        y = torch.ones(size=(10, 10))
        y.index_put_((mask, ), y[mask], accumulate=True)
        self.assertEqual(y, torch.ones(size=(10, 10)))
*/
TEST(TensorIndexingTest, TestBoolIndicesAccumulate) {
  auto mask = torch::zeros({10}, torch::kBool);
  auto y = torch::ones({10, 10});
  y.index_put_({mask}, y(mask), /*accumulate=*/true);
  assert_equal(y, torch::ones({10, 10}));
}


/*
    def test_multiple_bool_indices(self):
        v = torch.randn(5, 7, 3)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool)
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))
*/
TEST(TensorIndexingTest, TestMultipleBoolIndices) {
  auto v = torch::randn({5, 7, 3});
  // note: these broadcast together and are transposed to the first dim
  auto mask1 = torch::tensor({1, 0, 1, 1, 0}, torch::kBool);
  auto mask2 = torch::tensor({1, 1, 1}, torch::kBool);
  assert_equal(v(mask1, {}, mask2).sizes(), {3, 7});
}


/*
    def test_byte_mask(self):
        v = torch.randn(5, 7, 3)
        mask = torch.ByteTensor([1, 0, 1, 1, 0])
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[mask].shape, (3, 7, 3))
            self.assertEqual(v[mask], torch.stack([v[0], v[2], v[3]]))
            self.assertEquals(len(w), 2)

        v = torch.tensor([1.])
        self.assertEqual(v[v == 0], torch.tensor([]))
*/
TEST(TensorIndexingTest, TestByteMask) {
  {
    auto v = torch::randn({5, 7, 3});
    auto mask = torch::tensor({1, 0, 1, 1, 0}, torch::kByte);
    {
      std::stringstream buffer;
      CerrRedirect cerr_redirect(buffer.rdbuf());

      assert_equal(v(mask).sizes(), {3, 7, 3});
      assert_equal(v(mask), torch::stack({v(0), v(2), v(3)}));

      ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);
    }
  }
  {
    auto v = torch::tensor({1.});
    assert_equal(v(v == 0), torch::randn({0}));
  }
}

/*
    def test_byte_mask_accumulate(self):
        mask = torch.zeros(size=(10, ), dtype=torch.uint8)
        y = torch.ones(size=(10, 10))
        with warnings.catch_warnings(record=True) as w:
            y.index_put_((mask, ), y[mask], accumulate=True)
            self.assertEqual(y, torch.ones(size=(10, 10)))
            self.assertEquals(len(w), 2)
*/
TEST(TensorIndexingTest, TestByteMaskAccumulate) {
  auto mask = torch::zeros({10}, torch::kUInt8);
  auto y = torch::ones({10, 10});
  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    y.index_put_({mask}, y(mask), /*accumulate=*/true);
    assert_equal(y, torch::ones({10, 10}));

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);
  }
}

/*
    def test_multiple_byte_mask(self):
        v = torch.randn(5, 7, 3)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.ByteTensor([1, 0, 1, 1, 0])
        mask2 = torch.ByteTensor([1, 1, 1])
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[mask1, :, mask2].shape, (3, 7))
            self.assertEquals(len(w), 2)
*/
TEST(TensorIndexingTest, TestMultipleByteMask) {
  auto v = torch::randn({5, 7, 3});
  // note: these broadcast together and are transposed to the first dim
  auto mask1 = torch::tensor({1, 0, 1, 1, 0}, torch::kByte);
  auto mask2 = torch::tensor({1, 1, 1}, torch::kByte);
  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    assert_equal(v(mask1, {}, mask2).sizes(), {3, 7});

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);
  }
}

/*
    def test_byte_mask2d(self):
        v = torch.randn(5, 7, 3)
        c = torch.randn(5, 7)
        num_ones = (c > 0).sum()
        r = v[c > 0]
        self.assertEqual(r.shape, (num_ones, 3))
*/
TEST(TensorIndexingTest, TestByteMask2d) {
  auto v = torch::randn({5, 7, 3});
  auto c = torch::randn({5, 7});
  int64_t num_ones = (c > 0).sum().item().to<int64_t>();
  auto r = v(c > 0);
  assert_equal(r.sizes(), {num_ones, 3});
}

/*
    def test_int_indices(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
        self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
        self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))
*/
TEST(TensorIndexingTest, TestIntIndices) {
  auto v = torch::randn({5, 7, 3});
  assert_equal(v(torch::tensor({0, 4, 2})).sizes(), {3, 7, 3});
  assert_equal(v({}, torch::tensor({0, 4, 2})).sizes(), {5, 3, 3});
  assert_equal(v({}, torch::tensor({{0, 1}, {4, 3}})).sizes(), {5, 2, 2, 3});
}


/*
    def test_int_indices2d(self):
        # From the NumPy indexing example
        x = torch.arange(0, 12).view(4, 3)
        rows = torch.tensor([[0, 0], [3, 3]])
        columns = torch.tensor([[0, 2], [0, 2]])
        self.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])
*/
TEST(TensorIndexingTest, TestIntIndices2d) {
  // From the NumPy indexing example
  auto x = torch::arange(0, 12, torch::kLong).view({4, 3});
  auto rows = torch::tensor({{0, 0}, {3, 3}});
  auto columns = torch::tensor({{0, 2}, {0, 2}});
  assert_equal(x(rows, columns), torch::tensor({{0, 2}, {9, 11}}));
}

/*
    def test_int_indices_broadcast(self):
        # From the NumPy indexing example
        x = torch.arange(0, 12).view(4, 3)
        rows = torch.tensor([0, 3])
        columns = torch.tensor([0, 2])
        result = x[rows[:, None], columns]
        self.assertEqual(result.tolist(), [[0, 2], [9, 11]])
*/
TEST(TensorIndexingTest, TestIntIndicesBroadcast) {
  // From the NumPy indexing example
  auto x = torch::arange(0, 12, torch::kLong).view({4, 3});
  auto rows = torch::tensor({0, 3});
  auto columns = torch::tensor({0, 2});
  auto result = x(rows({}, None), columns);
  assert_equal(result, torch::tensor({{0, 2}, {9, 11}}));
}

/*
    def test_empty_index(self):
        x = torch.arange(0, 12).view(4, 3)
        idx = torch.tensor([], dtype=torch.long)
        self.assertEqual(x[idx].numel(), 0)

        # empty assignment should have no effect but not throw an exception
        y = x.clone()
        y[idx] = -1
        self.assertEqual(x, y)

        mask = torch.zeros(4, 3).bool()
        y[mask] = -1
        self.assertEqual(x, y)
*/
TEST(TensorIndexingTest, TestEmptyIndex) {
  auto x = torch::arange(0, 12).view({4, 3});
  auto idx = torch::tensor({}, torch::kLong);
  ASSERT_EQ(x(idx).numel(), 0);

  // empty assignment should have no effect but not throw an exception
  auto y = x.clone();
  y(idx) = -1;
  assert_equal(x, y);

  auto mask = torch::zeros({4, 3}, torch::kBool);
  y(mask) = -1;
  assert_equal(x, y);
}

/*
    def test_empty_ndim_index(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(5, device=device)
            self.assertEqual(torch.empty(0, 2, device=device), x[torch.empty(0, 2, dtype=torch.int64, device=device)])

            x = torch.randn(2, 3, 4, 5, device=device)
            self.assertEqual(torch.empty(2, 0, 6, 4, 5, device=device),
                             x[:, torch.empty(0, 6, dtype=torch.int64, device=device)])

        x = torch.empty(10, 0)
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        self.assertEqual(x[[], []].shape, (0,))
        with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
            x[:, [0, 1]]
*/
TEST(TensorIndexingTest, TestEmptyNdimIndex) {
  torch::Device device(torch::kCPU);
  {
    auto x = torch::randn({5}, device);
    assert_equal(
      torch::empty({0, 2}, device),
      x(torch::empty({0, 2}, torch::TensorOptions(torch::kInt64).device(device))));
  }
  {
    auto x = torch::randn({2, 3, 4, 5}, device);
    assert_equal(
      torch::empty({2, 0, 6, 4, 5}, device),
      x({}, torch::empty({0, 6}, torch::TensorOptions(torch::kInt64).device(device))));
  }
  {
    auto x = torch::empty({10, 0});
    assert_equal(x(torch::tensor({1, 2})).sizes(), {2, 0});
    assert_equal(x(torch::tensor({}, torch::kLong), torch::tensor({}, torch::kLong)).sizes(), {0});
    ASSERT_THROWS_WITH(x({}, torch::tensor({0, 1})), "for dimension with size 0");
  }
}

/*
    def test_empty_ndim_index(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(5, device=device)
            self.assertEqual(torch.empty(0, 2, device=device), x[torch.empty(0, 2, dtype=torch.int64, device=device)])

            x = torch.randn(2, 3, 4, 5, device=device)
            self.assertEqual(torch.empty(2, 0, 6, 4, 5, device=device),
                             x[:, torch.empty(0, 6, dtype=torch.int64, device=device)])

        x = torch.empty(10, 0)
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        self.assertEqual(x[[], []].shape, (0,))
        with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
            x[:, [0, 1]]
*/
TEST(TensorIndexingTest, TestEmptyNdimIndex_CUDA) {
  torch::Device device(torch::kCUDA);
  {
    auto x = torch::randn({5}, device);
    assert_equal(
      torch::empty({0, 2}, device),
      x(torch::empty({0, 2}, torch::TensorOptions(torch::kInt64).device(device))));
  }
  {
    auto x = torch::randn({2, 3, 4, 5}, device);
    assert_equal(
      torch::empty({2, 0, 6, 4, 5}, device),
      x({}, torch::empty({0, 6}, torch::TensorOptions(torch::kInt64).device(device))));
  }
}


/*
    def test_empty_ndim_index_bool(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(5, device=device)
            self.assertRaises(IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)])
*/
TEST(TensorIndexingTest, TestEmptyNdimIndexBool) {
  torch::Device device(torch::kCPU);
  auto x = torch::randn({5}, device);
  ASSERT_THROW(x(torch::empty({0, 2}, torch::TensorOptions(torch::kUInt8).device(device))), c10::Error);
}

/*
    def test_empty_ndim_index_bool(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(5, device=device)
            self.assertRaises(IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)])
*/
TEST(TensorIndexingTest, TestEmptyNdimIndexBool_CUDA) {
  torch::Device device(torch::kCUDA);
  auto x = torch::randn({5}, device);
  ASSERT_THROW(x(torch::empty({0, 2}, torch::TensorOptions(torch::kUInt8).device(device))), c10::Error);
}

/*
    def test_empty_slice(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(2, 3, 4, 5, device=device)
            y = x[:, :, :, 1]
            z = y[:, 1:1, :]
            self.assertEqual((2, 0, 4), z.shape)
            # this isn't technically necessary, but matches NumPy stride calculations.
            self.assertEqual((60, 20, 5), z.stride())
            self.assertTrue(z.is_contiguous())
*/
TEST(TensorIndexingTest, TestEmptySlice) {
  torch::Device device(torch::kCPU);
  auto x = torch::randn({2, 3, 4, 5}, device);
  auto y = x({}, {}, {}, 1);
  auto z = y({}, {1, 1}, {});
  assert_equal({2, 0, 4}, z.sizes());
  // this isn't technically necessary, but matches NumPy stride calculations.
  assert_equal({60, 20, 5}, z.strides());
  ASSERT_TRUE(z.is_contiguous());
}

/*
    def test_empty_slice(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(2, 3, 4, 5, device=device)
            y = x[:, :, :, 1]
            z = y[:, 1:1, :]
            self.assertEqual((2, 0, 4), z.shape)
            # this isn't technically necessary, but matches NumPy stride calculations.
            self.assertEqual((60, 20, 5), z.stride())
            self.assertTrue(z.is_contiguous())
*/
TEST(TensorIndexingTest, TestEmptySlice_CUDA) {
  torch::Device device(torch::kCUDA);
  auto x = torch::randn({2, 3, 4, 5}, device);
  auto y = x({}, {}, {}, 1);
  auto z = y({}, {1, 1}, {});
  assert_equal({2, 0, 4}, z.sizes());
  // this isn't technically necessary, but matches NumPy stride calculations.
  assert_equal({60, 20, 5}, z.strides());
  ASSERT_TRUE(z.is_contiguous());
}

/*
    def test_index_getitem_copy_bools_slices(self):
        true = torch.tensor(1, dtype=torch.uint8)
        false = torch.tensor(0, dtype=torch.uint8)

        tensors = [torch.randn(2, 3), torch.tensor(3)]

        for a in tensors:
            self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[False])
            self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[false])
            self.assertEqual(a.data_ptr(), a[None].data_ptr())
            self.assertEqual(a.data_ptr(), a[...].data_ptr())
*/
TEST(TensorIndexingTest, TestIndexGetitemCopyBoolsSlices) {
  auto true_tensor = torch::tensor(1, torch::kUInt8);
  auto false_tensor = torch::tensor(0, torch::kUInt8);

  std::vector<torch::Tensor> tensors = {torch::randn({2, 3}), torch::tensor(3)};

  for (auto& a : tensors) {
    ASSERT_NE(a.data_ptr(), a(true).data_ptr());
    {
      std::vector<int64_t> sizes = {0};
      sizes.insert(sizes.end(), a.sizes().begin(), a.sizes().end());
      assert_equal(torch::empty(sizes), a(false));
    }
    ASSERT_NE(a.data_ptr(), a(true_tensor).data_ptr());
    {
      std::vector<int64_t> sizes = {0};
      sizes.insert(sizes.end(), a.sizes().begin(), a.sizes().end());
      assert_equal(torch::empty(sizes), a(false_tensor));
    }
    ASSERT_EQ(a.data_ptr(), a(None).data_ptr());
    ASSERT_EQ(a.data_ptr(), a("...").data_ptr());
  }
}

/*
    def test_index_setitem_bools_slices(self):
        true = torch.tensor(1, dtype=torch.uint8)
        false = torch.tensor(0, dtype=torch.uint8)

        tensors = [torch.randn(2, 3), torch.tensor(3)]

        for a in tensors:
            # prefix with a 1,1, to ensure we are compatible with numpy which cuts off prefix 1s
            # (some of these ops already prefix a 1 to the size)
            neg_ones = torch.ones_like(a) * -1
            neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
            a[True] = neg_ones_expanded
            self.assertEqual(a, neg_ones)
            a[False] = 5
            self.assertEqual(a, neg_ones)
            a[true] = neg_ones_expanded * 2
            self.assertEqual(a, neg_ones * 2)
            a[false] = 5
            self.assertEqual(a, neg_ones * 2)
            a[None] = neg_ones_expanded * 3
            self.assertEqual(a, neg_ones * 3)
            a[...] = neg_ones_expanded * 4
            self.assertEqual(a, neg_ones * 4)
            if a.dim() == 0:
                with self.assertRaises(IndexError):
                    a[:] = neg_ones_expanded * 5
*/
TEST(TensorIndexingTest, TestIndexSetitemBoolsSlices) {
  auto true_tensor = torch::tensor(1, torch::kUInt8);
  auto false_tensor = torch::tensor(0, torch::kUInt8);

  std::vector<torch::Tensor> tensors = {torch::randn({2, 3}), torch::tensor(3)};

  for (auto& a : tensors) {
    // prefix with a 1,1, to ensure we are compatible with numpy which cuts off prefix 1s
    // (some of these ops already prefix a 1 to the size)
    auto neg_ones = torch::ones_like(a) * -1;
    auto neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0);
    a(true) = neg_ones_expanded;
    assert_equal(a, neg_ones);
    a(false) = 5;
    assert_equal(a, neg_ones);
    a(true_tensor) = neg_ones_expanded * 2;
    assert_equal(a, neg_ones * 2);
    a(false_tensor) = 5;
    assert_equal(a, neg_ones * 2);
    a(None) = neg_ones_expanded * 3;
    assert_equal(a, neg_ones * 3);
    a("...") = neg_ones_expanded * 4;
    assert_equal(a, neg_ones * 4);
    if (a.dim() == 0) {
      ASSERT_THROW(a({}) = neg_ones_expanded * 5, c10::Error);
    }
  }
}

/*
    def test_index_scalar_with_bool_mask(self):
        for device in torch.testing.get_all_device_types():
            a = torch.tensor(1, device=device)
            uintMask = torch.tensor(True, dtype=torch.uint8, device=device)
            boolMask = torch.tensor(True, dtype=torch.bool, device=device)
            self.assertEqual(a[uintMask], a[boolMask])
            self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

            a = torch.tensor(True, dtype=torch.bool, device=device)
            self.assertEqual(a[uintMask], a[boolMask])
            self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)
*/
TEST(TensorIndexingTest, TestIndexScalarWithBoolMask) {
  torch::Device device(torch::kCPU);

  auto a = torch::tensor(1, device);
  auto uintMask = torch::tensor(true, torch::TensorOptions(torch::kUInt8).device(device));
  auto boolMask = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_equal(a(uintMask), a(boolMask));
  ASSERT_EQ(a(uintMask).dtype(), a(boolMask).dtype());

  a = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_equal(a(uintMask), a(boolMask));
  ASSERT_EQ(a(uintMask).dtype(), a(boolMask).dtype());
}

TEST(TensorIndexingTest, TestIndexScalarWithBoolMask_CUDA) {
  torch::Device device(torch::kCUDA);

  auto a = torch::tensor(1, device);
  auto uintMask = torch::tensor(true, torch::TensorOptions(torch::kUInt8).device(device));
  auto boolMask = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_equal(a(uintMask), a(boolMask));
  ASSERT_EQ(a(uintMask).dtype(), a(boolMask).dtype());

  a = torch::tensor(true, torch::TensorOptions(torch::kBool).device(device));
  assert_equal(a(uintMask), a(boolMask));
  ASSERT_EQ(a(uintMask).dtype(), a(boolMask).dtype());
}

/*
    def test_setitem_expansion_error(self):
        true = torch.tensor(True)
        a = torch.randn(2, 3)
        # check prefix with  non-1s doesn't work
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        # NumPy: ValueError
        with self.assertRaises(RuntimeError):
            a[True] = a_expanded
        with self.assertRaises(RuntimeError):
            a[true] = a_expanded
*/
// TEST(TensorIndexingTest, TestSetitemExpansionError) {
//   auto true_tensor = torch::tensor(true);
//   auto a = torch::randn({2, 3});
//   // check prefix with  non-1s doesn't work
//
//   auto a_expanded = a.expand(torch::utils::concat({5, 1}, a.sizes()));
//   // NumPy: ValueError
//   ASSERT_THROW(a(true) = a_expanded, c10::Error);
//   ASSERT_THROW(a(true_tensor) = a_expanded, c10::Error);
// }

/*
    def test_getitem_scalars(self):
        zero = torch.tensor(0, dtype=torch.int64)
        one = torch.tensor(1, dtype=torch.int64)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3)
        self.assertEqual(a[0], a[zero])
        self.assertEqual(a[0][1], a[zero][one])
        self.assertEqual(a[0, 1], a[zero, one])
        self.assertEqual(a[0, one], a[zero, 1])

        # indexing by a scalar should slice (not copy)
        self.assertEqual(a[0, 1].data_ptr(), a[zero, one].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.int()].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.short()].data_ptr())

        # scalar indexed with scalar
        r = torch.randn(())
        with self.assertRaises(IndexError):
            r[:]
        with self.assertRaises(IndexError):
            r[zero]
        self.assertEqual(r, r[...])
*/
TEST(TensorIndexingTest, TestGetitemScalars) {
  auto zero = torch::tensor(0, torch::kInt64);
  auto one = torch::tensor(1, torch::kInt64);

  // non-scalar indexed with scalars
  auto a = torch::randn({2, 3});
  assert_equal(a(0), a(zero));
  assert_equal(a(0)(1), a(zero)(one));
  assert_equal(a(0, 1), a(zero, one));
  assert_equal(a(0, one), a(zero, 1));

  // indexing by a scalar should slice (not copy)
  ASSERT_EQ(a(0, 1).data_ptr(), a(zero, one).data_ptr());
  ASSERT_EQ(a(1).data_ptr(), a(one.to(torch::kInt)).data_ptr());
  ASSERT_EQ(a(1).data_ptr(), a(one.to(torch::kShort)).data_ptr());

  // scalar indexed with scalar
  auto r = torch::randn({});
  ASSERT_THROW(r({}), c10::Error);
  ASSERT_THROW(r(zero), c10::Error);
  assert_equal(r, r("..."));
}

/*
    def test_setitem_scalars(self):
        zero = torch.tensor(0, dtype=torch.int64)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3)
        a_set_with_number = a.clone()
        a_set_with_scalar = a.clone()
        b = torch.randn(3)

        a_set_with_number[0] = b
        a_set_with_scalar[zero] = b
        self.assertEqual(a_set_with_number, a_set_with_scalar)
        a[1, zero] = 7.7
        self.assertEqual(7.7, a[1, 0])

        # scalar indexed with scalars
        r = torch.randn(())
        with self.assertRaises(IndexError):
            r[:] = 8.8
        with self.assertRaises(IndexError):
            r[zero] = 8.8
        r[...] = 9.9
        self.assertEqual(9.9, r)
*/
TEST(TensorIndexingTest, TestSetitemScalars) {
  auto zero = torch::tensor(0, torch::kInt64);

  // non-scalar indexed with scalars
  auto a = torch::randn({2, 3});
  auto a_set_with_number = a.clone();
  auto a_set_with_scalar = a.clone();
  auto b = torch::randn({3});

  a_set_with_number(0) = b;
  a_set_with_scalar(zero) = b;
  assert_equal(a_set_with_number, a_set_with_scalar);
  a(1, zero) = 7.7;
  ASSERT_TRUE(almost_equal(a(1, 0), 7.7));

  // scalar indexed with scalars
  auto r = torch::randn({});
  ASSERT_THROW(r({}) = 8.8, c10::Error);
  ASSERT_THROW(r(zero) = 8.8, c10::Error);
  r("...") = 9.9;
  ASSERT_TRUE(almost_equal(r, 9.9));
}

/*
    def test_basic_advanced_combined(self):
        # From the NumPy indexing example
        x = torch.arange(0, 12).view(4, 3)
        self.assertEqual(x[1:2, 1:3], x[1:2, [1, 2]])
        self.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

        # Check that it is a copy
        unmodified = x.clone()
        x[1:2, [1, 2]].zero_()
        self.assertEqual(x, unmodified)

        # But assignment should modify the original
        unmodified = x.clone()
        x[1:2, [1, 2]] = 0
        self.assertNotEqual(x, unmodified)
*/
TEST(TensorIndexingTest, TestBasicAdvancedCombined) {
  // From the NumPy indexing example
  auto x = torch::arange(0, 12).to(torch::kLong).view({4, 3});
  assert_equal(x({1, 2}, {1, 3}), x({1, 2}, torch::tensor({1, 2})));
  assert_equal(x({1, 2}, {1, 3}), torch::tensor({{4, 5}}));

  // Check that it is a copy
  {
    auto unmodified = x.clone();
    x({1, 2}, torch::tensor({1, 2})).zero_();
    assert_equal(x, unmodified);
  }

  // But assignment should modify the original
  /* yf225 TODO analysis:
`x({1, 2}, torch::tensor({1, 2})) = 0;` is not able to modify x, because there is no way to call `index_put_(indices, value)` in `Tensor::operator=(Scalar v) &&` since we don't know the indices
There are a few possible solutions to this:
1. Subclass `Tensor` and have an `IndexedTensor` class, which has member field that stores indices information. And we overload its `Tensor::operator=(Scalar v) &&` method to take the member field into consideration.
`Tensor::operator()` always returns `IndexedTensor`. Note that object slicing when being passed around as `Tensor` is okay, because it only needs to work for the `x({1, 2}, torch::tensor({1, 2})) = 0;` use case.
Pro: No performance penalty or extra memory allocation for existing code paths
Con: More complex class hierarchy
2. Have a thread_local dict that stores Tensor -> indices information, and check it in `Tensor::operator=(Scalar v) &&`
Pro: Always return Tensor type from `Tensor::operator()`
Con: Slow down `Tensor::operator=(Scalar v) &&` because of the extra check (which is really bad!!)
3. Ask people to use tensor.index(indices) / tensor.index_put_(indices, value) APIs instead
Pro: No need for complex class hierarchy or performance penalty. Clear semantics.
Con:
  1. API inconsistency with non-Tensor indexing use cases. (However, if we use this API also for non-Tensor indexing use cases, then this point is moot.)
  2. Much more verbose than the `operator()` API.

yf225 TODO: for now, I will choose the first approach, to see how it goes.
  */
  {
    auto unmodified = x.clone();
    x({1, 2}, torch::tensor({1, 2})) = 0;
    assert_not_equal(x, unmodified);
  }
}

/*
    def test_int_assignment(self):
        x = torch.arange(0, 4).view(2, 2)
        x[1] = 5
        self.assertEqual(x.tolist(), [[0, 1], [5, 5]])

        x = torch.arange(0, 4).view(2, 2)
        x[1] = torch.arange(5, 7)
        self.assertEqual(x.tolist(), [[0, 1], [5, 6]])
*/
TEST(TensorIndexingTest, TestIntAssignment) {
  {
    auto x = torch::arange(0, 4).to(torch::kLong).view({2, 2});
    x(1) = 5;
    assert_equal(x, torch::tensor({{0, 1}, {5, 5}}));
  }

  {
    auto x = torch::arange(0, 4).to(torch::kLong).view({2, 2});
    x(1) = torch::arange(5, 7).to(torch::kLong);
    assert_equal(x, torch::tensor({{0, 1}, {5, 6}}));
  }
}


/*
    def test_byte_tensor_assignment(self):
        x = torch.arange(0., 16).view(4, 4)
        b = torch.ByteTensor([True, False, True, False])
        value = torch.tensor([3., 4., 5., 6.])

        with warnings.catch_warnings(record=True) as w:
            x[b] = value
            self.assertEquals(len(w), 1)

        self.assertEqual(x[0], value)
        self.assertEqual(x[1], torch.arange(4, 8))
        self.assertEqual(x[2], value)
        self.assertEqual(x[3], torch.arange(12, 16))
*/
TEST(TensorIndexingTest, TestByteTensorAssignment) {
  auto x = torch::arange(0., 16).to(torch::kFloat).view({4, 4});
  auto b = torch::tensor({true, false, true, false}, torch::kByte);
  auto value = torch::tensor({3., 4., 5., 6.});

  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    x(b) = value;

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);  // yf225 TODO: this is changed from 1 to 2, likely because our implementation of `x(b) = value` (aka. through IndexedTensor) needs to index the tensor twice
  }

  assert_equal(x(0), value);
  assert_equal(x(1), torch::arange(4, 8).to(torch::kLong));
  assert_equal(x(2), value);
  assert_equal(x(3), torch::arange(12, 16).to(torch::kLong));
}

/*
    def test_variable_slicing(self):
        x = torch.arange(0, 16).view(4, 4)
        indices = torch.IntTensor([0, 1])
        i, j = indices
        self.assertEqual(x[i:j], x[0:1])
*/
TEST(TensorIndexingTest, TestVariableSlicing) {
  auto x = torch::arange(0, 16).view({4, 4});
  auto indices = torch::tensor({0, 1}, torch::kInt);
  int i = indices[0].item<int>();
  int j = indices[1].item<int>();
  assert_equal(x({i, j}), x({0, 1}));
}

/*
    def test_ellipsis_tensor(self):
        x = torch.arange(0, 9).view(3, 3)
        idx = torch.tensor([0, 2])
        self.assertEqual(x[..., idx].tolist(), [[0, 2],
                                                [3, 5],
                                                [6, 8]])
        self.assertEqual(x[idx, ...].tolist(), [[0, 1, 2],
                                                [6, 7, 8]])
*/
TEST(TensorIndexingTest, TestEllipsisTensor) {
  auto x = torch::arange(0, 9).to(torch::kLong).view({3, 3});
  auto idx = torch::tensor({0, 2});
  assert_equal(x("...", idx), torch::tensor({{0, 2},
                                             {3, 5},
                                             {6, 8}}));
  assert_equal(x(idx, "..."), torch::tensor({{0, 1, 2},
                                             {6, 7, 8}}));
}

/*
    def test_out_of_bound_index(self):
        x = torch.arange(0, 100).view(2, 5, 10)
        self.assertRaisesRegex(IndexError, 'index 5 is out of bounds for dimension 1 with size 5', lambda: x[0, 5])
        self.assertRaisesRegex(IndexError, 'index 4 is out of bounds for dimension 0 with size 2', lambda: x[4, 5])
        self.assertRaisesRegex(IndexError, 'index 15 is out of bounds for dimension 2 with size 10',
                               lambda: x[0, 1, 15])
        self.assertRaisesRegex(IndexError, 'index 12 is out of bounds for dimension 2 with size 10',
                               lambda: x[:, :, 12])
*/
TEST(TensorIndexingTest, TestOutOfBoundIndex) {
  auto x = torch::arange(0, 100).view({2, 5, 10});
  ASSERT_THROWS_WITH(x(0, 5), "index 5 is out of bounds for dimension 1 with size 5");
  ASSERT_THROWS_WITH(x(4, 5), "index 4 is out of bounds for dimension 0 with size 2");
  ASSERT_THROWS_WITH(x(0, 1, 15), "index 15 is out of bounds for dimension 2 with size 10");
  ASSERT_THROWS_WITH(x({}, {}, 12), "index 12 is out of bounds for dimension 2 with size 10");
}

/*
    def test_zero_dim_index(self):
        x = torch.tensor(10)
        self.assertEqual(x, x.item())

        def runner():
            print(x[0])
            return x[0]

        self.assertRaisesRegex(IndexError, 'invalid index', runner)
*/
TEST(TensorIndexingTest, TestZeroDimIndex) {
  auto x = torch::tensor(10);
  ASSERT_TRUE(exactly_equal(x, x.item<int64_t>()));

  auto runner = [&]() -> torch::Tensor {
    std::cout << x(0) << std::endl;
    return x(0);
  };

  ASSERT_THROWS_WITH(runner(), "invalid index");
}

// The tests below are from NumPy test_indexing.py with some modifications to
// make them compatible with libtorch. It's licensed under the BDS license below:
//
// Copyright (c) 2005-2017, NumPy Developers.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided
//        with the distribution.
//
//     * Neither the name of the NumPy Developers nor the names of any
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
    def test_none_index(self):
        # `None` index adds newaxis
        a = tensor([1, 2, 3])
        self.assertEqual(a[None].dim(), a.dim() + 1)
*/
TEST(NumpyTests, TestNoneIndex) {
  // `None` index adds newaxis
  auto a = torch::tensor({1, 2, 3});
  ASSERT_EQ(a(None).dim(), a.dim() + 1);
}

/*
    def test_empty_fancy_index(self):
        # Empty list index creates an empty array
        a = tensor([1, 2, 3])
        self.assertEqual(a[[]], torch.tensor([]))

        b = tensor([]).long()
        self.assertEqual(a[[]], torch.tensor([], dtype=torch.long))

        b = tensor([]).float()
        self.assertRaises(IndexError, lambda: a[b])
*/
TEST(NumpyTests, TestEmptyFancyIndex) {
  // Empty list index creates an empty array
  auto a = torch::tensor({1, 2, 3});
  assert_equal(a(torch::tensor({}, torch::kLong)), torch::tensor({}));

  auto b = torch::tensor({}).to(torch::kLong);
  assert_equal(a(torch::tensor({}, torch::kLong)), torch::tensor({}, torch::kLong));

  b = torch::tensor({}).to(torch::kFloat);
  ASSERT_THROW(a(b), c10::Error);
}

/*
    def test_ellipsis_index(self):
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        self.assertIsNot(a[...], a)
        self.assertEqual(a[...], a)
        # `a[...]` was `a` in numpy <1.9.
        self.assertEqual(a[...].data_ptr(), a.data_ptr())

        # Slicing with ellipsis can skip an
        # arbitrary number of dimensions
        self.assertEqual(a[0, ...], a[0])
        self.assertEqual(a[0, ...], a[0, :])
        self.assertEqual(a[..., 0], a[:, 0])

        # In NumPy, slicing with ellipsis results in a 0-dim array. In PyTorch
        # we don't have separate 0-dim arrays and scalars.
        self.assertEqual(a[0, ..., 1], torch.tensor(2))

        # Assignment with `(Ellipsis,)` on 0-d arrays
        b = torch.tensor(1)
        b[(Ellipsis,)] = 2
        self.assertEqual(b, 2)
*/
TEST(NumpyTests, TestEllipsisIndex) {
  auto a = torch::tensor({{1, 2, 3},
                          {4, 5, 6},
                          {7, 8, 9}});
  assert_is_not(a("..."), a);
  assert_equal(a("..."), a);
  // `a[...]` was `a` in numpy <1.9.
  ASSERT_EQ(a("...").data_ptr(), a.data_ptr());

  // Slicing with ellipsis can skip an
  // arbitrary number of dimensions
  assert_equal(a(0, "..."), a(0));
  assert_equal(a(0, "..."), a(0, {}));
  assert_equal(a("...", 0), a({}, 0));

  // In NumPy, slicing with ellipsis results in a 0-dim array. In PyTorch
  // we don't have separate 0-dim arrays and scalars.
  assert_equal(a(0, "...", 1), torch::tensor(2));

  // Assignment with `Ellipsis` on 0-d arrays
  auto b = torch::tensor(1);
  b(Ellipsis) = 2;
  ASSERT_EQ(b.item<int64_t>(), 2);
}

/*
    def test_single_int_index(self):
        # Single integer index selects one row
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        self.assertEqual(a[0], [1, 2, 3])
        self.assertEqual(a[-1], [7, 8, 9])

        # Index out of bounds produces IndexError
        self.assertRaises(IndexError, a.__getitem__, 1 << 30)
        # Index overflow produces Exception  NB: different exception type
        self.assertRaises(Exception, a.__getitem__, 1 << 64)
*/
TEST(NumpyTests, TestSingleIntIndex) {
  // Single integer index selects one row
  auto a = torch::tensor({{1, 2, 3},
                          {4, 5, 6},
                          {7, 8, 9}});

  assert_equal(a(0), torch::tensor({1, 2, 3}));
  assert_equal(a(-1), torch::tensor({7, 8, 9}));

  // Index out of bounds produces IndexError
  ASSERT_THROW(a(1 << 30), c10::Error);
  // NOTE: According to the standard (http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0543r0.html),
  // for signed integers, if during the evaluation of an expression, the result is not mathematically defined
  // or not in the range of representable values for its type, the behavior is undefined.
  // Therefore, there is no way to check for index overflow case because it might not throw exception.
  // yf225 TODO: should we actually throw exception??? think about how to do it.
  // ASSERT_THROW(a(1 << 64), c10::Error);
}

/*
    def test_single_bool_index(self):
        # Single boolean index
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        self.assertEqual(a[True], a[None])
        self.assertEqual(a[False], a[None][0:0])
*/
TEST(NumpyTests, TestSingleBoolIndex) {
  // Single boolean index
  auto a = torch::tensor({{1, 2, 3},
                   {4, 5, 6},
                   {7, 8, 9}});

  assert_equal(a(true), a(None));
  assert_equal(a(false), a(None)({0, 0}));
}

/*
    def test_boolean_shape_mismatch(self):
        arr = torch.ones((5, 4, 3))

        index = tensor([True])
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

        index = tensor([False] * 6)
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

        with warnings.catch_warnings(record=True) as w:
            index = torch.ByteTensor(4, 4).zero_()
            self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])
            self.assertRaisesRegex(IndexError, 'mask', lambda: arr[(slice(None), index)])
            self.assertEquals(len(w), 2)
*/
TEST(NumpyTests, TestBooleanShapeMismatch) {
  auto arr = torch::ones({5, 4, 3});

  auto index = torch::tensor({true});
  ASSERT_THROWS_WITH(arr(index), "mask");

  index = torch::tensor({false, false, false, false, false, false});
  ASSERT_THROWS_WITH(arr(index), "mask");

  {
    std::stringstream buffer;
    CerrRedirect cerr_redirect(buffer.rdbuf());

    index = torch::empty({4, 4}, torch::kByte).zero_();
    ASSERT_THROWS_WITH(arr(index), "mask");
    ASSERT_THROWS_WITH(arr({}, index), "mask");

    ASSERT_EQ(count_substr_occurrences(buffer.str(), "indexing with dtype torch.uint8 is now deprecated"), 2);
  }
}

/*
    def test_boolean_indexing_onedim(self):
        # Indexing a 2-dimensional array with
        # boolean array of length one
        a = tensor([[0., 0., 0.]])
        b = tensor([True])
        self.assertEqual(a[b], a)
        # boolean assignment
        a[b] = 1.
        self.assertEqual(a, tensor([[1., 1., 1.]]))
*/
TEST(NumpyTests, TestBooleanIndexingOnedim) {
  // Indexing a 2-dimensional array with
  // boolean array of length one
  auto a = torch::tensor({{0., 0., 0.}});
  auto b = torch::tensor({true});
  assert_equal(a(b), a);
  // boolean assignment
  a(b) = 1.;
  assert_equal(a, torch::tensor({{1., 1., 1.}}));
}

/*
    def test_boolean_assignment_value_mismatch(self):
        # A boolean assignment should fail when the shape of the values
        # cannot be broadcast to the subscription. (see also gh-3458)
        a = torch.arange(0, 4)

        def f(a, v):
            a[a > -1] = tensor(v)

        self.assertRaisesRegex(Exception, 'shape mismatch', f, a, [])
        self.assertRaisesRegex(Exception, 'shape mismatch', f, a, [1, 2, 3])
        self.assertRaisesRegex(Exception, 'shape mismatch', f, a[:1], [1, 2, 3])
*/
TEST(NumpyTests, TestBooleanAssignmentValueMismatch) {
  // A boolean assignment should fail when the shape of the values
  // cannot be broadcast to the subscription. (see also gh-3458)
  auto a = torch::arange(0, 4);

  auto f = [](torch::Tensor a, std::vector<int64_t> v) -> void {
    a(a > -1) = torch::tensor(v);
  };

  ASSERT_THROWS_WITH(f(a, {}), "shape mismatch");
  ASSERT_THROWS_WITH(f(a, {1, 2, 3}), "shape mismatch");
  ASSERT_THROWS_WITH(f(a({None, 1}), {1, 2, 3}), "shape mismatch");
}

/*
    def test_boolean_indexing_twodim(self):
        # Indexing a 2-dimensional array with
        # 2-dimensional boolean array
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        b = tensor([[True, False, True],
                    [False, True, False],
                    [True, False, True]])
        self.assertEqual(a[b], tensor([1, 3, 5, 7, 9]))
        self.assertEqual(a[b[1]], tensor([[4, 5, 6]]))
        self.assertEqual(a[b[0]], a[b[2]])

        # boolean assignment
        a[b] = 0
        self.assertEqual(a, tensor([[0, 2, 0],
                                    [4, 0, 6],
                                    [0, 8, 0]]))
*/
TEST(NumpyTests, TestBooleanIndexingTwodim) {
  // Indexing a 2-dimensional array with
  // 2-dimensional boolean array
  auto a = torch::tensor({{1, 2, 3},
                          {4, 5, 6},
                          {7, 8, 9}});
  auto b = torch::tensor({{true, false, true},
                          {false, true, false},
                          {true, false, true}});
  assert_equal(a(b), torch::tensor({1, 3, 5, 7, 9}));
  assert_equal(a(b(1)), torch::tensor({{4, 5, 6}}));
  assert_equal(a(b(0)), a(b(2)));

  // boolean assignment
  a(b) = 0;
  assert_equal(a, torch::tensor({{0, 2, 0},
                                 {4, 0, 6},
                                 {0, 8, 0}}));
}

/*
    def test_boolean_indexing_weirdness(self):
        # Weird boolean indexing things
        a = torch.ones((2, 3, 4))
        self.assertEqual((0, 2, 3, 4), a[False, True, ...].shape)
        self.assertEqual(torch.ones(1, 2), a[True, [0, 1], True, True, [1], [[2]]])
        self.assertRaises(IndexError, lambda: a[False, [0, 1], ...])
*/
TEST(NumpyTests, TestBooleanIndexingWeirdness) {
  // Weird boolean indexing things
  auto a = torch::ones({2, 3, 4});
  assert_equal(a(false, true, "...").sizes(), {0, 2, 3, 4});
  assert_equal(torch::ones({1, 2}), a(true, torch::tensor({0, 1}), true, true, torch::tensor({1}), torch::tensor({{2}})));
  ASSERT_THROW(a(false, torch::tensor({0, 1}), "..."), c10::Error);
}

/*
    def test_boolean_indexing_weirdness_tensors(self):
        # Weird boolean indexing things
        false = torch.tensor(False)
        true = torch.tensor(True)
        a = torch.ones((2, 3, 4))
        self.assertEqual((0, 2, 3, 4), a[False, True, ...].shape)
        self.assertEqual(torch.ones(1, 2), a[true, [0, 1], true, true, [1], [[2]]])
        self.assertRaises(IndexError, lambda: a[false, [0, 1], ...])
*/
TEST(NumpyTests, TestBooleanIndexingWeirdnessTensors) {
  // Weird boolean indexing things
  auto false_tensor = torch::tensor(false);
  auto true_tensor = torch::tensor(true);
  auto a = torch::ones({2, 3, 4});
  assert_equal(a(false, true, "...").sizes(), {0, 2, 3, 4});
  assert_equal(torch::ones({1, 2}), a(true_tensor, torch::tensor({0, 1}), true_tensor, true_tensor, torch::tensor({1}), torch::tensor({{2}})));
  ASSERT_THROW(a(false_tensor, torch::tensor({0, 1}), "..."), c10::Error);
}

/*
    def test_boolean_indexing_alldims(self):
        true = torch.tensor(True)
        a = torch.ones((2, 3))
        self.assertEqual((1, 2, 3), a[True, True].shape)
        self.assertEqual((1, 2, 3), a[true, true].shape)
*/
TEST(NumpyTests, TestBooleanIndexingAlldims) {
  auto true_tensor = torch::tensor(true);
  auto a = torch::ones({2, 3});
  assert_equal(a(true, true).sizes(), {1, 2, 3});
  assert_equal(a(true_tensor, true_tensor).sizes(), {1, 2, 3});
}

/*
    def test_boolean_list_indexing(self):
        # Indexing a 2-dimensional array with
        # boolean lists
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        b = [True, False, False]
        c = [True, True, False]
        self.assertEqual(a[b], tensor([[1, 2, 3]]))
        self.assertEqual(a[b, b], tensor([1]))
        self.assertEqual(a[c], tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(a[c, c], tensor([1, 5]))
*/
TEST(NumpyTests, TestBooleanListIndexing) {
  // Indexing a 2-dimensional array with
  // boolean lists
  auto a = torch::tensor({{1, 2, 3},
                          {4, 5, 6},
                          {7, 8, 9}});
  auto b = torch::tensor({true, false, false});
  auto c = torch::tensor({true, true, false});
  assert_equal(a(b), torch::tensor({{1, 2, 3}}));
  assert_equal(a(b, b), torch::tensor({1}));
  assert_equal(a(c), torch::tensor({{1, 2, 3}, {4, 5, 6}}));
  assert_equal(a(c, c), torch::tensor({1, 5}));
}

/*
    def test_everything_returns_views(self):
        # Before `...` would return a itself.
        a = tensor([5])

        self.assertIsNot(a, a[()])
        self.assertIsNot(a, a[...])
        self.assertIsNot(a, a[:])
*/
TEST(NumpyTests, TestEverythingReturnsViews) {
  // Before `...` would return a itself.
  auto a = torch::tensor({5});

  assert_is_not(a, a("..."));
  assert_is_not(a, a({}));
}


/*
    def test_broaderrors_indexing(self):
        a = torch.zeros(5, 5)
        self.assertRaisesRegex(IndexError, 'shape mismatch', a.__getitem__, ([0, 1], [0, 1, 2]))
        self.assertRaisesRegex(IndexError, 'shape mismatch', a.__setitem__, ([0, 1], [0, 1, 2]), 0)
*/
TEST(NumpyTests, TestBroaderrorsIndexing) {
  auto a = torch::zeros({5, 5});
  ASSERT_THROW(a(torch::tensor({0, 1}), torch::tensor({0, 1, 2})), c10::Error);
  ASSERT_THROW(a(torch::tensor({0, 1}), torch::tensor({0, 1, 2})) = 0, c10::Error);
}

/*
    def test_trivial_fancy_out_of_bounds(self):
        a = torch.zeros(5)
        ind = torch.ones(20, dtype=torch.int64)
        if a.is_cuda:
            raise unittest.SkipTest('CUDA asserts instead of raising an exception')
        ind[-1] = 10
        self.assertRaises(IndexError, a.__getitem__, ind)
        self.assertRaises(IndexError, a.__setitem__, ind, 0)
        ind = torch.ones(20, dtype=torch.int64)
        ind[0] = 11
        self.assertRaises(IndexError, a.__getitem__, ind)
        self.assertRaises(IndexError, a.__setitem__, ind, 0)
*/
TEST(NumpyTests, TestTrivialFancyOutOfBounds) {
  auto a = torch::zeros({5});
  auto ind = torch::ones({20}, torch::kInt64);
  ind(-1) = 10;
  ASSERT_THROW(a(ind), c10::Error);
  ASSERT_THROW(a(ind) = 0, c10::Error);
  ind = torch::ones({20}, torch::kInt64);
  ind(0) = 11;
  ASSERT_THROW(a(ind), c10::Error);
  ASSERT_THROW(a(ind) = 0, c10::Error);
}

/*
    def test_index_is_larger(self):
        # Simple case of fancy index broadcasting of the index.
        a = torch.zeros((5, 5))
        a[[[0], [1], [2]], [0, 1, 2]] = tensor([2., 3., 4.])

        self.assertTrue((a[:3, :3] == tensor([2., 3., 4.])).all())
*/
TEST(NumpyTests, TestIndexIsLarger) {
  // Simple case of fancy index broadcasting of the index.
  auto a = torch::zeros({5, 5});
  a(torch::tensor({{0}, {1}, {2}}), torch::tensor({0, 1, 2})) = torch::tensor({2., 3., 4.});

  ASSERT_TRUE((a({None, 3}, {None, 3}) == torch::tensor({2., 3., 4.})).all().item<bool>());
}

/*
    def test_broadcast_subspace(self):
        a = torch.zeros((100, 100))
        v = torch.arange(0., 100)[:, None]
        b = torch.arange(99, -1, -1).long()
        a[b] = v
        expected = b.double().unsqueeze(1).expand(100, 100)
        self.assertEqual(a, expected)
*/
TEST(NumpyTests, TestBroadcastSubspace) {
  auto a = torch::zeros({100, 100});
  auto v = torch::arange(0., 100)({}, None);
  auto b = torch::arange(99, -1, -1).to(torch::kLong);
  // yf225 TODO:
  // Some investigation:
  // -1. `a(b) = v` calls the scalar assignment function? Is it because IndexedTensor rhs can only map to a Scalar type? How does it even make sense?? yf225 TODO: how do we ban this conversion? wait it doesn't seem to happen anymore???
  // 0. `auto v = torch::arange(0., 100).view({100, 1})` works!!!!
  // 1. Even if we comment out the `operator=(Scalar v)` overload, the following still won't take the `operator=(Tensor const & rhs)` overload
  // 2. If we comment out the `operator=(Scalar v)` overload, it thorws `error: use of deleted function ‘at::IndexedTensor& at::IndexedTensor::operator=(const at::IndexedTensor&)’`
  // 3. If we comment out the `operator=(Scalar v)` overload, and change `auto v = torch::arange(0., 100)({}, None);`'s type to Tensor, `a(b) = v` will call the `operator=(Tensor const & rhs) &&` overload!
  // 4. Tried to add a Tensor -> IndexedTensor conversion operator `operator Tensor() const` to IndexedTensor class, but it doesn't work!!!
  // [WORKS] 5. Tried to explicitly add `Tensor & operator=(IndexedTensor const & rhs) &&` and `Tensor & operator=(IndexedTensor && rhs) &&`, and IT WORKS!!!!!
  // [TODO TRY THIS] 5. Add std::unique_ptr<index_history> field to TensorImpl, and check it in the Tensor’s assignment operator. We need to benchmark its effect on the Tensor's assignment operator!
  // [DOESN'T WORK] 6. Don't think about banning `auto b = a(indices...)`, because even if we ban `auto b = a(indices...)`, `IndexedTensor` type will still be leaked into user space when people do `some_function(a(indices...), ...)`, so this point is moot.
  // yf225 TODO: in light of this problem, we need to add tests for basic operations on IndexedTensor, like `add_` etc. to make sure normal tensor operations on IndexedTensor actually works
  a(b) = v;
  auto expected = b.to(torch::kDouble).unsqueeze(1).expand({100, 100});
  assert_equal(a, expected);
}

// yf225 TODO: we need test for the following:L
// 1. assigning IndexedTensor to Tensor (e.g. `a[0] = b({}, None)`)
// 2. assigning IndexedTensor to IndexedTensor (e.g. `a({}, None) = b({}, None)`)
// 3. assigning Tensor to IndexedTensor (e.g. `b({}, None) = a[0]`)
