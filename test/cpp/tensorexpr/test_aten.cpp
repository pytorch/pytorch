#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include <c10/macros/Macros.h>
#include <c10/util/irange.h>
#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

TEST(ATen, _cast_Float) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Cast::make(kFloat, load_a);
  StmtPtr store_b = b_buf.store({index}, to_float);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), static_cast<float>(i));
  }
}

TEST(ATen, negInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Sub::make(0, load_a);
  StmtPtr store_b = b_buf.store({index}, to_float);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -static_cast<float>(i));
  }
}

TEST(ATen, negFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Sub::make(0, load_a);
  StmtPtr store_b = b_buf.store({index}, to_float);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -i);
  }
}

TEST(ATen, addInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  StmtPtr store_d = d_buf.store({index}, load_a + load_b * load_c);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

TEST(ATen, addFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  StmtPtr store_d = d_buf.store({index}, load_a + load_b * load_c);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

TEST(ATen, subInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  StmtPtr store_d = d_buf.store({index}, load_a - load_b * load_c);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

TEST(ATen, subFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  StmtPtr store_d = d_buf.store({index}, load_a - load_b * load_c);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

TEST(ATen, lerp) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  StmtPtr store_d = d_buf.store({index}, load_a + load_c * (load_b - load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + c_v(i) * (b_v(i) - a_v(i)));
  }
}

TEST(ATen, addcmulInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kInt);
  BufHandle e_buf("E", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  ExprHandle load_d = d_buf.load(index);
  StmtPtr store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);
  PaddedBuffer<int> e_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf, e_buf});
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}

TEST(ATen, addcmulFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle d_buf("D", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle e_buf("E", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  ExprHandle load_d = d_buf.load(index);
  StmtPtr store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);
  PaddedBuffer<float> e_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf, e_buf});
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_FLOAT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}

TEST(ATen, mulInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, load_a * load_b);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}

TEST(ATen, mulFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, load_a * load_b);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}

TEST(ATen, divInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, load_a / load_b);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

TEST(ATen, divFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, load_a / load_b);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

TEST(ATen, maxInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::max(a_v(i), b_v(i)));
  }
}

TEST(ATen, maxFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmax(a_v(i), b_v(i)));
  }
}

TEST(ATen, minInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::min(a_v(i), b_v(i)));
  }
}

TEST(ATen, minFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle c_buf("C", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  StmtPtr store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmin(a_v(i), b_v(i)));
  }
}

void __ubsan_ignore_float_divide_by_zero__ testATenreciprocal() {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, FloatImm::make(1.0f) / load_a);
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 1.0f / i);
  }
}

TEST(ATen, reluInt) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kInt);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kInt);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, Max::make(load_a, 0, false));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::max(a_v(i), 0));
  }
}

TEST(ATen, reluFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store(
      {index}, Max::make(load_a, 0, false) // relu does not propagate nans
  );
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::fmax(a_v(i), 0));
  }
}

TEST(ATen, logFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, log(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log(a_v(i)));
  }
}

TEST(ATen, fastLogFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, fast_log(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    auto ref = std::log(a_v(i));
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      ASSERT_FLOAT_EQ(test, ref);
    }
  }
}

TEST(ATen, fastTanhFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, fast_tanh(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    auto ref = std::tanh(a_v(i));
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      ASSERT_NEAR(test, ref, 1e-6);
    }
  }
}

TEST(ATen, fastSigmoidFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, fast_sigmoid(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    auto test = b_v(i);
    at::Tensor t = at::ones({1}) * a_v(i);
    float ref = at::sigmoid(t).item().to<float>();
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      ASSERT_NEAR(test, ref, 1e-6);
    }
  }
}

TEST(ATen, log10Float) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, log10(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log10(a_v(i)));
  }
}

TEST(ATen, log2Float) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, log2(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log2(a_v(i)));
  }
}

TEST(ATen, expFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, exp(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::exp(a_v(i)));
  }
}

TEST(ATen, erfFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, erf(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::erf(a_v(i)));
  }
}

TEST(ATen, cosFloat) {
  const int kTotalSize = 128;
  BufHandle a_buf("A", {ExprHandle(kTotalSize)}, kFloat);
  BufHandle b_buf("B", {ExprHandle(kTotalSize)}, kFloat);

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  StmtPtr store_b = b_buf.store({index}, cos(load_a));
  StmtPtr stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (const auto i : c10::irange(kTotalSize)) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (const auto i : c10::irange(kTotalSize)) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::cos(a_v(i)));
  }
}

TEST(ATen, eqInt) {
  constexpr int N = 128;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kEQ)));

  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, geInt) {
  constexpr int N = 128;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGE)));

  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, gtInt) {
  constexpr int N = 128;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  std::vector<int> a_buffer(N, 6);
  std::vector<int> b_buffer(N, 3);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kGT)));

  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, leInt) {
  constexpr int N = 128;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 0);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLE)));

  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

TEST(ATen, ltInt) {
  constexpr int N = 128;
  BufHandle a("A", {N}, kInt);
  BufHandle b("B", {N}, kInt);
  BufHandle c("C", {N}, kInt);
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 1);

  VarHandle i("i", kInt);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      c.store(
          {i},
          CompareSelect::make(
              a.load(i), b.load(i), CompareSelectOperation::kLT)));

  SimpleIREvaluator ir_eval(memcpy_expr, {a, b, c});
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 0);
}

} // namespace jit
} // namespace torch
