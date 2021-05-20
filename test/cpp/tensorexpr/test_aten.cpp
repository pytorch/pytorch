#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include <c10/macros/Macros.h>
#include "test/cpp/tensorexpr/padded_buffer.h"
#include "test/cpp/tensorexpr/test_base.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, _cast_Float) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Cast::make(kFloat, load_a);
  Stmt* store_b = b_buf.store({index}, to_float);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), static_cast<float>(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, negInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Sub::make(0, load_a);
  Stmt* store_b = b_buf.store({index}, to_float);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -static_cast<float>(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, negFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle to_float = Sub::make(0, load_a);
  Stmt* store_b = b_buf.store({index}, to_float);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), -i);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, addInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a + load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, addFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a + load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, subInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a - load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, subFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a - load_b * load_c);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, lerp) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  Stmt* store_d = d_buf.store({index}, load_a + load_c * (load_b - load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_d);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf});
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), a_v(i) + c_v(i) * (b_v(i) - a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, addcmulInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kInt));
  Placeholder e_buf(BufHandle("E", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  ExprHandle load_d = d_buf.load(index);
  Stmt* store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);
  PaddedBuffer<int> d_v(kTotalSize);
  PaddedBuffer<int> e_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf, e_buf});
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, addcmulFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder d_buf(BufHandle("D", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder e_buf(BufHandle("E", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  ExprHandle load_c = c_buf.load(index);
  ExprHandle load_d = d_buf.load(index);
  Stmt* store_e = e_buf.store({index}, load_a + load_b * load_c * load_d);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_e);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);
  PaddedBuffer<float> d_v(kTotalSize);
  PaddedBuffer<float> e_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
    c_v(i) = 3 * i + 2;
    d_v(i) = 5 * i + 3;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf, d_buf, e_buf});
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), 3 * i + 2);
    ASSERT_EQ(d_v(i), 5 * i + 3);
    ASSERT_FLOAT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, mulInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a * load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, mulFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a * load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), a_v(i) * b_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, divInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a / load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, divFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, load_a / load_b);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), 2 * i + 1);
    ASSERT_EQ(b_v(i), i + 1);
    ASSERT_EQ(c_v(i), a_v(i) / b_v(i));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, maxInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::max(a_v(i), b_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, maxFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Max::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmax(a_v(i), b_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, minInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::min(a_v(i), b_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, minFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder c_buf(BufHandle("C", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  ExprHandle load_b = b_buf.load(index);
  Stmt* store_c = c_buf.store({index}, Min::make(load_a, load_b, true));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf, c_buf});
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 2 * i + 1);
    ASSERT_EQ(c_v(i), std::fmin(a_v(i), b_v(i)));
  }
}

void __ubsan_ignore_float_divide_by_zero__ testATenreciprocal() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, FloatImm::make(1.0f) / load_a);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i);
    ASSERT_EQ(b_v(i), 1.0f / i);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, reluInt) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kInt));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kInt));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, Max::make(load_a, 0, false));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::max(a_v(i), 0));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, reluFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store(
      {index}, Max::make(load_a, 0, false) // relu does not propagate nans
  );
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i - 64);
    ASSERT_EQ(b_v(i), std::fmax(a_v(i), 0));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, logFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, log(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log(a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, fastLogFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, fast_log(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    auto test = b_v(i);
    auto ref = std::log(a_v(i));
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      ASSERT_FLOAT_EQ(test, ref);
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, fastTanhFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, fast_tanh(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    auto test = b_v(i);
    auto ref = std::tanh(a_v(i));
    if (std::isnan(ref)) {
      ASSERT_EQ(std::isnan(test), true);
    } else {
      ASSERT_NEAR(test, ref, 1e-6);
    }
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, fastSigmoidFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, fast_sigmoid(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = at::randn({1}).item().to<float>();
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, log10Float) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, log10(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log10(a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, log2Float) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, log2(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i + 10);
    ASSERT_EQ(b_v(i), std::log2(a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, expFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, exp(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::exp(a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, erfFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, erf(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::erf(a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, cosFloat) {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Placeholder a_buf(BufHandle("A", {ExprHandle(kTotalSize)}, kFloat));
  Placeholder b_buf(BufHandle("B", {ExprHandle(kTotalSize)}, kFloat));

  VarHandle index = VarHandle("index", kInt);
  ExprHandle load_a = a_buf.load(index);
  Stmt* store_b = b_buf.store({index}, cos(load_a));
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, {a_buf, b_buf});
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    ASSERT_EQ(a_v(i), i / 10.0f);
    ASSERT_EQ(b_v(i), std::cos(a_v(i)));
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, eqInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, geInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, gtInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, leInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ATen, ltInt) {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Placeholder a(BufHandle("A", {N}, kInt));
  Placeholder b(BufHandle("B", {N}, kInt));
  Placeholder c(BufHandle("C", {N}, kInt));
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
