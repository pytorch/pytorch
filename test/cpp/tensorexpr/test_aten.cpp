#include "test/cpp/tensorexpr/test_base.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "test/cpp/tensorexpr/padded_buffer.h"

namespace torch {
namespace jit {

using namespace torch::jit::tensorexpr;

void testATen_cast_Float() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle to_float = Cast::make(kFloat32, load_a);
  Stmt* store_b = Store::make(b_buf, index, to_float, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), static_cast<float>(i)) << "index: " << i;
  }
}

void testATennegInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle to_float = Sub::make(0, load_a);
  Stmt* store_b = Store::make(b_buf, index, to_float, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), -static_cast<float>(i)) << "index: " << i;
  }
}

void testATennegFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle to_float = Sub::make(0, load_a);
  Stmt* store_b = Store::make(b_buf, index, to_float, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), -i) << "index: " << i;
  }
}

void testATenaddInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  Stmt* store_d = Store::make(d_buf, index, load_a + load_b * load_c, 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i)) << "index: " << i;
  }
}

void testATenaddFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  Stmt* store_d = Store::make(d_buf, index, load_a + load_b * load_c, 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i) + b_v(i) * c_v(i)) << "index: " << i;
  }
}

void testATensubInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  Stmt* store_d = Store::make(d_buf, index, load_a - load_b * load_c, 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i)) << "index: " << i;
  }
}

void testATensubFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  Stmt* store_d = Store::make(d_buf, index, load_a - load_b * load_c, 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i) - b_v(i) * c_v(i)) << "index: " << i;
  }
}

void testATenlerp() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  Stmt* store_d =
      Store::make(d_buf, index, load_a + load_c * (load_b - load_a), 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf);
  ir_eval(a_v, b_v, c_v, d_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), a_v(i) + c_v(i) * (b_v(i) - a_v(i))) << "index: " << i;
  }
}

void testATenaddcmulInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer e_buf(VarHandle("E", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  ExprHandle load_d = Load::make(d_buf, index, 1);
  Stmt* store_e =
      Store::make(e_buf, index, load_a + load_b * load_c * load_d, 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf, e_buf);
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), 5 * i + 3) << "index: " << i;
    EXPECT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i)) << "index: " << i;
  }
}

void testATenaddcmulFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer d_buf(VarHandle("D", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer e_buf(VarHandle("E", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  ExprHandle load_c = Load::make(c_buf, index, 1);
  ExprHandle load_d = Load::make(d_buf, index, 1);
  Stmt* store_e =
      Store::make(e_buf, index, load_a + load_b * load_c * load_d, 1);
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

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf, d_buf, e_buf);
  ir_eval(a_v, b_v, c_v, d_v, e_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), 3 * i + 2) << "index: " << i;
    EXPECT_EQ(d_v(i), 5 * i + 3) << "index: " << i;
    EXPECT_EQ(e_v(i), a_v(i) + b_v(i) * c_v(i) * d_v(i)) << "index: " << i;
  }
}

void testATenmulInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, load_a * load_b, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * b_v(i)) << "index: " << i;
  }
}

void testATenmulFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, load_a * load_b, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * b_v(i)) << "index: " << i;
  }
}

void testATendivInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, load_a / load_b, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(b_v(i), i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) / b_v(i)) << "index: " << i;
  }
}

void testATendivFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, load_a / load_b, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = 2 * i + 1;
    b_v(i) = i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(b_v(i), i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) / b_v(i)) << "index: " << i;
  }
}

void testATenmaxInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, Max::make(load_a, load_b, true), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::max(a_v(i), b_v(i))) << "index: " << i;
  }
}

void testATenmaxFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, Max::make(load_a, load_b, true), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::fmax(a_v(i), b_v(i))) << "index: " << i;
  }
}

void testATenminInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, Min::make(load_a, load_b, true), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);
  PaddedBuffer<int> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::min(a_v(i), b_v(i))) << "index: " << i;
  }
}

void testATenminFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(c_buf, index, Min::make(load_a, load_b, true), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), std::fmin(a_v(i), b_v(i))) << "index: " << i;
  }
}

void testATen_sigmoid_backward() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(
      c_buf, index, load_a * load_b * (FloatImm::make(1.0f) - load_b), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * b_v(i) * (1.0f - b_v(i))) << "index: " << i;
  }
}

void testATen_tanh_backward() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer c_buf(VarHandle("C", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  ExprHandle load_b = Load::make(b_buf, index, 1);
  Stmt* store_c = Store::make(
      c_buf, index, load_a * (FloatImm::make(1.0f) - (load_b * load_b)), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_c);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);
  PaddedBuffer<float> c_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
    b_v(i) = 2 * i + 1;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf, c_buf);
  ir_eval(a_v, b_v, c_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 2 * i + 1) << "index: " << i;
    EXPECT_EQ(c_v(i), a_v(i) * (1.0f - (b_v(i) * b_v(i)))) << "index: " << i;
  }
}

void testATenreciprocal() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, FloatImm::make(1.0f) / load_a, 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i) << "index: " << i;
    EXPECT_EQ(b_v(i), 1.0f / i) << "index: " << i;
  }
}

void testATenreluInt() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kInt32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kInt32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, Max::make(load_a, 0, false), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<int> a_v(kTotalSize);
  PaddedBuffer<int> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i - 64) << "index: " << i;
    EXPECT_EQ(b_v(i), std::max(a_v(i), 0)) << "index: " << i;
  }
}

void testATenreluFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(
      b_buf,
      index,
      Max::make(load_a, 0, false), // relu does not propagate nans
      1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i - 64;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i - 64) << "index: " << i;
    EXPECT_EQ(b_v(i), std::fmax(a_v(i), 0)) << "index: " << i;
  }
}

void testATenlogFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, log(load_a), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i + 10) << "index: " << i;
    EXPECT_EQ(b_v(i), std::log(a_v(i))) << "index: " << i;
  }
}

void testATenlog10Float() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, log10(load_a), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i + 10) << "index: " << i;
    EXPECT_EQ(b_v(i), std::log10(a_v(i))) << "index: " << i;
  }
}

void testATenlog2Float() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, log2(load_a), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i + 10;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i + 10) << "index: " << i;
    EXPECT_EQ(b_v(i), std::log2(a_v(i))) << "index: " << i;
  }
}

void testATenexpFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, exp(load_a), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i / 10.0f) << "index: " << i;
    EXPECT_EQ(b_v(i), std::exp(a_v(i))) << "index: " << i;
  }
}

void testATenerfFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, erf(load_a), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i / 10.0f) << "index: " << i;
    EXPECT_EQ(b_v(i), std::erf(a_v(i))) << "index: " << i;
  }
}

void testATencosFloat() {
  KernelScope kernel_scope;
  const int kTotalSize = 128;
  Buffer a_buf(VarHandle("A", kHandle), kFloat32, {ExprHandle(kTotalSize)});
  Buffer b_buf(VarHandle("B", kHandle), kFloat32, {ExprHandle(kTotalSize)});

  VarHandle index = VarHandle("index", kInt32);
  ExprHandle load_a = Load::make(a_buf, index, 1);
  Stmt* store_b = Store::make(b_buf, index, cos(load_a), 1);
  Stmt* stmt = For::make(index, 0, kTotalSize, store_b);

  PaddedBuffer<float> a_v(kTotalSize);
  PaddedBuffer<float> b_v(kTotalSize);

  for (int i = 0; i < kTotalSize; ++i) {
    a_v(i) = i / 10.0f;
  }

  SimpleIREvaluator ir_eval(stmt, a_buf, b_buf);
  ir_eval(a_v, b_v);

  for (int i = 0; i < kTotalSize; ++i) {
    EXPECT_EQ(a_v(i), i / 10.0f) << "index: " << i;
    EXPECT_EQ(b_v(i), std::cos(a_v(i))) << "index: " << i;
  }
}

void testATeneqInt() {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 1);
  std::vector<int> b_buffer(N, 1);
  std::vector<int> c_buffer(N, 0);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kEQ),
          mask));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

void testATengeInt() {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kGE),
          mask));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 0);
}

void testATengtInt() {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 6);
  std::vector<int> b_buffer(N, 3);
  std::vector<int> c_buffer(N, 0);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kGT),
          mask));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

void testATenleInt() {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 0);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kLE),
          mask));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 1);
}

void testATenltInt() {
  KernelScope kernel_scope;
  constexpr int N = 128;
  Buffer a(VarHandle("A", kHandle), kInt32, {N});
  Buffer b(VarHandle("B", kHandle), kInt32, {N});
  Buffer c(VarHandle("C", kHandle), kInt32, {N});
  std::vector<int> a_buffer(N, 5);
  std::vector<int> b_buffer(N, 5);
  std::vector<int> c_buffer(N, 1);

  auto mask = IntImm::make(1);
  VarHandle i("i", kInt32);
  auto memcpy_expr = For::make(
      i,
      0,
      N,
      Store::make(
          c,
          i,
          CompareSelect::make(
              Load::make(a, i, mask),
              Load::make(b, i, mask),
              CompareSelectOperation::kLT),
          mask));

  SimpleIREvaluator ir_eval(memcpy_expr, a, b, c);
  ir_eval(a_buffer, b_buffer, c_buffer);

  assertAllEqual(c_buffer, 0);
}

} // namespace jit
} // namespace torch
