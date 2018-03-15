
#include "ATen/ATen.h"

// for TH compat test only...
struct THFloatTensor;
extern "C" THFloatTensor * THFloatTensor_newWithSize2d(size_t a, size_t b);
extern "C" void THFloatTensor_fill(THFloatTensor *, float v);

#include <iostream>
#include <chrono>
#include <string.h>
#include <sstream>
#include "test_assert.h"
#include "test_seed.h"

using namespace at;



static void test(Type & type) {
  {
    std::cout << "resize:" << std::endl;
    auto a = type.tensor();
    a.resize_({3,4});
    std::cout << a.numel() << std::endl;
    ASSERT(a.numel() == 12);
    a.resize_({5, 7});
    std::cout << a.numel() << std::endl;
    ASSERT(a.numel() == 35);

  }

  {
    std::cout << "ones and dot:" << std::endl;
    Tensor b0 = ones(type, {1, 1});
    std::cout << b0 << std::endl;
    ASSERT(2 == (b0+b0).sum().toCDouble());

    Tensor b1 = ones(type, {1, 2});
    std::cout << b1 << std::endl;
    ASSERT(4 == (b1+b1).sum().toCDouble());

    Tensor b = ones(type, {3, 4});
    std::cout << b << std::endl;
    ASSERT(24 == (b+b).sum().toCDouble());
    std::cout << b.numel() << std::endl;
    ASSERT(12 == b.numel());
    std::cout << b.view(-1).dot(b.view(-1)) << std::endl;
    ASSERT(b.view(-1).dot(b.view(-1)).toCDouble() == 12);
  }

  {
    std::cout << "rand:" << std::endl;
    for(auto i = 0; i < 10; i++) {
      Tensor a = rand(type.toScalarType(i % 2 == 0 ? kFloat : kDouble), {3,4});
      std::cout << a << std::endl;
    }
  }

  {
    std::cout << "sort:" << std::endl;
    Tensor b = rand(type, {3, 4});

    std::cout << b << std::endl;
    auto z = b.sort(1);
    std::cout << std::get<0>(z) << std::endl;
    std::cout << std::get<1>(z) << std::endl;
  }
  if(type.backend() != kCUDA)
  {
    std::cout << "randperm:" << std::endl;
    Tensor b = randperm(type, 15);
    std::cout << b << std::endl;
    Tensor rv, ri;
    std::tie(rv, ri) = sort(b, 0);
    std::cout << rv << std::endl;
    std::cout << ri << std::endl;
  }

  {
    std::cout << "context: " << std::hex << (int64_t)&globalContext() << std::endl;
  }

  {
    std::cout << "add:" << std::endl;
    Tensor a = rand(type, {3, 4});
    Tensor b = rand(type, {3, 4});
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    Tensor c = add(a, add(a, b));
    std::cout << c << std::endl;
    //TODO:0-dim Tensor d(3.f);
    Scalar d = 3.f;
    std::cout << d << std::endl;
    std::cout << add(c, d) << std::endl;
  }


  {
    std::cout << "loads of adds:" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones(type, {3, 4});
    Tensor r = zeros(type, {3, 4});
    for(auto i = 0; i < 100000; i++) {
      add_out(r, r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    ASSERT(norm(100000*d).toCDouble() == norm(r).toCDouble());
    std::cout << "   norm: " << norm(r).toCDouble() << std::endl;
  }

  {
    std::cout << "loads of adds (with copy):" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones(type, {3, 4});
    Tensor r = zeros(type, {3, 4});
    for(auto i = 0; i < 100000; i++) {
      r = add(r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    ASSERT(norm(100000*d).toCDouble() == norm(r).toCDouble());
    std::cout << "   norm: " << norm(r).toCDouble() << std::endl;
  }

  {
    std::cout << "isContiguous:" << std::endl;
    Tensor a = rand(type, {3, 4});
    std::cout << a.is_contiguous() << std::endl;
    ASSERT(a.is_contiguous());
    a = a.transpose(0, 1);
    ASSERT(!a.is_contiguous());
  }

  {
    Tensor a = rand(type, {3, 4, 5});
    Tensor b = a.permute({1, 2, 0});
    ASSERT(b.sizes().equals({4, 5, 3}));
    ASSERT(b.strides().equals({5, 1, 20}));
  }

  {
    std::cout << "mm:" << std::endl;
    Tensor a = rand(type, {3, 4});
    Tensor b = rand(type, {4});
    Tensor c = mv(a, b);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    ASSERT(c.equal(addmv(zeros(type, {3}), a, b, 0, 1)));
  }

  {
    std::cout << "squeeze:" << std::endl;
    Tensor a = rand(type, {2, 1});
    std::cout << a << std::endl;
    Tensor b = squeeze(a);
    ASSERT(b.dim() == 1);
    std::cout << b << std::endl;
    a = rand(type, {1});
    std::cout << a << std::endl;
    b = squeeze(a);
    //TODO 0-dim squeeze
    std::cout << b << std::endl;
  }

  {
    std::cout << "copy:" << std::endl;
    Tensor a = zeros(type, {4, 3});
    std::cout << a << std::endl;
    Tensor e = rand(type, {4, 3});
    std::cout << e << std::endl;
    a.copy_(e);
    std::cout << a << std::endl;
    ASSERT(a.equal(e));
  }

  {
    std::cout << "copy [broadcasting]:" << std::endl;
    Tensor a = zeros(type, {4, 3});
    Tensor e = rand(type, {3});
    a.copy_(e);
    for (int i = 0; i < 4; ++i) {
      ASSERT(a[i].equal(e));
    }
  }

  {
    std::cout << "abs(value):" << std::endl;
    Tensor r = at::abs(type.scalarTensor(-3));
    std::cout << r;
    ASSERT(Scalar(r).toInt() == 3);
  }

//TODO(zach): operator overloads
#if 0
  {
    std::cout << "eq (value):" << std::endl;
    Tensor a = Tensor(10.f);
    std::cout << (a == 11_i64) << " -- should be 0" << std::endl;
    std::cout << (a == 10_i64) << " -- should be 1" << std::endl;
    std::cout << (a == 10.) << " -- should be 1" << std::endl;
  }
#endif

  {
    std::cout << "adding a value with a salar:" << std::endl;
    Tensor a = rand(type, {4, 3});
    std::cout << a << std::endl;
    std::cout << add(a, 1) << std::endl;
    ASSERT((ones(type, {4,3}) + a).equal(add(a,1)));
  }

  {
    std::cout << "select:" << std::endl;
    Tensor a = rand(type, {3, 7});
    std::cout << a << std::endl;
    std::cout << select(a, 1, 3) << std::endl;
    std::cout << select(select(a, 1, 3), 0, 2) << std::endl;
  }

  {
      std::cout << "zero-dim: " << std::endl;
      Tensor a =  type.scalarTensor(4); //rand(type, {1});

      std::cout << a << "dims: " << a.dim() << std::endl;
      std::cout << Scalar(a) << std::endl;
      Tensor b = rand(type, {3,4});
      std::cout << b + a << std::endl;
      std::cout << a + b << std::endl;
      ASSERT((a+a).dim() == 0);
      ASSERT((1+a).dim() == 0);
      auto c = rand(type, {3,4});
      std::cout << c[1][2] << std::endl;

      auto f = rand(type, {3,4});
      f[2] = zeros(type, {4});
      f[1][0] = -1;
      std:: cout << f << std::endl;
      ASSERT(Scalar(f[2][0]).toDouble() == 0);
  }
  {
    int a = 4;
    THFloatTensor *t = THFloatTensor_newWithSize2d(a, a);
    THFloatTensor_fill(t, a);
    Tensor tt = CPU(kFloat).unsafeTensorFromTH(t,false);
    std::cout << tt << std::endl;
  }
  {
      Tensor a = zeros(CPU(kFloat), {3,4});
      Tensor b = ones(CPU(kFloat), {3,7});
      Tensor c = cat({a,b},1);
      std::cout << c.sizes() << std::endl;
      ASSERT(c.size(1) == 11);
      std::cout << c << std::endl;

      Tensor e = rand(CPU(kFloat), {});
      ASSERT(*e.data<float>()== e.sum().toCFloat());
  }
  {
    Tensor b = ones(CPU(kFloat), {3,7})*.0000001f;
    std::stringstream s;
    s << b << "\n";
    std::string expect = "1e-07 *";
    ASSERT(s.str().substr(0,expect.size()) == expect);
  }
  {
    // Indexing by Scalar
    Tensor tensor = CPU(kInt).arange(0, 10);
    Tensor one = CPU(kInt).ones({1});
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      ASSERT(tensor[i].equal(one * i));
    }
    for (int i = 0; i < tensor.numel(); ++i) {
      ASSERT(tensor[i].equal(one * i));
    }
    for (int16_t i = 0; i < tensor.numel(); ++i) {
      ASSERT(tensor[i].equal(one * i));
    }
    for (int8_t i = 0; i < tensor.numel(); ++i) {
      ASSERT(tensor[i].equal(one * i));
    }
    try {
      ASSERT(tensor[Scalar(3.14)].equal(one));
    } catch (const std::runtime_error& error) {
      ASSERT(
          std::string(error.what()) ==
          "Can only index tensors with integral scalars (got CPUDoubleType)");
    }
  }
  {
    // Indexing by zero-dim tensor
    Tensor tensor = CPU(kInt).arange(0, 10);
    Tensor one = CPU(kInt).ones({});
    for (int i = 0; i < tensor.numel(); ++i) {
      ASSERT(tensor[one * i].equal(one * i));
    }
    try {
      ASSERT(tensor[CPU(kFloat).ones({}) * 3.14].equal(one));
    } catch (const std::runtime_error& error) {
      ASSERT(
          std::string(error.what()) ==
          "Can only index tensors with integral scalars (got CPUFloatType)");
    }
    try {
      ASSERT(tensor[Tensor()].equal(one));
    } catch (const std::runtime_error& error) {
      ASSERT(
          std::string(error.what()) ==
          "Can only index with tensors that are defined");
    }
    try {
      ASSERT(tensor[CPU(kInt).ones({2, 3, 4})].equal(one));
    } catch (const std::runtime_error& error) {
      ASSERT(
        std::string(error.what()) ==
        "Can only index with tensors that are scalars (zero-dim)");
    }
  }
}

int main(int argc, char ** argv)
{
  manual_seed(123);

  std::cout << "=========================== CPU ===========================" << std::endl;
  test(CPU(kFloat));
  if(at::hasCUDA()) {
    if(argc == 2 && 0 == strcmp(argv[1],"-n")) {
      std::cout << "skipping cuda...\n";
    } else {
      std::cout << "=========================== GPU ===========================" << std::endl;
      test(CUDA(kFloat));
    }
  }
  return 0;
}
