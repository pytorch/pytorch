
#include "ATen/ATen.h"

// for TH compat test only...
struct THFloatTensor;
extern "C" THFloatTensor * THFloatTensor_newWithSize2d(size_t a, size_t b);
extern "C" void THFloatTensor_fill(THFloatTensor *, float v);

#include <iostream>
#include <chrono>

using namespace at;

void check(bool c) {
  if(!c)
    throw std::runtime_error("check failed.");
}

static void test(Type & type) {
  {
    std::cout << "resize:" << std::endl;
    auto a = type.tensor();
    a.resize_({3,4});
    std::cout << a.numel() << std::endl;
    a.resize_({5, 7});
    std::cout << a.numel() << std::endl;
  }

  {
    std::cout << "ones and dot:" << std::endl;
    Tensor b = type.ones({3, 4});
    std::cout << b << std::endl;
    std::cout << b.numel() << std::endl;
    std::cout << b.dot(b) << std::endl;
  }

  {
    std::cout << "rand:" << std::endl;
    for(auto i = 0; i < 10; i++) {
      Tensor a = type.toScalarType(i % 2 == 0 ? kFloat : kDouble).rand({3,4});
      std::cout << a << std::endl;
    }
  }

  {
    std::cout << "sort:" << std::endl;
    Tensor b = type.rand({3, 4});

    std::cout << b << std::endl;
    auto z = b.sort(1);
    std::cout << std::get<0>(z) << std::endl;
    std::cout << std::get<1>(z) << std::endl;
  }
  if(type.backend() != kCUDA)
  {
    std::cout << "randperm:" << std::endl;
    Tensor b = type.randperm(15);
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
    Tensor a = type.rand({3, 4});
    Tensor b = type.rand({3, 4});
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
    Tensor d = type.ones({3, 4});
    Tensor r = type.zeros({3,4});
    for(auto i = 0; i < 100000; i++) {
      add_out(r, d, r);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    std::cout << "   norm: " << norm(r).toDouble() << std::endl;
  }

  {
    std::cout << "loads of adds (with copy):" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = type.ones({3, 4});
    Tensor r = type.zeros({3, 4});
    for(auto i = 0; i < 100000; i++) {
      r = add(r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    std::cout << "   norm: " << norm(r).toDouble() << std::endl;
  }

  {
    std::cout << "isContiguous:" << std::endl;
    Tensor a = type.rand({3, 4});
    std::cout << a.is_contiguous() << std::endl;
  }

  {
    std::cout << "mm:" << std::endl;
    Tensor a = type.rand({3, 4});
    Tensor b = type.rand({4});
    Tensor c = mv(a, b);
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
  }

  {
    std::cout << "squeeze:" << std::endl;
    Tensor a = type.rand({2, 1});
    std::cout << a << std::endl;
    Tensor b = squeeze(a);
    std::cout << b << std::endl;
    a = type.rand({1});
    std::cout << a << std::endl;
    b = squeeze(a);
    std::cout << b << std::endl;
  }

  {
    std::cout << "copy:" << std::endl;
    Tensor a = type.zeros({4, 3});
    std::cout << a << std::endl;
    Tensor e = type.rand({3, 4});
    std::cout << e << std::endl;
    a.copy_(e);
    std::cout << a << std::endl;
  }

  {
    //TODO(zach): 0-dim
    //std::cout << "abs(value):" << std::endl;
    //std::cout << at::abs(-3);
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
    std::cout << "adding a value with different type:" << std::endl;
    Tensor a = type.rand({4, 3});
    std::cout << a << std::endl;
    std::cout << add(a, 1) << std::endl;
  }

  {
    std::cout << "select:" << std::endl;
    Tensor a = type.rand({3, 7});
    std::cout << a << std::endl;
    std::cout << select(a, 1, 3) << std::endl;
    std::cout << select(select(a, 1, 3), 0, 2) << std::endl;
  }

  {
      std::cout << "zero-dim: " << std::endl;
      Tensor a =  type.scalarTensor(4); //type.rand({1});

      std::cout << a << "dims: " << a.dim() << std::endl;
      std::cout << Scalar(a) << std::endl;
      Tensor b = type.rand({3,4});
      std::cout << b + a << std::endl;
      std::cout << a + b << std::endl;
      check((a+a).dim() == 0);
      check((1+a).dim() == 0);
      auto c = type.rand({3,4});
      std::cout << c[1][2] << std::endl;

      auto f = type.rand({3,4});
      f[2] = type.zeros({4});
      f[1][0] = -1;
      std:: cout << f << std::endl;
  }
  {
    int a = 4;
    THFloatTensor *t = THFloatTensor_newWithSize2d(a, a);
    THFloatTensor_fill(t, a);
    Tensor tt = CPU(kFloat).unsafeTensorFromTH(t,false);
    std::cout << tt << std::endl;
  }
  {
      Tensor a = CPU(kFloat).zeros({3,4});
      Tensor b = CPU(kFloat).ones({3,7});
      Tensor c = cat({a,b},1);
      std::cout << c << std::endl;

      Tensor e = CPU(kFloat).rand({});
      check(*e.data<float>()== e.sum().toFloat());
  }

}

int main()
{
  std::cout << "=========================== CPU ===========================" << std::endl;
  test(CPU(kFloat));
  if(at::hasCUDA()) {
    std::cout << "=========================== GPU ===========================" << std::endl;
    test(CUDA(kFloat));
  }
  return 0;
}
