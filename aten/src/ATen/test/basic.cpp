#define CATCH_CONFIG_MAIN
#include "catch_utils.hpp"

#include "ATen/ATen.h"
#include "ATen/core/Reduction.h"

// for TH compat test only...
struct THFloatTensor;
extern "C" THFloatTensor * THFloatTensor_newWithSize2d(size_t a, size_t b);
extern "C" void THFloatTensor_fill(THFloatTensor *, float v);

#include <iostream>
#include <chrono>
#include <string.h>
#include <sstream>
#include "test_seed.h"

using namespace at;

using Catch::Matchers::StartsWith;

static void test(Type & type) {
  CATCH_SECTION( "resize" ) {
    auto a = type.tensor();
    a.resize_({3,4});
    CATCH_REQUIRE(a.numel() == 12);
    a.resize_({5, 7});
    CATCH_REQUIRE(a.numel() == 35);

  }

  CATCH_SECTION( "ones and dot" ) {
    Tensor b0 = ones({1, 1}, type);
    CATCH_REQUIRE(2 == (b0+b0).sum().toCDouble());

    Tensor b1 = ones({1, 2}, type);
    CATCH_REQUIRE(4 == (b1+b1).sum().toCDouble());

    Tensor b = ones({3, 4}, type);
    CATCH_REQUIRE(24 == (b+b).sum().toCDouble());
    CATCH_REQUIRE(12 == b.numel());
    CATCH_REQUIRE(b.view(-1).dot(b.view(-1)).toCDouble() == 12);
  }

  CATCH_SECTION( "rand" ) {
    for(auto i = 0; i < 10; i++) {
      Tensor a = rand({3,4}, type.toScalarType(i % 2 == 0 ? kFloat : kDouble));
    }
  }

  CATCH_SECTION( "sort" ) {
    Tensor b = rand({3, 4}, type);

    auto z = b.sort(1);
    auto z_sorted = std::get<0>(z);

    CATCH_REQUIRE(z_sorted[0][0].toCFloat() < z_sorted[0][1].toCFloat());
  }

  if(type.backend() != Backend::CUDA)
  CATCH_SECTION( "randperm" ) {
    Tensor b = randperm(15, type);
    Tensor rv, ri;
    std::tie(rv, ri) = sort(b, 0);
    CATCH_REQUIRE(rv[0].toCFloat() <= rv[1].toCFloat());
  }

  CATCH_SECTION( "context" ) {
    std::stringstream ss;
    ss << "context: " << std::hex << (int64_t)&globalContext() << std::endl;
  }

  CATCH_SECTION( "add" ) {
    Tensor a = rand({3, 4}, type);
    Tensor b = rand({3, 4}, type);
    Tensor c = add(a, add(a, b));
    //TODO:0-dim Tensor d(3.f);
    Scalar d = 3.f;
    CATCH_REQUIRE( add(c, d).allclose(a + a + b + d) );
  }

  CATCH_SECTION( "loads of adds" ) {
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones({3, 4}, type);
    Tensor r = zeros({3, 4}, type);
    for(auto i = 0; i < 100000; i++) {
      add_out(r, r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    //TODO TEST PERF?
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    CATCH_REQUIRE(norm(100000*d).toCDouble() == norm(r).toCDouble());
  }

  CATCH_SECTION( "loads of adds (with copy)" ) {
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones({3, 4}, type);
    Tensor r = zeros({3, 4}, type);
    for(auto i = 0; i < 100000; i++) {
      r = add(r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    //TODO TEST PERF?
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    CATCH_REQUIRE(norm(100000*d).toCDouble() == norm(r).toCDouble());
  }

  CATCH_SECTION( "isContiguous" ) {
    Tensor a = rand({3, 4}, type);
    CATCH_REQUIRE(a.is_contiguous());
    a = a.transpose(0, 1);
    CATCH_REQUIRE(!a.is_contiguous());
  }

  CATCH_SECTION( "permute" ) {
    Tensor a = rand({3, 4, 5}, type);
    Tensor b = a.permute({1, 2, 0});
    CATCH_REQUIRE(b.sizes().equals({4, 5, 3}));
    CATCH_REQUIRE(b.strides().equals({5, 1, 20}));
  }

  CATCH_SECTION( "mm" ) {
    Tensor a = rand({3, 4}, type);
    Tensor b = rand({4}, type);
    Tensor c = mv(a, b);
    CATCH_REQUIRE(c.equal(addmv(zeros({3}, type), a, b, 0, 1)));
  }

  CATCH_SECTION( "squeeze" ) {
    Tensor a = rand({2, 1}, type);
    Tensor b = squeeze(a);
    CATCH_REQUIRE(b.dim() == 1);
    a = rand({1}, type);
    b = squeeze(a);
    //TODO 0-dim squeeze
    CATCH_REQUIRE(a[0].equal(b));
  }

  CATCH_SECTION( "copy" ) {
    Tensor a = zeros({4, 3}, type);
    Tensor e = rand({4, 3}, type);
    a.copy_(e);
    CATCH_REQUIRE(a.equal(e));
  }

  CATCH_SECTION( "copy (broadcasting)" ) {
    Tensor a = zeros({4, 3}, type);
    Tensor e = rand({3}, type);
    a.copy_(e);
    for (int i = 0; i < 4; ++i) {
      CATCH_REQUIRE(a[i].equal(e));
    }
  }

  CATCH_SECTION( "abs(value)" ) {
    Tensor r = at::abs(type.scalarTensor(-3));
    CATCH_REQUIRE(r.toCInt() == 3);
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

  CATCH_SECTION( "adding a value with a scalar" ) {
    Tensor a = rand({4, 3}, type);
    CATCH_REQUIRE((ones({4,3}, type) + a).equal(add(a,1)));
  }

  CATCH_SECTION( "select" ) {
    Tensor a = rand({3, 7}, type);
    auto a_13 = select(a, 1, 3);
    auto a_13_02 = select(select(a, 1, 3), 0, 2);
    CATCH_REQUIRE( a[0][3].equal(a_13[0]) );
    CATCH_REQUIRE( a[2][3].equal(a_13_02) );
  }

  CATCH_SECTION( "zero-dim" ) {
    Tensor a =  type.scalarTensor(4); //rand(type, {1});

    Tensor b = rand({3,4}, type);
    CATCH_REQUIRE((a + a).dim() == 0);
    CATCH_REQUIRE((1 + a).dim() == 0);
    CATCH_REQUIRE((b + a).dim() == 2);
    CATCH_REQUIRE((a + b).dim() == 2);
    auto c = rand({3,4}, type);
    CATCH_REQUIRE(c[1][2].dim() == 0);

    auto f = rand({3,4}, type);
    f[2] = zeros({4}, type);
    f[1][0] = -1;
    CATCH_REQUIRE(f[2][0].toCDouble() == 0);
  }

  CATCH_SECTION( "tensor from TH" ) {
    int a = 4;
    THFloatTensor *t = THFloatTensor_newWithSize2d(a, a);
    THFloatTensor_fill(t, a);
    Tensor tt = CPU(kFloat).unsafeTensorFromTH(t,false);
    CATCH_REQUIRE_NOTHROW(tt);
  }

  CATCH_SECTION( "toCFloat" ) {
    Tensor a = zeros({3,4});
    Tensor b = ones({3,7});
    Tensor c = cat({a,b},1);
    CATCH_REQUIRE(c.size(1) == 11);

    Tensor e = rand({});
    CATCH_REQUIRE(*e.data<float>() == e.sum().toCFloat());
  }

  CATCH_SECTION( "to string" ) {
    Tensor b = ones({3,7})*.0000001f;
    std::stringstream s;
    s << b << "\n";
    std::string expect = "1e-07 *";
    CATCH_REQUIRE(s.str().substr(0,expect.size()) == expect);
  }
  CATCH_SECTION("indexing by Scalar") {
    Tensor tensor = arange(0, 10, kInt);
    Tensor one = ones({}, kInt);
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      CATCH_REQUIRE(tensor[i].equal(one * i));
    }
    for (size_t i = 0; i < static_cast<uint64_t>(tensor.numel()); ++i) {
      CATCH_REQUIRE(tensor[i].equal(one * static_cast<int64_t>(i)));
    }
    for (int i = 0; i < tensor.numel(); ++i) {
      CATCH_REQUIRE(tensor[i].equal(one * i));
    }
    for (int16_t i = 0; i < tensor.numel(); ++i) {
      CATCH_REQUIRE(tensor[i].equal(one * i));
    }
    for (int8_t i = 0; i < tensor.numel(); ++i) {
      CATCH_REQUIRE(tensor[i].equal(one * i));
    }
    CATCH_REQUIRE_THROWS_WITH(
        tensor[Scalar(3.14)].equal(one),
        StartsWith(
            "Can only index tensors with integral scalars"));
  }
  CATCH_SECTION("indexing by zero-dim tensor") {
    Tensor tensor = arange(0, 10, kInt);
    Tensor one = ones({}, kInt);
    for (int i = 0; i < tensor.numel(); ++i) {
      CATCH_REQUIRE(tensor[one * i].equal(one * i));
    }
    CATCH_REQUIRE_THROWS_WITH(
        tensor[ones({}) * 3.14].equal(one),
        StartsWith(
            "Can only index tensors with integral scalars"));
    CATCH_REQUIRE_THROWS_WITH(
        tensor[Tensor()].equal(one),
        StartsWith("Can only index with tensors that are defined"));
    CATCH_REQUIRE_THROWS_WITH(
        tensor[ones({2, 3, 4}, kInt)].equal(one),
        StartsWith("Can only index with tensors that are scalars (zero-dim)"));
  }
  CATCH_SECTION("dispatch") {
    Tensor tensor = randn({20, 20});
    Tensor other = randn({20, 20});
    auto result = tensor.m(relu).m(mse_loss, other, Reduction::ElementwiseMean);
    CATCH_REQUIRE(result.allclose(mse_loss(relu(tensor), other)));
  }
  CATCH_SECTION("core") {
    int i = CoreTest();
    CATCH_REQUIRE(i + 1 == CoreTest());
  }
}

CATCH_TEST_CASE( "basic tests CPU", "[cpu]" ) {
  manual_seed(123, at::kCPU);

  test(CPU(kFloat));
}

CATCH_TEST_CASE( "basic tests GPU", "[cuda]" ) {
  manual_seed(123, at::kCUDA);

  if(at::hasCUDA()) {
    test(CUDA(kFloat));
  }
}
