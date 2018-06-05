#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"

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
  SECTION( "resize" ) {
    auto a = type.tensor();
    a.resize_({3,4});
    REQUIRE(a.numel() == 12);
    a.resize_({5, 7});
    REQUIRE(a.numel() == 35);

  }

  SECTION( "ones and dot" ) {
    Tensor b0 = ones(type, {1, 1});
    REQUIRE(2 == (b0+b0).sum().toCDouble());

    Tensor b1 = ones(type, {1, 2});
    REQUIRE(4 == (b1+b1).sum().toCDouble());

    Tensor b = ones(type, {3, 4});
    REQUIRE(24 == (b+b).sum().toCDouble());
    REQUIRE(12 == b.numel());
    REQUIRE(b.view(-1).dot(b.view(-1)).toCDouble() == 12);
  }

  SECTION( "rand" ) {
    for(auto i = 0; i < 10; i++) {
      Tensor a = rand(type.toScalarType(i % 2 == 0 ? kFloat : kDouble), {3,4});
      //std::cout << a << std::endl;
      //TODO EXPECT
    }
  }

  SECTION( "sort" ) {
    Tensor b = rand(type, {3, 4});

    auto z = b.sort(1);
    auto z_sorted = std::get<0>(z);

    REQUIRE(Scalar(z_sorted[0][0]).toFloat() < Scalar(z_sorted[0][1]).toFloat());
  }

  if(type.backend() != kCUDA)
  SECTION( "randperm" ) {
    Tensor b = randperm(type, 15);
    Tensor rv, ri;
    std::tie(rv, ri) = sort(b, 0);
    REQUIRE(Scalar(rv[0]).toFloat() <= Scalar(rv[1]).toFloat());
  }

  SECTION( "context" ) {
    std::stringstream ss;
    ss << "context: " << std::hex << (int64_t)&globalContext() << std::endl;
  }

  SECTION( "add" ) {
    Tensor a = rand(type, {3, 4});
    Tensor b = rand(type, {3, 4});
    Tensor c = add(a, add(a, b));
    //TODO:0-dim Tensor d(3.f);
    Scalar d = 3.f;
    REQUIRE( add(c, d).allclose(a + a + b + d) );
  }

  SECTION( "loads of adds" ) {
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones(type, {3, 4});
    Tensor r = zeros(type, {3, 4});
    for(auto i = 0; i < 100000; i++) {
      add_out(r, r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    //TODO TEST PERF?
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    REQUIRE(norm(100000*d).toCDouble() == norm(r).toCDouble());
  }

  SECTION( "loads of adds (with copy)" ) {
    auto begin = std::chrono::high_resolution_clock::now();
    Tensor d = ones(type, {3, 4});
    Tensor r = zeros(type, {3, 4});
    for(auto i = 0; i < 100000; i++) {
      r = add(r, d);
    }
    auto end = std::chrono::high_resolution_clock::now();
    //TODO TEST PERF?
    std::cout << std::dec << "   " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;
    REQUIRE(norm(100000*d).toCDouble() == norm(r).toCDouble());
  }

  SECTION( "isContiguous" ) {
    Tensor a = rand(type, {3, 4});
    REQUIRE(a.is_contiguous());
    a = a.transpose(0, 1);
    REQUIRE(!a.is_contiguous());
  }

  SECTION( "permute" ) {
    Tensor a = rand(type, {3, 4, 5});
    Tensor b = a.permute({1, 2, 0});
    REQUIRE(b.sizes().equals({4, 5, 3}));
    REQUIRE(b.strides().equals({5, 1, 20}));
  }

  SECTION( "mm" ) {
    Tensor a = rand(type, {3, 4});
    Tensor b = rand(type, {4});
    Tensor c = mv(a, b);
    REQUIRE(c.equal(addmv(zeros(type, {3}), a, b, 0, 1)));
  }

  SECTION( "squeeze" ) {
    Tensor a = rand(type, {2, 1});
    Tensor b = squeeze(a);
    REQUIRE(b.dim() == 1);
    a = rand(type, {1});
    b = squeeze(a);
    //TODO 0-dim squeeze
    REQUIRE(a[0].equal(b));
  }

  SECTION( "copy" ) {
    Tensor a = zeros(type, {4, 3});
    Tensor e = rand(type, {4, 3});
    a.copy_(e);
    REQUIRE(a.equal(e));
  }

  SECTION( "copy (broadcasting)" ) {
    Tensor a = zeros(type, {4, 3});
    Tensor e = rand(type, {3});
    a.copy_(e);
    for (int i = 0; i < 4; ++i) {
      REQUIRE(a[i].equal(e));
    }
  }

  SECTION( "abs(value)" ) {
    Tensor r = at::abs(type.scalarTensor(-3));
    REQUIRE(Scalar(r).toInt() == 3);
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

  SECTION( "adding a value with a scalar" ) {
    Tensor a = rand(type, {4, 3});
    REQUIRE((ones(type, {4,3}) + a).equal(add(a,1)));
  }

  SECTION( "select" ) {
    Tensor a = rand(type, {3, 7});
    auto a_13 = select(a, 1, 3);
    auto a_13_02 = select(select(a, 1, 3), 0, 2);
    REQUIRE( a[0][3].equal(a_13[0]) );
    REQUIRE( a[2][3].equal(a_13_02) );
  }

  SECTION( "zero-dim" ) {
    Tensor a =  type.scalarTensor(4); //rand(type, {1});

    REQUIRE_NOTHROW(Scalar(a));
    Tensor b = rand(type, {3,4});
    REQUIRE((a + a).dim() == 0);
    REQUIRE((1 + a).dim() == 0);
    REQUIRE((b + a).dim() == 2);
    REQUIRE((a + b).dim() == 2);
    auto c = rand(type, {3,4});
    REQUIRE(c[1][2].dim() == 0);

    auto f = rand(type, {3,4});
    f[2] = zeros(type, {4});
    f[1][0] = -1;
    REQUIRE(Scalar(f[2][0]).toDouble() == 0);
  }

  SECTION( "tensor from TH" ) {
    int a = 4;
    THFloatTensor *t = THFloatTensor_newWithSize2d(a, a);
    THFloatTensor_fill(t, a);
    Tensor tt = CPU(kFloat).unsafeTensorFromTH(t,false);
    REQUIRE_NOTHROW(tt);
  }

  SECTION( "toCFloat" ) {
    Tensor a = zeros(CPU(kFloat), {3,4});
    Tensor b = ones(CPU(kFloat), {3,7});
    Tensor c = cat({a,b},1);
    REQUIRE(c.size(1) == 11);

    Tensor e = rand(CPU(kFloat), {});
    REQUIRE(*e.data<float>() == e.sum().toCFloat());
  }

  SECTION( "to string" ) {
    Tensor b = ones(CPU(kFloat), {3,7})*.0000001f;
    std::stringstream s;
    s << b << "\n";
    std::string expect = "1e-07 *";
    REQUIRE(s.str().substr(0,expect.size()) == expect);
  }
  SECTION("indexing by Scalar") {
    Tensor tensor = CPU(kInt).arange(0, 10);
    Tensor one = CPU(kInt).ones({1});
    for (int64_t i = 0; i < tensor.numel(); ++i) {
      REQUIRE(tensor[i].equal(one * i));
    }
    for (size_t i = 0; i < static_cast<uint64_t>(tensor.numel()); ++i) {
      REQUIRE(tensor[i].equal(one * static_cast<int64_t>(i)));
    }
    for (int i = 0; i < tensor.numel(); ++i) {
      REQUIRE(tensor[i].equal(one * i));
    }
    for (int16_t i = 0; i < tensor.numel(); ++i) {
      REQUIRE(tensor[i].equal(one * i));
    }
    for (int8_t i = 0; i < tensor.numel(); ++i) {
      REQUIRE(tensor[i].equal(one * i));
    }
    REQUIRE_THROWS_WITH(
        tensor[Scalar(3.14)].equal(one),
        StartsWith(
            "Can only index tensors with integral scalars (got CPUDoubleType)"));
  }
  SECTION("indexing by zero-dim tensor") {
    Tensor tensor = CPU(kInt).arange(0, 10);
    Tensor one = CPU(kInt).ones({});
    for (int i = 0; i < tensor.numel(); ++i) {
      REQUIRE(tensor[one * i].equal(one * i));
    }
    REQUIRE_THROWS_WITH(
        tensor[CPU(kFloat).ones({}) * 3.14].equal(one),
        StartsWith(
            "Can only index tensors with integral scalars (got CPUFloatType)"));
    REQUIRE_THROWS_WITH(
        tensor[Tensor()].equal(one),
        StartsWith("Can only index with tensors that are defined"));
    REQUIRE_THROWS_WITH(
        tensor[CPU(kInt).ones({2, 3, 4})].equal(one),
        StartsWith("Can only index with tensors that are scalars (zero-dim)"));
  }
  SECTION("dispatch") {
    Tensor tensor = CPU(kFloat).randn({20, 20});
    Tensor other = CPU(kFloat).randn({20, 20});
    auto result = tensor.m(relu).m(mse_loss, other, true, true);
    REQUIRE(result.allclose(mse_loss(relu(tensor), other)));
  }
}

TEST_CASE( "basic tests CPU", "[cpu]" ) {
  manual_seed(123, at::Backend::CPU);

  test(CPU(kFloat));
}

TEST_CASE( "basic tests GPU", "[cuda]" ) {
  manual_seed(123, at::Backend::CUDA);

  if(at::hasCUDA()) {
    test(CUDA(kFloat));
  }
}
