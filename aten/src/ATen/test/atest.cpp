#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "ATen/ATen.h"
#include "test_seed.h"

#include<iostream>
using namespace std;
using namespace at;

void trace() {
  Tensor foo = rand(CPU(kFloat), {12,12});

  // ASSERT foo is 2-dimensional and holds floats.
  auto foo_a = foo.accessor<float,2>();
  float trace = 0;

  for(int i = 0; i < foo_a.size(0); i++) {
    trace += foo_a[i][i];
  }

  REQUIRE(Scalar(foo.trace()).toFloat() == Approx(trace));
}

TEST_CASE( "atest", "[]" ) {

  manual_seed(123, at::Backend::CPU);
  manual_seed(123, at::Backend::CUDA);

  auto foo = rand(CPU(kFloat), {12,6});
  REQUIRE(foo.data<float>() == foo.toFloatData());

  REQUIRE(foo.size(0) == 12);
  REQUIRE(foo.size(1) == 6);

  foo = foo+foo*3;
  foo -= 4;

  {
    Tensor no;
    REQUIRE_THROWS(add_out(no,foo,foo));
  }
  Scalar a = 4;

  float b = a.to<float>();
  REQUIRE(b == 4);

  foo = (foo*foo) == (foo.pow(3));
  foo =  2 + (foo+1);
  //foo = foo[3];
  auto foo_v = foo.accessor<uint8_t,2>();

  for(int i = 0; i < foo_v.size(0); i++) {
    for(int j = 0; j < foo_v.size(1); j++) {
      foo_v[i][j]++;
    }
  }

  REQUIRE(foo.equal(4 * CPU(kByte).ones({12, 6})));

  trace();

  float data[] = { 1, 2, 3,
                   4, 5, 6};

  auto f = CPU(kFloat).tensorFromBlob(data, {1,2,3});
  auto f_a = f.accessor<float,3>();

  REQUIRE(f_a[0][0][0] == 1.0);
  REQUIRE(f_a[0][1][1] == 5.0);

  REQUIRE(f.strides()[0] == 6);
  REQUIRE(f.strides()[1] == 3);
  REQUIRE(f.strides()[2] == 1);
  REQUIRE(f.sizes()[0] == 1);
  REQUIRE(f.sizes()[1] == 2);
  REQUIRE(f.sizes()[2] == 3);

  REQUIRE_THROWS(f.resize_({3,4,5}));
  {
    int isgone = 0;
    {
      auto f2 = CPU(kFloat).tensorFromBlob(data, {1,2,3}, [&](void*) {
        isgone++;
      });
    }
    REQUIRE(isgone == 1);
  }
  {
    int isgone = 0;
    Tensor a_view;
    {
      auto f2 = CPU(kFloat).tensorFromBlob(data, {1,2,3}, [&](void*) {
        isgone++;
      });
      a_view = f2.view({3,2,1});
    }
    REQUIRE(isgone == 0);
    a_view.reset();
    REQUIRE(isgone == 1);
  }

  if(at::hasCUDA()) {
    int isgone = 0;
    {
      auto f2 = CUDA(kFloat).tensorFromBlob(nullptr, {1,2,3}, [&](void*) {
        isgone++;
      });
    }
    REQUIRE(isgone==1);
  }
}
