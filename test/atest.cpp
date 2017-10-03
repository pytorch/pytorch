#include "ATen/ATen.h"

#include<iostream>
using namespace std;
using namespace at;

void check(bool c) {
  if(!c)
    throw std::runtime_error("check failed.");
}

void trace() {
  Tensor foo = CPU(kFloat).rand({12,12});

  // assert foo is 2-dimensional and holds floats.
  auto foo_a = foo.accessor<float,2>();
  float trace = 0;

  for(int i = 0; i < foo_a.size(0); i++) {
    trace += foo_a[i][i];
  }
  cout << trace << "\n" << foo << "\n";
}
int main() {
  auto foo = CPU(kFloat).rand({12,6});
  assert(foo.data<float>() == foo.toFloatData());

  cout << foo << "\n" << foo.size(0) << " " << foo.size(1) << endl;

  foo = foo+foo*3;
  foo -= 4;

  bool threw = false;
  try {
    Tensor no;
    add_out(no,foo,foo);
  } catch (std::runtime_error&) {
    threw = true;
  }
  check(threw);
  Scalar a = 4;

  float b = a.to<float>();
  check(b == 4);

  foo = (foo*foo) == (foo.pow(3));
  foo =  2 + (foo+1);
  //foo = foo[3];
  auto foo_v = foo.accessor<uint8_t,2>();

  cout << foo_v.size(0) << " " << foo_v.size(1) << endl;
  for(int i = 0; i < foo_v.size(0); i++) {
    for(int j = 0; j < foo_v.size(1); j++) {
      //cout << foo_v[i][j] << " ";
      foo_v[i][j]++;
    }
    //cout << "\n";
  }


  cout << foo << "\n";

  trace();

  float data[] = { 1, 2, 3,
                   4, 5, 6};

  auto f = CPU(kFloat).tensorFromBlob(data, {1,2,3});

  cout << f << endl;
  cout << f.strides() << " " << f.sizes() << endl;
  threw = false;
  try {
    f.resize_({3,4,5});
  } catch(std::runtime_error&) {
    threw = true;
  }
  check(threw);
  return 0;
}
