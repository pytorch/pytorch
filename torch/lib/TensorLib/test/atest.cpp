#include "TensorLib/TensorLib.h"

#include<iostream>
using namespace std;
using namespace tlib;

int main() {
  auto foo = CPU(kFloat).rand({12,6});
  cout << foo << "\n" << foo.size(0) << " " << foo.size(1) << endl;

  foo = foo+foo*3;
  foo -= 4;


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
  return 0;
}
