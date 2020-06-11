// Standalone demo. Compile with:
// g++ -o test --include=prelude.h test.cpp
// LD_PRELOAD=./tracker.so ./test

#include <atomic>
#include <iostream>
#include <unordered_map>

using namespace std;

void foo();

int main() {
  cout << "before atomics" << endl;
  foo();
  cout << "after atomics" << endl;
  return 0;
}
