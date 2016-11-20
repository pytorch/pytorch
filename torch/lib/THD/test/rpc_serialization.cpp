#include <iostream>
#include <cassert>
#include <climits>
#include <typeinfo>

#include "../master_worker/common/RPC.hpp"

using namespace std;
using namespace thd::rpc;

int main() {
  auto msg = packMessage(1, 3, 1.0f, 100l, -12, LLONG_MAX);
  uint16_t fid = unpackFunctionId(msg);
  assert(fid == 1);
  uint16_t num_args = unpackArgCount(msg);
  assert(num_args == 3);
  double arg1 = unpackFloat(msg);
  assert(arg1 == 1.0);
  long long arg2 = unpackInteger(msg);
  assert(arg2 == 100);
  long long arg3 = unpackInteger(msg);
  assert(arg3 == -12);
  long long arg4 = unpackInteger(msg);
  assert(arg4 == LLONG_MAX);
  assert(msg.isEmpty());
  try {
    double arg5 = unpackFloat(msg);
    assert(false);
  } catch (exception &e) {}
  std::cout << "OK" << std::endl;
  return 0;
}
