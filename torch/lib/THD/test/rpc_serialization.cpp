#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <typeinfo>

#include "../master_worker/common/RPC.hpp"
#include "TH/THStorage.h"

using namespace std;
using namespace thd::rpc;

constexpr ptrdiff_t STORAGE_SIZE = 10;

int main() {
  THLongStorage *storage1 = THLongStorage_newWithSize(STORAGE_SIZE);
  long *data = storage1->data;
  for (long i = 0; i < STORAGE_SIZE; i++)
    data[i] = i;
  std::unique_ptr<RPCMessage> msg_ptr =
    packMessage(1, 1.0f, 100l, -12, LLONG_MAX, storage1);
  auto &msg = *msg_ptr;
  uint16_t fid = unpackFunctionId(msg);
  assert(fid == 1);
  double arg1 = unpackFloat(msg);
  assert(arg1 == 1.0);
  long long arg2 = unpackInteger(msg);
  assert(arg2 == 100);
  long long arg3 = unpackInteger(msg);
  assert(arg3 == -12);
  long long arg4 = unpackInteger(msg);
  assert(arg4 == LLONG_MAX);
  THLongStorage *storage2 = unpackTHLongStorage(msg);
  assert(storage2->size == STORAGE_SIZE);
  for (long i = 0; i < STORAGE_SIZE; i++)
    assert(storage2->data[i] == i);
  assert(msg.isEmpty());
  try {
    double arg6 = unpackFloat(msg);
    assert(false);
  } catch (exception &e) {}
  std::cout << "OK" << std::endl;
  return 0;
}
