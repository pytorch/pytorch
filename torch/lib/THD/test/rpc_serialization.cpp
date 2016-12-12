#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <typeinfo>
#include <vector>

#include <THPP/Type.hpp>

#include "../master_worker/common/RPC.hpp"
#include "TH/THStorage.h"

using namespace std;
using namespace thd;
using namespace thd::rpc;

constexpr ptrdiff_t STORAGE_SIZE = 10;
constexpr size_t VEC_SIZE = 3;

int main() {
  THLongStorage *storage1 = THLongStorage_newWithSize(STORAGE_SIZE);
  long *data = storage1->data;
  for (long i = 0; i < STORAGE_SIZE; i++)
    data[i] = i;
  std::vector<long> vec(VEC_SIZE, 7);  // VEC_SIZE sevens
  std::unique_ptr<RPCMessage> msg_ptr =
    packMessage(1, 1.0f, 100l, -12, LLONG_MAX, storage1, vec);
  auto &msg = *msg_ptr;

  uint16_t fid = unpackFunctionId(msg);
  assert(fid == 1);

  assert(peekType(msg) == thpp::Type::FLOAT);
  double arg1 = unpackFloat(msg);
  assert(arg1 == 1.0);

  assert(peekType(msg) == thpp::Type::LONG);
  long long arg2 = unpackInteger(msg);
  assert(arg2 == 100);

  assert(peekType(msg) == thpp::Type::INT);
  long long arg3 = unpackInteger(msg);
  assert(arg3 == -12);

  assert(peekType(msg) == thpp::Type::LONG_LONG);
  long long arg4 = unpackInteger(msg);
  assert(arg4 == LLONG_MAX);

  assert(peekType(msg) == thpp::Type::LONG_STORAGE);
  THLongStorage *storage2 = unpackTHLongStorage(msg);
  assert(storage2->size == STORAGE_SIZE);
  for (long i = 0; i < STORAGE_SIZE; i++)
    assert(storage2->data[i] == i);
  
  int vec_size = unpackInteger(msg);
  assert(vec_size == VEC_SIZE);
  for (int i = 0; i < VEC_SIZE; i++)
    assert(unpackInteger(msg) == 7);

  assert(msg.isEmpty());
  try {
    double arg6 = unpackFloat(msg);
    assert(false);
  } catch (exception &e) {}
  std::cout << "OK" << std::endl;
  return 0;
}
