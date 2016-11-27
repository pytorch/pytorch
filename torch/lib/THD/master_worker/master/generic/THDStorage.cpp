#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDStorage.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

static THDStorage* THDStorage_(_alloc)() {
  THDStorage* new_tensor = new THDStorage();
  new_tensor->storage_id = THDState::s_nextId++;
  new_tensor->node_id = THDState::s_current_worker;
  return new_tensor;
}

THDStorage* THDStorage_(new)() {
  THDStorage* tensor = THDStorage_(_alloc)();
  std::unique_ptr<RPCMessage> construct_message = packMessage(
      Functions::construct,
      static_cast<char>(tensor_type_traits<real>::type),
      *tensor
  );
  masterCommandChannel->sendMessage(
      std::move(construct_message),
      THDState::s_current_worker
  );
  return tensor;
}

THDStorage* THDStorage_(newWithSize)(THLongStorage *sizes, THLongStorage *strides) {
  THDStorage* tensor = THDStorage_(_alloc)();
  std::unique_ptr<RPCMessage> construct_message = packMessage(
      Functions::constructWithSize,
      static_cast<char>(tensor_type_traits<real>::type),
      *tensor,
      sizes,
      strides
  );
  masterCommandChannel->sendMessage(
      std::move(construct_message),
      THDState::s_current_worker
  );
  return tensor;
}

void THDStorage_(free)(THDStorage *tensor) {
  std::unique_ptr<RPCMessage> free_message = packMessage(
      Functions::free,
      tensor->tensor_id
  );
  masterCommandChannel->sendMessage(
      std::move(free_message),
      THDState::s_current_worker
  );
}

#endif
