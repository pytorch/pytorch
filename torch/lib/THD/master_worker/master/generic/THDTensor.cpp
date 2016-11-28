#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

static THDTensor* THDTensor_(_alloc)() {
  THDTensor* new_tensor = new THDTensor();
  new_tensor->tensor_id = THDState::s_nextId++;
  new_tensor->refcount = 1;
  return new_tensor;
}

THDTensor* THDTensor_(new)() {
  THDTensor* tensor = THDTensor_(_alloc)();
  Type constructed_type = tensor_type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::construct,
      constructed_type,
      tensor
    ),
    THDState::s_current_worker
  );
  return tensor;
}

THDTensor* THDTensor_(newWithSize)(THLongStorage *sizes, THLongStorage *strides) {
  THDTensor* tensor = THDTensor_(_alloc)();
  Type constructed_type = tensor_type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::constructWithSize,
      constructed_type,
      tensor,
      sizes,
      strides
    ),
    THDState::s_current_worker
  );
  return tensor;
}

void THDTensor_(free)(THDTensor *tensor) {
  // TODO: refcount
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::free,
      tensor->tensor_id
    ),
    THDState::s_current_worker
  );
}


#endif
