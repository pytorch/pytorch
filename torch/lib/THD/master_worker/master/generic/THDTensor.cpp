#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

void THDTensor_(new)() {
  std::unique_ptr<RPCMessage> construct_message =
      packMessage(Functions::construct);
  masterCommandChannel->sendMessage(
      std::move(construct_message),
      THDState::s_current_worker
  );
}

void THDTensor_(newWithSize)(THLongStorage *sizes, THLongStorage *strides) {
  std::unique_ptr<RPCMessage> construct_message = packMessage(
            Functions::constructWithSize,
            sizes,
            strides
  );
  masterCommandChannel->sendMessage(
            std::move(construct_message),
            THDState::s_current_worker
  );
}

#endif
