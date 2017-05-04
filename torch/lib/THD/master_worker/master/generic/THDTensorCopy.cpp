#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorCopy.cpp"
#else

// TODO implement
void THDTensor_(copy)(THDTensor *tensor, THDTensor *src) {
  throw std::runtime_error("copy not implemented yet");
}

void THDTensor_(copyTH)(thpp::Tensor &from, THDTensor *to) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCopyTH, to),
    THDState::s_current_worker
  );

  thd::dataChannel->send(from, THDState::s_current_worker);
}

void THDTensor_(copyTHD)(THDTensor *from, thpp::Tensor &to) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCopyTHD, from),
    THDState::s_current_worker
  );

  thd::dataChannel->receive(to, THDState::s_current_worker);
}

#endif
