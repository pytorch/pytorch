#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorCopy.cpp"
#else

// TODO implement
void THDTensor_(copy)(THDTensor *tensor, THDTensor *src) {
  throw std::runtime_error("copy not implemented yet");
}

#endif
