#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensor.h"
#else

struct THDTensor {
  unsigned long long tensor_id;
  int node_id;
  int device_id;
};

THD_API void THDTensor_(new)();
THD_API void THDTensor_(newWithSize)(THLongStorage *sizes, THLongStorage *strides);
THD_API void THDTensor_(add)(THDTensor *result, THDTensor *source, double value);

#endif
