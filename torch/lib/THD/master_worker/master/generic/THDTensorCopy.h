#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorCopy.h"
#else

THD_API void THDTensor_(copy)(THDTensor *tensor, THDTensor *src);

THD_API void THDTensor_(copyFromMaster)(THDTensorDescriptor* from, THDTensor *to);
THD_API void THDTensor_(copyFromWorker)(THDTensor *from, THDTensorDescriptor *to);

#endif
