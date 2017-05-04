#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorCopy.h"
#else

THD_API void THDTensor_(copy)(THDTensor *tensor, THDTensor *src);

THD_API void THDTensor_(copyTH)(thpp::Tensor &from, THDTensor *to);
THD_API void THDTensor_(copyTHD)(THDTensor *from, thpp::Tensor &to);

#endif
