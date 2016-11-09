#ifndef THD_TENSOR_H
#define THD_TENSOR_H

struct THDTensor {
  unsigned long long tensor_id;
  int node_id;
  int device_id;
};

void THDTensor_add(THDTensor *result, THDTensor *source, double value);

#endif
