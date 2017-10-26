#ifndef AT_TENSOR_DATASET_H
#define AT_TENSOR_DATASET_H

#include "Dataset.h"
#include "ATen/ATen.h"
#include <string>

class TensorDataset : public Dataset
{
public:
   TensorDataset(at::Tensor& t, std::string& fieldkey);
   virtual void getField(uint64_t idx, std::string& fieldkey, at::Tensor& field);
   virtual uint64_t size();
private:
   at::Tensor t_;
   std::string fieldkey_;
};

#endif
