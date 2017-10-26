#ifndef AT_TRANSFORM_DATASET_H
#define AT_TRANSFORM_DATASET_H

#include "Dataset.h"
#include "ATen/ATen.h"
#include <functional>
#include <string>

using namespace at;

class TransformDataset : public Dataset
{
public:
   TransformDataset(Dataset& dataset, std::string& fieldkey, std::function<Tensor(Tensor)>& transform);
   virtual void getField(uint64_t idx, std::string& fieldkey, Tensor& field);
   virtual uint64_t size();
private:
   Dataset* dataset_;
   std::string fieldkey_;
   std::function<Tensor(Tensor)> transform_;
};

#endif
