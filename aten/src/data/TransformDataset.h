#ifndef XT_TRANSFORM_DATASET_H
#define XT_TRANSFORM_DATASET_H

#include "Dataset.h"
#include "TensorLib/TensorLib.h"
#include <functional>
#include <string>

using namespace tlib;

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
