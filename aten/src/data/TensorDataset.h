#ifndef XT_TENSOR_DATASET_H
#define XT_TENSOR_DATASET_H

#include "Dataset.h"
#include "TensorLib/TensorLib.h"
#include <string>

class TensorDataset : public Dataset
{
public:
   TensorDataset(tlib::Tensor& t, std::string& fieldkey);
   virtual void getField(uint64_t idx, std::string& fieldkey, tlib::Tensor& field);
   virtual uint64_t size();
private:
   tlib::Tensor t_;
   std::string fieldkey_;
};

#endif
