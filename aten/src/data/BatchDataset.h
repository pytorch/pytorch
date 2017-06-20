#ifndef XT_BATCH_DATASET_H
#define XT_BATCH_DATASET_H

#include "Dataset.h"
#include "TensorLib/TensorLib.h"

class BatchDataset : public Dataset
{
public:
   BatchDataset(Dataset& dataset, uint64_t batchsize);
   BatchDataset(Dataset& dataset, uint64_t batchsize, bool fullbatches);
   virtual void getField(uint64_t idx, std::string& fieldkey, tlib::Tensor& field);
   virtual uint64_t size();
private:
   Dataset* dataset_;
   uint64_t batchsize_;
   uint64_t size_;
   bool fullbatches_;
};

#endif
