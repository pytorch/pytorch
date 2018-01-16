#ifndef AT_BATCH_DATASET_H
#define AT_BATCH_DATASET_H

#include "Dataset.h"
#include "ATen/ATen.h"

class BatchDataset : public Dataset
{
public:
   BatchDataset(Dataset& dataset, uint64_t batchsize);
   BatchDataset(Dataset& dataset, uint64_t batchsize, bool fullbatches);
   virtual void getField(uint64_t idx, std::string& fieldkey, at::Tensor& field);
   virtual uint64_t size();
private:
   Dataset* dataset_;
   uint64_t batchsize_;
   uint64_t size_;
   bool fullbatches_;
};

#endif
