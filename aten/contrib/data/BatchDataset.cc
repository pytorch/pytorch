#include "BatchDataset.h"
#include "Dataset.h"
#include "ATen/ATen.h"
#include <vector>
#include <cassert>
#include <math.h>

using namespace at;

BatchDataset::BatchDataset(Dataset& dataset, uint64_t batchsize) {
   BatchDataset(dataset, batchsize, true);
}

BatchDataset::BatchDataset(Dataset& dataset, uint64_t batchsize, bool fullbatches) {
   dataset_ = &dataset;
   size_ = dataset_->size();
   batchsize_ = batchsize;
   fullbatches_ = fullbatches;
}

void BatchDataset::getField(uint64_t idx, std::string& fieldkey, at::Tensor& field) {

   // assertions:
   assert(idx < size());
   assert(hasField(fieldkey));

   // loop over samples:
   Tensor singlefield, buffer;
   uint64_t maxsize = std::min(batchsize_, size_ - idx * batchsize_);
   for(int n = 0; n < maxsize; n++) {

      // get sample:
      uint64_t batchidx = idx * batchsize_ + n;
      dataset_->getField(batchidx, fieldkey, singlefield);

      // allocate memory for batch:
      if(n == 0) {

         // determine size of batch:
         std::vector<int64_t> fieldsize;
         fieldsize.push_back(maxsize);
         for(uint64_t d = 0; d < singlefield.dim(); ++d) {
            fieldsize.push_back(singlefield.size(d));
         }

         // resize buffer:
         field.resize_(fieldsize);
      }

      // copy sample into batch:
      buffer = select(field, 0, n);
      buffer.copy_(singlefield);
   }
}

uint64_t BatchDataset::size() {
   if(fullbatches_)
      return floor(size_ / batchsize_);
   else
      return ceil(size_ / batchsize_);
}
