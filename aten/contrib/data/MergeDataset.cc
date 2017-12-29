#include "MergeDataset.h"
#include <cassert>

using namespace at;

MergeDataset::MergeDataset(std::vector<Dataset*>& datasets) {
   datasets_ = &datasets;
   uint64_t idx = 0;
   for(Dataset* dataset : *datasets_) {
      for(auto& fieldkey : dataset->fieldKeys()) {
         std::string fieldkeyc = fieldkey;
         addFieldKey(fieldkeyc);
         datasetidx_[fieldkeyc] = idx;
      }
   }
}

void MergeDataset::getField(uint64_t idx, std::string& fieldkey, Tensor& field) {
   assert(idx < size());
   assert(hasField(fieldkey));
   Dataset* curdataset = (*datasets_)[datasetidx_[fieldkey]];
   return curdataset->getField(idx, fieldkey, field);
}

uint64_t MergeDataset::size() {
   uint64_t size = 0;
   for(Dataset* dataset : *datasets_)
      size += dataset->size();
   return size;
}
