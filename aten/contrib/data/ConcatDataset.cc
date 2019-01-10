#include "ConcatDataset.h"
#include "Dataset.h"
#include <vector>
#include <cassert>

using namespace at;

ConcatDataset::ConcatDataset(std::vector<Dataset*>& datasets) {
   datasets_ = &datasets;
   size_ = 0;
   beginindices_  = std::vector<uint64_t>();
   endindices_    = std::vector<uint64_t>();
   beginindices_.push_back(0);
   for(Dataset* dataset : datasets) {
      size_ += dataset->size();
      uint64_t curidx = endindices_.back();
      endindices_.push_back(beginindices_.back() + dataset->size());
      if(endindices_.size() > 1)
         beginindices_.push_back(curidx + 1);
   }
}

void ConcatDataset::getField(uint64_t idx, std::string& fieldkey, Tensor &field) {

   // assertions:
   assert(idx < size());
   assert(hasField(fieldkey));

   // get sample from correct dataset:
   uint64_t datasetidx = binarySearch(idx);
   Dataset* curdataset = (*datasets_)[datasetidx];
   curdataset->getField(idx - beginindices_[datasetidx], fieldkey, field);
}

uint64_t ConcatDataset::binarySearch(uint64_t idx) {
   assert(idx < size());  // TODO: Add caching to this method.
   uint64_t left = 0;
   uint64_t right = size_ - 1;
   while(left != right) {
      uint64_t middle = (right - left) / 2;
      if(left == middle) {
         if(idx > endindices_[left])
            left = right;
         else
            right = left;
      }
      else {
         if(idx > endindices_[middle])
            left = middle;
         else if(idx < beginindices_[middle])
            right = middle;
         else {
            left  = middle;
            right = middle;
         }
      }
   }
   return left;
}

uint64_t ConcatDataset::size() {
   return size_;
}
