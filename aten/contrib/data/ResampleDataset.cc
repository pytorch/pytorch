#include "ResampleDataset.h"
#include "Dataset.h"
#include <vector>
#include <cassert>

using namespace at;

ResampleDataset::ResampleDataset(Dataset& dataset) {
   dataset_ = &dataset;
   size_ = dataset.size();
   perm_ = std::vector<uint64_t>();
   perm_.reserve(size_);
   for(int n = 0; n < size_; ++n)
      perm_[n] = n;
}

ResampleDataset::ResampleDataset(Dataset& dataset, std::vector<uint64_t>& perm) {
   dataset_ = &dataset;
   size_ = dataset.size();
   perm_ = perm;
   assert(perm_.size() == size_);
}

ResampleDataset::ResampleDataset(Dataset& dataset, std::function<uint64_t(uint64_t)> perm) {
   dataset_ = &dataset;
   size_ = dataset.size();
   permfunc_ = perm;
   resample();
}

void ResampleDataset::getField(uint64_t idx, std::string& fieldkey, at::Tensor& field) {
   assert(idx < size());
   assert(hasField(fieldkey));
   dataset_->getField(perm_[idx], fieldkey, field);
}

void ResampleDataset::resample() {
   if(permfunc_) {
      perm_.reserve(size_);
      for(int n = 0; n < size_; ++n)
         perm_[n] = permfunc_(n);
   }
}

uint64_t ResampleDataset::size() {
   return size_;
}
