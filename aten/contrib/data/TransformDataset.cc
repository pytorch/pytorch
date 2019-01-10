#include "TransformDataset.h"
#include "ATen/ATen.h"
#include "ATen/ATen.h"
#include <cassert>

using namespace at;

TransformDataset::TransformDataset(Dataset& dataset, std::string& fieldkey, std::function<Tensor(Tensor)>& transform) {
   assert(hasField(fieldkey));
   dataset_ = &dataset;
   fieldkey_ = fieldkey;
   transform_ = transform;
}

void TransformDataset::getField(uint64_t idx, std::string& fieldkey, Tensor& field) {
   dataset_->getField(idx, fieldkey, field);
   if(fieldkey.compare(fieldkey_) == 0) {
      Tensor transformed = transform_(field);
      field.copy_(transformed);
   }
}

uint64_t TransformDataset::size() {
   return dataset_->size();
}
