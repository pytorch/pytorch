#include "TensorDataset.h"
#include "ATen/ATen.h"
#include <cassert>

using namespace at;

TensorDataset::TensorDataset(Tensor& t, std::string& fieldkey) {
   t_ = t;
   fieldkey_ = fieldkey;
   addFieldKey(fieldkey);
}

void TensorDataset::getField(uint64_t idx, std::string& fieldkey, Tensor& field) {

   // assertions:
   assert(idx < size());
   assert(fieldkey_.compare(fieldkey) == 0);

   // get sample:
   Tensor buffer = select(t_, 0, idx);
   field.copy_(buffer);

}

uint64_t TensorDataset::size() {
   return t_.size(0);
}
