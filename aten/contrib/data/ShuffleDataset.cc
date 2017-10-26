#include "ShuffleDataset.h"
#include "Dataset.h"
#include <algorithm>

using namespace at;

ShuffleDataset::ShuffleDataset(Dataset& dataset) : ResampleDataset(dataset) {
   resample();
}

void ShuffleDataset::resample() {
   perm_.reserve(size_);
   for(int n = 0; n < size_; ++n)
      perm_[n] = n;
   std::random_shuffle(perm_.begin(), perm_.end());
}
