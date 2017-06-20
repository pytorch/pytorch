#include "Dataset.h"
#include "DatasetIterator.h"
#include "TensorDataset.h"
#include <iostream>

using namespace tlib;

int main()
{
   std::cout << "hello\n";

   Tensor tensor = CPU(kDouble).rand({256,32});

   TensorDataset dataset(tensor);
   DatasetIterator datasetiterator(dataset);
   uint64_t cnt = 0;
   for(auto& sample : datasetiterator) {
      std::cout << "got sample " << cnt << std:endl;
      cnt++;
   }
   return 0;
}
