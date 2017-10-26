#ifndef AT_SHUFFLE_DATASET_H
#define AT_SHUFFLE_DATASET_H

#include "Dataset.h"
#include "ResampleDataset.h"

class ShuffleDataset : public ResampleDataset
{
public:
   ShuffleDataset(Dataset& dataset);
   virtual void resample();
};

#endif
