#ifndef XT_MERGE_DATASET_H
#define XT_MERGE_DATASET_H

#include "Dataset.h"
#include <vector>
#include <string>

class MergeDataset : public Dataset
{
public:
   MergeDataset(std::vector<Dataset*>& datasets);
   virtual void getField(uint64_t idx, std::string& fieldkey, tlib::Tensor& field);
   virtual uint64_t size();
private:
   std::vector<Dataset*>* datasets_;
   std::map<std::string, int> datasetidx_;
};

#endif
