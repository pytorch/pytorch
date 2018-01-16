#ifndef AT_DATASET_H
#define AT_DATASET_H

#include "ATen/ATen.h"
#include <string>
#include <map>
#include <set>

typedef std::map<std::string, at::Tensor> Fields;

class Dataset {
   std::set<std::string> fieldkeys_;
public:
   virtual uint64_t size() = 0;  // pure virtual function
   virtual void getField(uint64_t idx, std::string& fieldkey, at::Tensor& field) = 0;
   virtual bool hasField(std::string& fieldkey);
   virtual std::set<std::string>& fieldKeys();
   virtual void addFieldKey(std::string& fieldkey);
   virtual void get(int64_t idx, Fields& fields);
   virtual ~Dataset();
};

#endif
