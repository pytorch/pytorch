#include "Dataset.h"
#include <cassert>

typedef std::map<std::string, at::Tensor> Fields;

void Dataset::get(int64_t idx, Fields& fields) {
   for(auto& field : fields) {
      std::string fieldname = field.first;
      assert(hasField(fieldname));
      getField(idx, fieldname, field.second);
   }
}

bool Dataset::hasField(std::string& fieldkey) {
   auto search = fieldkeys_.find(fieldkey);
   return (search != fieldkeys_.end());
}

std::set<std::string>& Dataset::fieldKeys() {
   return fieldkeys_;
}

void Dataset::addFieldKey(std::string& fieldkey) {
   fieldkeys_.insert(fieldkey);
}

Dataset::~Dataset() {
}
