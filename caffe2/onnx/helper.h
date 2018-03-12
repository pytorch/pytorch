#pragma once

#include <set>
#include <string>
#include <unordered_set>

namespace caffe2 { namespace onnx {
class DummyName {
  public:
    static std::string NewDummyName();

    static void Reset(const std::unordered_set<std::string>& used_names);

    static void AddName(const std::string& new_used) {
      get_used_names().insert(new_used);
    }

   private:
     static std::unordered_set<std::string>& get_used_names();
     static size_t counter_;
};

}}
