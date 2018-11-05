#pragma once


namespace torch {
namespace jit {

class ScriptModuleSerializer {
 public:
  ScriptModuleSerializer(const std::string& filename) {
    // TODO
  }

  ScriptModuleSerializer(std::istream* is) {
    // TODO
  }

  void serialize(const script::Module& smodule) {
    // TODO
  }

 private:
  

};

class ScriptModuleDeserializer {
 public:
  ScriptModuleDeserializer(const std::string& filename) {
    // TODO
  }

  ScriptModuleSerializer(std::ostream* os) {
    // TODO
  }

  void deserialize(script::Module* smodule) {
    // TODO
  }
};

} // namespace jit
} // namespace torch
