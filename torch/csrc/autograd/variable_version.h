#pragma once

#include <memory>

// Every Variable has a version counter. Version counters are incremented
// whenever  the data or shape of a tensor changes through Variable operations.
// These are typicallly in-place operations. Version counters are used to
// detect modifications to saved varaibles which would result in incorrect
// gradient calculations. Version counters may be shared between Variables:
//
// 1. A view shares the version counter of the base Variable
// 2. Detached variables share the version counter of the source
// 3. Unpacked saved variables share the version counter of the source

namespace torch { namespace autograd {

struct VersionBlock {
  VersionBlock() : version() {}

  // monotonically increasing version
  std::atomic<int> version;
};

struct SavedVersion;

struct VariableVersion {
  VariableVersion() : version_block(std::make_shared<VersionBlock>()) {}
  VariableVersion(const VariableVersion&) = delete;
  VariableVersion(VariableVersion&&) = delete;

  // increment the version counter
  void increment() { version_block->version++; }

  // current version
  int current_version() const { return version_block->version.load(); }

  // creates a saved reference with the current version and the counter
  inline SavedVersion save() const;

  // Uses another variable's version counter. Used for variables which share storages
  // NOTE: not thread-safe to call this from multiple threads without synchronization
  // because shared_ptr assignment isn't thread-safe.
  VariableVersion& operator=(const VariableVersion& other) {
    version_block = other.version_block;
    return *this;
  }

  // Uses the version counter from a SavedVariable
  // NOTE: not thread-safe to call this from multiple threads without synchronization
  inline VariableVersion& operator=(const SavedVersion& other);

private:
  friend struct SavedVersion;
  std::shared_ptr<VersionBlock> version_block; // always non-null
};

// The version counter used in SavedVariables. Saves the expected_version (the
// version at the time of save) and a reference to the version counter's
// version_block.
struct SavedVersion {
  SavedVersion() {}
  SavedVersion(const VariableVersion& version)
    : expected_version(version.current_version())
    , version_block(version.version_block) {}

  // if the version_block has been modified since when it was saved
  bool is_modified() const {
    return expected_version != version_block->version.load();
  }

  // true if the version_block is defined
  bool defined() const {
    return static_cast<bool>(version_block);
  }

private:
  friend struct VariableVersion;
  int expected_version;
  std::shared_ptr<VersionBlock> version_block;  // may be null
};

SavedVersion VariableVersion::save() const {
  return SavedVersion(*this);
}

VariableVersion& VariableVersion::operator=(const SavedVersion& other) {
  if (!other.version_block) {
    throw std::runtime_error(
        "Can't take version counter from empty SavedVersion. File a bug report.");
  }
  version_block = other.version_block;
  return *this;
}


}} // namespace torch::autograd
