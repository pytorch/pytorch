#pragma once

#include <THPP/THPP.h>
#include <memory>

namespace torch { namespace autograd {

struct VariableVersion;

struct SavedVariable {
  SavedVariable()
    : data()
    , expected_version(-1)
    , version() {}

  SavedVariable(
      std::unique_ptr<thpp::Tensor> data,
      int expected_version,
      std::unique_ptr<VariableVersion> version)
    : data(std::move(data))
    , expected_version(expected_version)
    , version(std::move(version)) {}

  std::unique_ptr<thpp::Tensor> data;
  int expected_version;
  std::unique_ptr<VariableVersion> version;

  std::unique_ptr<thpp::Tensor>& unpack();
};

}} // namespace torch::autograd
