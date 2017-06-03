#pragma once

#include <Python.h>
#include <memory>
#include <vector>
#include <THPP/THPP.h>

namespace torch {

struct TupleParser {
  TupleParser(PyObject* args, int num_args=-1);

  void parse(bool& x, const std::string& param_name);
  void parse(int& x, const std::string& param_name);
  void parse(double& x, const std::string& param_name);
  void parse(std::unique_ptr<thpp::Tensor>& x, const std::string& param_name);
  void parse(std::shared_ptr<thpp::Tensor>& x, const std::string& param_name);
  void parse(std::vector<int>& x, const std::string& param_name);
  void parse(std::string& x, const std::string& param_name);

protected:
  std::runtime_error invalid_type(const std::string& expected, const std::string& param_name);
  PyObject* next_arg();

private:
  PyObject* args;
  int idx;
};

} // namespace torch
