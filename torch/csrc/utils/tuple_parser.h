#pragma once

#include <Python.h>
#include <memory>
#include <vector>
#include <THPP/THPP.h>

namespace torch {

struct TupleParser {
  TupleParser(PyObject* args, int num_args=-1);

  void parse(bool& x);
  void parse(int& x);
  void parse(double& x);
  void parse(std::unique_ptr<thpp::Tensor>& x);
  void parse(std::shared_ptr<thpp::Tensor>& x);
  void parse(std::vector<int>& x);

protected:
  std::runtime_error invalid_type(const char* expected);
  PyObject* next_arg();

private:
  PyObject* args;
  int idx;
};

} // namespace torch
