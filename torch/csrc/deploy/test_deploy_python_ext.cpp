#include <caffe2/torch/csrc/deploy/deploy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdio>
#include <iostream>

bool run() {
  torch::deploy::InterpreterManager m(2);
  m.register_module_source("check_none", "check = id(None)\n");
  int64_t id0 = 0, id1 = 0;
  {
    auto I = m.all_instances()[0].acquire_session();
    id0 = I.global("check_none", "check").toIValue().toInt();
  }
  {
    auto I = m.all_instances()[1].acquire_session();
    id1 = I.global("check_none", "check").toIValue().toInt();
  }
  return id0 != id1;
}

PYBIND11_MODULE(test_deploy_python_ext, m) {
  m.def("run", run);
}
