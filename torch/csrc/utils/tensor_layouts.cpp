#include <ATen/Layout.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_layouts.h>

namespace torch { namespace utils {

#define REGISTER_LAYOUT(layout, LAYOUT)                          \
    PyObject* layout##_layout =                                  \
      THPLayout_New(at::Layout::LAYOUT, "torch." # layout);      \
    Py_INCREF(layout##_layout);                                         \
    if (PyModule_AddObject(torch_module, "" # layout, layout##_layout) != 0) { \
      throw python_error();                                             \
    }                                                                   \
    registerLayoutObject((THPLayout*)layout##_layout, at::Layout::LAYOUT);

void initializeLayouts() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch_module) throw python_error();

  PyObject* strided_layout = THPLayout_New(at::Layout::Strided, "torch.strided");
  Py_INCREF(strided_layout);
  if (PyModule_AddObject(torch_module, "strided", strided_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)strided_layout, at::Layout::Strided);

  PyObject* sparse_coo_layout = THPLayout_New(at::Layout::Sparse, "torch.sparse_coo");
  Py_INCREF(sparse_coo_layout);
  if (PyModule_AddObject(torch_module, "sparse_coo", sparse_coo_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_coo_layout, at::Layout::Sparse);

  REGISTER_LAYOUT(sparse_csr, SparseCsr)
  REGISTER_LAYOUT(sparse_csc, SparseCsc)
  REGISTER_LAYOUT(sparse_bsr, SparseBsr)
  REGISTER_LAYOUT(sparse_bsc, SparseBsc)

  PyObject* sparse_csc_layout =
      THPLayout_New(at::Layout::SparseCsc, "torch.sparse_csc");
  Py_INCREF(sparse_csc_layout);
  if (PyModule_AddObject(torch_module, "sparse_csc", sparse_csc_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_csc_layout, at::Layout::SparseCsc);

  PyObject* sparse_bsr_layout =
      THPLayout_New(at::Layout::SparseBsr, "torch.sparse_bsr");
  Py_INCREF(sparse_bsr_layout);
  if (PyModule_AddObject(torch_module, "sparse_bsr", sparse_bsr_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_bsr_layout, at::Layout::SparseBsr);

  PyObject* sparse_bsc_layout =
      THPLayout_New(at::Layout::SparseBsc, "torch.sparse_bsc");
  Py_INCREF(sparse_bsc_layout);
  if (PyModule_AddObject(torch_module, "sparse_bsc", sparse_bsc_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)sparse_bsc_layout, at::Layout::SparseBsc);

  PyObject* mkldnn_layout = THPLayout_New(at::Layout::Mkldnn, "torch._mkldnn");
  Py_INCREF(mkldnn_layout);
  if (PyModule_AddObject(torch_module, "_mkldnn", mkldnn_layout) != 0) {
    throw python_error();
  }
  registerLayoutObject((THPLayout*)mkldnn_layout, at::Layout::Mkldnn);
}

}} // namespace torch::utils
