#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "caffe2/perfkernels/fused_8bit_rowwise_conversion.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace caffe2 {
namespace python {
namespace py = pybind11;

PYBIND11_MODULE(caffe2_perfkernels_pybind11, m) {
  // in place fake rowwise uint8. Original tensor will be destroyed.
  m.def("rowwise_uint8_fake_quant", [](py::object& arg) {
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(arg.ptr());
    array = PyArray_GETCONTIGUOUS(array);
    const auto npy_type = PyArray_TYPE(array);
    assert(npy_type == NPY_FLOAT);

    // get dims
    int ndim = PyArray_NDIM(array);
    assert(ndim == 2);
    npy_intp* npy_dims = PyArray_DIMS(array);
    int num_row = npy_dims[0];
    int num_col = npy_dims[1];

    float* input = reinterpret_cast<float*>(PyArray_DATA(array));

    int uint8_num_col = num_col + 8;
    std::vector<std::uint8_t> rowwise_uint8(num_row * uint8_num_col, 0);

    caffe2::FloatToFused8BitRowwiseQuantized(
        input, num_row, num_col, rowwise_uint8.data());
    caffe2::Fused8BitRowwiseQuantizedToFloat(
        rowwise_uint8.data(), num_row, uint8_num_col, input);
  });
}
} // namespace python
} // namespace caffe2
