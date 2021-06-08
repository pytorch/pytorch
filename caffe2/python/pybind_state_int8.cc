/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Note(jiayq): the import_array function is done inside
// caffe2_python.cc. Read
// http://docs.scipy.org/doc/numpy-1.10.1/reference/c-api.array.html#miscellaneous
// for more details.
#define NO_IMPORT_ARRAY
#include "caffe2/python/pybind_state.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "caffe2/core/tensor_int8.h"

namespace caffe2 {
namespace python {

class Int8TensorFetcher : public BlobFetcherBase {
 public:
  pybind11::object Fetch(const Blob& blob) override {
#ifdef USE_NUMPY
    const caffe2::int8::Int8TensorCPU& src =
        blob.template Get<caffe2::int8::Int8TensorCPU>();
    const int numpy_type = CaffeToNumpyType(src.t.dtype());
    CAFFE_ENFORCE(numpy_type != -1, "Int8Tensor contains unknown type data");
    std::vector<npy_intp> npy_dims;
    for (const auto dim : src.t.sizes()) {
      npy_dims.push_back(dim);
    }
    auto data_array = pybind11::reinterpret_steal<pybind11::object>(
        PyArray_SimpleNew(src.t.sizes().size(), npy_dims.data(), numpy_type));
    void* ptr = static_cast<void*>(
        PyArray_DATA(reinterpret_cast<PyArrayObject*>(data_array.ptr())));
    CPUContext context;
    context.CopyBytesSameDevice(src.t.nbytes(), src.t.raw_data(), ptr);
    context.FinishDeviceComputation();

    auto result = pybind11::cast<pybind11::object>(
        pybind11::make_tuple(data_array, src.scale, src.zero_point));
    return result;
#else
    CAFFE_THROW("Caffe2 was compiled without NumPy support.");
#endif // USE_NUMPY
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_BLOB_FETCHER(
    (TypeMeta::Id<caffe2::int8::Int8TensorCPU>()),
    caffe2::python::Int8TensorFetcher);
} // namespace  python

} // namespace caffe2
