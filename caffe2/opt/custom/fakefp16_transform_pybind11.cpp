#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "caffe2/opt/custom/fakefp16_transform.h"

#include <gtest/gtest.h>

namespace py = pybind11;

namespace caffe2 {
namespace opt {

PYBIND11_MODULE(fakefp16_transform_pybind11, m) {
    m.doc() = "Fake Fp16 transformations";
    m.def("fakeFp16FuseOps", [](const py::bytes& net_str) {
        caffe2::NetDef netDef;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(
                net_str.cast<std::string>(), &netDef),
            "broken pred_net protobuf");
        fakeFp16FuseOps(&netDef);
        std::string out_net;
        netDef.SerializeToString(&out_net);
        return py::bytes(out_net);
    }, "Fuse Fake FP16 operators");
}

}
}
