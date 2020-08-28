#include <torch/csrc/utils/pybind.h>
#include <nvToolsExt.h>

namespace torch { namespace cuda { namespace shared {

void initNvtxBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto nvtx = m.def_submodule("_nvtx", "libNvToolsExt.so bindings");
  nvtx.def("rangePushA", nvtxRangePushA);
  nvtx.def("rangePop", nvtxRangePop);
  nvtx.def("markA", nvtxMarkA);
  //newer additions, to add color
  nvtx.def("rangePushEx", nvtxRangePushEx);
  nvtx.def("markEx", nvtxMarkEx);
  nvtx.attr("version") = NVTX_VERSION;
  nvtx.attr("size") = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  // THIS DOESNT WORK
  // nvtx.attr("NVTX_COLOR_UNKNOWN") = nvtxColorType_t::NVTX_COLOR_UNKNOWN;
  py::enum_<nvtxColorType_t>(nvtx, "nvtxColorType_t")
    .value("NVTX_COLOR_UNKNOWN",    nvtxColorType_t::NVTX_COLOR_UNKNOWN)
    .value("NVTX_COLOR_ARGB",       nvtxColorType_t::NVTX_COLOR_ARGB)
    .export_values();
  py::enum_<nvtxMessageType_t>(nvtx, "nvtxMessageType_t")
    .value("NVTX_MESSAGE_UNKNOWN",          nvtxMessageType_t::NVTX_MESSAGE_UNKNOWN)
    .value("NVTX_MESSAGE_TYPE_ASCII",       nvtxMessageType_t::NVTX_MESSAGE_TYPE_ASCII)
    .value("NVTX_MESSAGE_TYPE_UNICODE",     nvtxMessageType_t::NVTX_MESSAGE_TYPE_UNICODE)
    .value("NVTX_MESSAGE_TYPE_REGISTERED",  nvtxMessageType_t::NVTX_MESSAGE_TYPE_REGISTERED) //TODO: nvToolsExt.h says only NVTX_VERSION_2, do I have to be careful with this?
    .export_values();
  /* //I could never get this union to work
  py::class_<nvtxMessageValue_t>(nvtx, "nvtxMessageValue_t")
    .def(py::init<>())
    .def_property("ascii",
            [](nvtxMessageValue_t& self) -> const char*& 
            { 
                return self.ascii; 
            },
            [](nvtxMessageValue_t& self , const char*& value)
            { 
                self.ascii = value; 
            }) 
    .def_property("unicode",
            [](nvtxMessageValue_t& self) -> const wchar_t*& 
            { 
                return self.unicode;
            },
            [](nvtxMessageValue_t& self , const wchar_t*& value)
            { 
                self.unicode = value; 
            }); 
    */
  py::class_<nvtxMessageValue_t>(nvtx, "nvtxMessageValue_t")
    .def(py::init<>())
    .def_readwrite("ascii",&nvtxMessageValue_t::ascii)
    .def_readwrite("unicode",&nvtxMessageValue_t::unicode);
    //.def_readwrite("registered",&nvtxMessageValue_t::registered);
  py::class_<nvtxEventAttributes_t>(nvtx, "nvtxEventAttributes_t")
    .def(py::init<>())
    .def_readwrite("version",&nvtxEventAttributes_t::version)
    .def_readwrite("size",&nvtxEventAttributes_t::size)
    .def_readwrite("colorType",&nvtxEventAttributes_t::colorType)
    .def_readwrite("color",&nvtxEventAttributes_t::color)
    .def_readwrite("messageType",&nvtxEventAttributes_t::messageType)
    .def_readwrite("message",&nvtxEventAttributes_t::message);
}

} // namespace shared
} // namespace cuda
} // namespace torch
