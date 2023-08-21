#if (!defined(FBCODE_CAFFE2) && defined(BUILD_ONEDNN_GRAPH))

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/csrc/jit/codegen/onednn/python_bindings.h>
#include <torch/csrc/utils/pybind.h>
#include <sstream>

#include <oneapi/dnnl/dnnl_graph.hpp>

using cpartition = dnnl::graph::compiled_partition;
using engine = dnnl::graph::engine;
using graph = dnnl::graph::graph;
using logical_tensor = dnnl::graph::logical_tensor;
using op = dnnl::graph::op;
using partition = dnnl::graph::partition;
using stream = dnnl::graph::stream;
using tensor = dnnl::graph::tensor;

void bind_cpartition(pybind11::module& m) {
  pybind11::class_<cpartition> p(m, "compiled_partition");

  p.def(pybind11::init<>())
      .def("query_logical_tensor", &cpartition::query_logical_tensor)
      .def("get_inplace_ports", &cpartition::get_inplace_ports)
      .def("execute", &cpartition::execute);
}

const std::string engine_kind2str(engine::kind v) {
  if (v == engine::kind::any)
    return "any";
  if (v == engine::kind::cpu)
    return "cpu";
  if (v == engine::kind::gpu)
    return "gpu";
  return "unknown engine_kind";
}

auto eng2string = [](const engine& eng) {
  std::stringstream ss;
  ss << "engine(kind = " << engine_kind2str(eng.get_kind()) << ")";
  return ss.str();
};

void bind_engine(pybind11::module& m) {
  pybind11::class_<engine> eng(m, "engine");

  eng.def(pybind11::init([](engine::kind akind, int index) {
    static dnnl::graph::allocator alloc{};
    auto aengine = engine(akind, index, alloc);
    return aengine;
  }));
  eng.def("get_kind", &engine::get_kind);
  eng.def("__repr__", eng2string);

  pybind11::enum_<engine::kind>(eng, "kind")
      .value("any", engine::kind::any)
      .value("cpu", engine::kind::cpu)
      .value("gpu", engine::kind::gpu)
      .export_values();
}

void bind_graph(pybind11::module& m) {
  pybind11::class_<graph> g(m, "graph");

  g.def(pybind11::init<engine::kind>());
  g.def(pybind11::init<engine::kind, graph::fpmath_mode>());
  g.def("add_op", &graph::add_op);
  // g.def("finalize", &graph::finalize);
  // g.def("is_finalized", &graph::is_finalized);
  g.def("get_partitions", &graph::get_partitions);

  pybind11::enum_<graph::fpmath_mode>(g, "fpmath_mode")
      .value("strict", graph::fpmath_mode::strict)
      .value("bf16", graph::fpmath_mode::bf16)
      .value("f16", graph::fpmath_mode::f16)
      .value("any", graph::fpmath_mode::any)
      .value("tf32", graph::fpmath_mode::tf32)
      .export_values();
}

const std::string data_type2str(logical_tensor::data_type v) {
  if (v == logical_tensor::data_type::undef)
    return "undef";
  if (v == logical_tensor::data_type::f16)
    return "f16";
  if (v == logical_tensor::data_type::bf16)
    return "bf16";
  if (v == logical_tensor::data_type::f32)
    return "f32";
  if (v == logical_tensor::data_type::s32)
    return "s32";
  if (v == logical_tensor::data_type::s8)
    return "s8";
  if (v == logical_tensor::data_type::u8)
    return "u8";
  return "unknown data_type";
}

const std::string layout_type2str(logical_tensor::layout_type v) {
  if (v == logical_tensor::layout_type::undef)
    return "undef";
  if (v == logical_tensor::layout_type::any)
    return "any";
  if (v == logical_tensor::layout_type::strided)
    return "strided";
  if (v == logical_tensor::layout_type::opaque)
    return "opaque";
  return "unknown layout_type";
}

const std::string dims2string(const std::vector<int64_t>& dims) {
  std::stringstream ss;
  ss << "(";
  const char* delimer = "";
  for (const auto& d : dims) {
    ss << delimer << d;
    delimer = ", ";
  }
  ss << ")";
  return ss.str();
};

auto lt2string = [](const logical_tensor& lt) {
  std::stringstream ss;
  ss << "logical_tensor(id = " << lt.get_id()
     << ", dtype = " << data_type2str(lt.get_data_type())
     << ", layout = " << layout_type2str(lt.get_layout_type())
     << ", shape = " << dims2string(lt.get_dims());
  if (lt.get_layout_type() == logical_tensor::layout_type::opaque) {
    ss << ", layout_id = " << lt.get_layout_id();
  } else {
    ss << ", stride = " << dims2string(lt.get_strides());
  }
  ss << ")";
  return ss.str();
};

void bind_logical_tensor(pybind11::module& m) {
  pybind11::class_<logical_tensor> lt(m, "logical_tensor");

  lt.def(pybind11::init<
             size_t,
             logical_tensor::data_type,
             int32_t,
             logical_tensor::layout_type,
             logical_tensor::property_type>())
      .def(pybind11::init<
           size_t,
           logical_tensor::data_type,
           logical_tensor::dims_t,
           logical_tensor::layout_type,
           logical_tensor::property_type>())
      .def(pybind11::init<
           size_t,
           logical_tensor::data_type,
           logical_tensor::dims_t,
           logical_tensor::dims_t,
           logical_tensor::property_type>())
      .def(pybind11::init<
           size_t,
           logical_tensor::data_type,
           logical_tensor::layout_type>())
      .def("get_id", &logical_tensor::get_id)
      .def("get_data_type", &logical_tensor::get_data_type)
      .def("get_layout_type", &logical_tensor::get_layout_type)
      .def("get_property_type", &logical_tensor::get_property_type)
      .def("get_layout_id", &logical_tensor::get_layout_id)
      .def("get_mem_size", &logical_tensor::get_mem_size)
      .def("get_dims", &logical_tensor::get_dims)
      .def_property_readonly("shape", &logical_tensor::get_dims)
      .def("get_strides", &logical_tensor::get_strides)
      .def("__repr__", lt2string);

  pybind11::enum_<logical_tensor::data_type>(lt, "data_type")
      .value("dt_undef", logical_tensor::data_type::undef)
      .value("f16", logical_tensor::data_type::f16)
      .value("bf16", logical_tensor::data_type::bf16)
      .value("f32", logical_tensor::data_type::f32)
      .value("s32", logical_tensor::data_type::s32)
      .value("s8", logical_tensor::data_type::s8)
      .value("u8", logical_tensor::data_type::u8)
      .value("boolean", logical_tensor::data_type::boolean)
      .export_values();

  pybind11::enum_<logical_tensor::layout_type>(lt, "layout_type")
      .value("lt_undef", logical_tensor::layout_type::undef)
      .value("any", logical_tensor::layout_type::any)
      .value("strided", logical_tensor::layout_type::strided)
      .value("opaque", logical_tensor::layout_type::opaque)
      .export_values();

  pybind11::enum_<logical_tensor::property_type>(lt, "property_type")
      .value("pt_undef", logical_tensor::property_type::undef)
      .value("variable", logical_tensor::property_type::variable)
      .value("constant", logical_tensor::property_type::constant)
      .export_values();
}

template <class T>
void set_op_attribute(op& aop, T x, op::attr attr) {
  if (pybind11::isinstance<pybind11::list>(x)) {
    if (pybind11::isinstance<pybind11::int_>(
            x.template cast<pybind11::list>()[0])) {
      std::vector<int64_t> int_attr = {};
      for (auto val : x.template cast<pybind11::list>()) {
        int_attr.push_back(val.template cast<int64_t>());
      }
      aop.set_attr<std::vector<int64_t>>(attr, int_attr);
    } else if (pybind11::isinstance<pybind11::float_>(
                   x.template cast<pybind11::list>()[0])) {
      std::vector<float> int_attr = {};
      for (auto val : x.template cast<pybind11::list>()) {
        int_attr.push_back(val.template cast<float>());
      }
      aop.set_attr<std::vector<float>>(attr, int_attr);
    } else {
      assert(!"unknown vector type");
    }
  } else if (pybind11::isinstance<pybind11::bool_>(x)) {
    aop.set_attr<bool>(attr, x.template cast<bool>());
  } else if (pybind11::isinstance<pybind11::int_>(x)) {
    aop.set_attr<int64_t>(attr, x.template cast<int64_t>());
  } else if (pybind11::isinstance<pybind11::float_>(x)) {
    aop.set_attr<float>(attr, x.template cast<float>());
  } else if (pybind11::isinstance<pybind11::str>(x)) {
    aop.set_attr<std::string>(attr, x.template cast<std::string>());
  } else {
    assert(!"unknown attribute type");
  }
}

void bind_op(pybind11::module& m) {
  pybind11::class_<op> opr(m, "op");

  opr.def(pybind11::init<size_t, op::kind, std::string>());

  opr.def(pybind11::init([](size_t id, op::kind kind, std::string name) {
    auto aop = op(id, kind, name);
    return aop;
  }));

  opr.def("set_attributes", [](op& aop, op::attr key, pybind11::object val) {
    set_op_attribute(aop, val, key);
  });
  opr.def("add_input", &op::add_input);
  opr.def("add_inputs", &op::add_inputs);
  opr.def("add_output", &op::add_output);
  opr.def("add_outputs", &op::add_outputs);

  pybind11::enum_<op::kind>(opr, "kind")
      .value("Abs", op::kind::Abs)
      .value("AbsBackprop", op::kind::AbsBackprop)
      .value("Add", op::kind::Add)
      .value("AvgPool", op::kind::AvgPool)
      .value("AvgPoolBackprop", op::kind::AvgPoolBackprop)
      .value("BatchNormForwardTraining", op::kind::BatchNormForwardTraining)
      .value("BatchNormInference", op::kind::BatchNormInference)
      .value("BatchNormTrainingBackprop", op::kind::BatchNormTrainingBackprop)
      .value("BiasAdd", op::kind::BiasAdd)
      .value("BiasAddBackprop", op::kind::BiasAddBackprop)
      .value("Clamp", op::kind::Clamp)
      .value("ClampBackprop", op::kind::ClampBackprop)
      .value("Concat", op::kind::Concat)
      .value("Convolution", op::kind::Convolution)
      .value("ConvolutionBackpropData", op::kind::ConvolutionBackpropData)
      .value("ConvolutionBackpropFilters", op::kind::ConvolutionBackpropFilters)
      .value("ConvTranspose", op::kind::ConvTranspose)
      .value("ConvTransposeBackpropData", op::kind::ConvTransposeBackpropData)
      .value(
          "ConvTransposeBackpropFilters",
          op::kind::ConvTransposeBackpropFilters)
      .value("Dequantize", op::kind::Dequantize)
      .value("Divide", op::kind::Divide)
      .value("DynamicDequantize", op::kind::DynamicDequantize)
      .value("DynamicQuantize", op::kind::DynamicQuantize)
      .value("Elu", op::kind::Elu)
      .value("EluBackprop", op::kind::EluBackprop)
      .value("End", op::kind::End)
      .value("Exp", op::kind::Exp)
      .value("GELU", op::kind::GELU)
      .value("GELUBackprop", op::kind::GELUBackprop)
      .value("HardSigmoid", op::kind::HardSigmoid)
      //.value("HardSigmoidBackward", op::kind::HardSigmoidBackward)
      .value("HardSwish", op::kind::HardSwish)
      .value("HardSwishBackprop", op::kind::HardSwishBackprop)
      .value("Interpolate", op::kind::Interpolate)
      .value("InterpolateBackprop", op::kind::InterpolateBackprop)
      .value("LayerNorm", op::kind::LayerNorm)
      .value("LayerNormBackprop", op::kind::LayerNormBackprop)
      .value("LeakyReLU", op::kind::LeakyReLU)
      .value("Log", op::kind::Log)
      .value("LogSoftmax", op::kind::LogSoftmax)
      .value("LogSoftmaxBackprop", op::kind::LogSoftmaxBackprop)
      .value("MatMul", op::kind::MatMul)
      .value("Maximum", op::kind::Maximum)
      .value("MaxPool", op::kind::MaxPool)
      .value("MaxPoolBackprop", op::kind::MaxPoolBackprop)
      .value("Minimum", op::kind::Minimum)
      .value("Mish", op::kind::Mish)
      .value("MishBackprop", op::kind::MishBackprop)
      .value("Multiply", op::kind::Multiply)
      .value("PReLU", op::kind::PReLU)
      .value("PReLUBackprop", op::kind::PReLUBackprop)
      .value("Quantize", op::kind::Quantize)
      .value("Reciprocal", op::kind::Reciprocal)
      .value("ReduceL1", op::kind::ReduceL1)
      .value("ReduceL2", op::kind::ReduceL2)
      .value("ReduceMax", op::kind::ReduceMax)
      .value("ReduceMean", op::kind::ReduceMean)
      .value("ReduceMin", op::kind::ReduceMin)
      .value("ReduceProd", op::kind::ReduceProd)
      .value("ReduceSum", op::kind::ReduceSum)
      .value("ReLU", op::kind::ReLU)
      .value("ReLUBackprop", op::kind::ReLUBackprop)
      .value("Reorder", op::kind::Reorder)
      .value("Round", op::kind::Round)
      .value("Select", op::kind::Select)
      .value("Sigmoid", op::kind::Sigmoid)
      .value("SigmoidBackprop", op::kind::SigmoidBackprop)
      .value("SoftMax", op::kind::SoftMax)
      .value("SoftMaxBackprop", op::kind::SoftMaxBackprop)
      .value("SoftPlus", op::kind::SoftPlus)
      .value("SoftPlusBackprop", op::kind::SoftPlusBackprop)
      .value("Sqrt", op::kind::Sqrt)
      .value("SqrtBackprop", op::kind::SqrtBackprop)
      .value("Square", op::kind::Square)
      .value("SquaredDifference", op::kind::SquaredDifference)
      .value("StaticReshape", op::kind::StaticReshape)
      .value("StaticTranspose", op::kind::StaticTranspose)
      .value("Subtract", op::kind::Subtract)
      .value("Tanh", op::kind::Tanh)
      .value("TanhBackprop", op::kind::TanhBackprop)
      .value("TypeCast", op::kind::TypeCast)
      .value("Wildcard", op::kind::Wildcard)
      .export_values();

  pybind11::enum_<op::attr>(opr, "attr")
      .value("undef", op::attr::undef)
      .value("alpha", op::attr::alpha)
      .value("beta", op::attr::beta)
      .value("epsilon", op::attr::epsilon)
      .value("max", op::attr::max)
      .value("min", op::attr::min)
      .value("momentum", op::attr::momentum)
      .value("scales", op::attr::scales)
      .value("axis", op::attr::axis)
      .value("begin_norm_axis", op::attr::begin_norm_axis)
      .value("groups", op::attr::groups)
      .value("axes", op::attr::axes)
      .value("dilations", op::attr::dilations)
      .value("filter_shape", op::attr::filter_shape)
      .value("input_shape", op::attr::input_shape)
      .value("kernel", op::attr::kernel)
      .value("order", op::attr::order)
      .value("output_padding", op::attr::output_padding)
      .value("output_shape", op::attr::output_shape)
      .value("pads_begin", op::attr::pads_begin)
      .value("pads_end", op::attr::pads_end)
      .value("shape", op::attr::shape)
      .value("sizes", op::attr::sizes)
      .value("strides", op::attr::strides)
      .value("zps", op::attr::zps)
      .value("exclude_pad", op::attr::exclude_pad)
      .value("keep_dims", op::attr::keep_dims)
      .value("keep_stats", op::attr::keep_stats)
      .value("per_channel_broadcast", op::attr::per_channel_broadcast)
      .value("special_zero", op::attr::special_zero)
      .value("transpose_a", op::attr::transpose_a)
      .value("transpose_b", op::attr::transpose_b)
      .value("use_affine", op::attr::use_affine)
      .value("use_dst", op::attr::use_dst)
      .value("auto_broadcast", op::attr::auto_broadcast)
      .value("auto_pad", op::attr::auto_pad)
      .value(
          "coordinate_transformation_mode",
          op::attr::coordinate_transformation_mode)
      .value("data_format", op::attr::data_format)
      .value("filter_format", op::attr::filter_format)
      .value("mode", op::attr::mode)
      .value("qtype", op::attr::qtype)
      .value("rounding_type", op::attr::rounding_type)
      .export_values();
}

void bind_partition(pybind11::module& m) {
  pybind11::class_<partition> p(m, "partition");

  p.def(pybind11::init<>())
      .def("get_ops_num", &partition::get_ops_num)
      .def("get_ops", &partition::get_ops)
      .def("get_id", &partition::get_id)
      .def("is_supported", &partition::is_supported)
      .def("get_in_ports", &partition::get_in_ports)
      .def("get_out_ports", &partition::get_out_ports)
      .def("get_engine_kind", &partition::get_engine_kind)
      .def(
          "compile",
          [](partition& self,
             std::vector<logical_tensor>& inputs,
             std::vector<logical_tensor>& outputs,
             engine& e) { return self.compile(inputs, outputs, e); },
          pybind11::arg("inputs"),
          pybind11::arg("outputs"),
          pybind11::arg("engine"));

  pybind11::enum_<partition::policy>(p, "policy")
      .value("fusion", partition::policy::fusion)
      .value("debug", partition::policy::debug)
      .export_values();
}

void bind_stream(pybind11::module& m) {
  pybind11::class_<stream> strm(m, "stream");

  strm.def(pybind11::init([](engine& eng) { return stream(eng); }))
      .def("wait", &stream::wait);
}

static size_t size_of(logical_tensor::data_type dtype) {
  switch (dtype) {
    case logical_tensor::data_type::f32:
    case logical_tensor::data_type::s32:
      return 4U;
    case logical_tensor::data_type::s8:
    case logical_tensor::data_type::u8:
      return 1U;
    case logical_tensor::data_type::f16:
    case logical_tensor::data_type::bf16:
      return 2U;
    default:
      return 0;
  }
}

static std::string format_string(logical_tensor::data_type dtype) {
  switch (dtype) {
    case logical_tensor::data_type::f32:
    case logical_tensor::data_type::f16:
    case logical_tensor::data_type::bf16:
      return pybind11::format_descriptor<float>::format();
      break;
    case logical_tensor::data_type::u8:
      return pybind11::format_descriptor<uint8_t>::format();
      break;
    case logical_tensor::data_type::s8:
      return pybind11::format_descriptor<int8_t>::format();
      break;
    default:
      throw std::runtime_error("Not supported data type in current example.");
  }
}

pybind11::buffer_info to_buffer_info(tensor& t, logical_tensor& lt) {
  auto strides = lt.get_strides();
  auto shapes = lt.get_dims();
  auto dtype = lt.get_data_type();
  std::transform(
      strides.begin(), strides.end(), strides.begin(), [&](int64_t i) {
        return i * size_of(dtype);
      });
  return pybind11::buffer_info(
      t.get_data_handle(), /* Pointer to buffer */
      size_of(dtype), /* Size of one scalar */
      format_string(dtype), /* Python struct-style format descriptor */
      shapes.size(), /* Number of dimensions */
      shapes, /* Buffer dimensions */
      strides);
}

void bind_tensor(pybind11::module& m) {
  pybind11::class_<tensor>(m, "tensor", pybind11::buffer_protocol())
      .def(pybind11::init(
          [](logical_tensor& lt, engine& eng, pybind11::buffer b) {
            auto bufinfo = b.request();
            return tensor(lt, eng, bufinfo.ptr);
          }))
      .def(pybind11::init([](logical_tensor& lt, engine& eng) {
        return tensor(lt, eng, nullptr);
      }))
      .def(pybind11::init(
          [](logical_tensor& lt, engine& eng, uint64_t data_ptr) {
            return tensor(lt, eng, reinterpret_cast<void*>(data_ptr));
          }))
      .def("get_engine", &tensor::get_engine)
      .def(
          "from_aten",
          [](tensor& x, uint64_t data_ptr) {
            x.set_data_handle(reinterpret_cast<void*>(data_ptr));
          },
          pybind11::arg("data_ptr"))
      .def(
          "from_numpy",
          [](tensor& x, pybind11::buffer b) {
            auto bufinfo = b.request();
            x.set_data_handle(bufinfo.ptr);
          },
          pybind11::arg("b"))
      .def(
          "to_numpy",
          [](tensor& x, logical_tensor& lt) {
            auto bufinfo = to_buffer_info(x, lt);
            return pybind11::array(bufinfo);
          },
          pybind11::arg("lt"));
}

void bind_status(pybind11::module& m) {
  pybind11::enum_<dnnl::graph::status>(m, "status")
      .value("success", dnnl::graph::status::success)
      .value("out_of_memory", dnnl::graph::status::out_of_memory)
      .value("invalid_arguments", dnnl::graph::status::invalid_arguments)
      .value("unimplemented", dnnl::graph::status::unimplemented)
      .value("iterator_ends", dnnl::graph::status::interator_ends)
      .value("runtime_error", dnnl::graph::status::runtime_error)
      .value("not_required", dnnl::graph::status::not_required)
      .value("invalid_graph", dnnl::graph::status::invalid_graph)
      .value("invalid_graph_op", dnnl::graph::status::invalid_graph_op)
      .value("invalid_shape", dnnl::graph::status::invalid_shape)
      .value("invalid_data_type", dnnl::graph::status::invalid_data_type)
      .export_values();
}

void initOnednnPythonBindings(PyObject* module) {
  auto m = pybind11::handle(module).cast<pybind11::module>();
  //! Top Level oneDNN Python submodule
  auto llga = m.def_submodule("_onednn");
  llga.doc() = R"pbdoc(
        oneDNN Graph API Python binding
        -------------------------------
        .. currentmodule:: llgap
        .. autosummary::
        :toctree: _generate
    )pbdoc";
  bind_status(llga);
  bind_graph(llga);
  bind_logical_tensor(llga);
  bind_engine(llga);
  bind_op(llga);
  bind_tensor(llga);
  bind_partition(llga);
  bind_cpartition(llga);
  bind_stream(llga);
}

#endif
