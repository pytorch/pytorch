#include <c10/util/irange.h>
#include <torch/csrc/jit/python/python_arg_flatten.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/six.h>

#include <torch/csrc/autograd/grad_mode.h>

namespace torch::jit::python {

using namespace torch::autograd;
using namespace at;

// Alphabet used to describe structure of inputs/outputs (D for desc)
namespace D {
static constexpr char DictOpen = '<';
static constexpr char DictClose = '>';
static constexpr char ListOpen = '[';
static constexpr char ListClose = ']';
static constexpr char TupleOpen = '(';
static constexpr char TupleClose = ')';
static constexpr char Variable = 'v';
static constexpr char Bool = 'b';
static constexpr char Long = 'l';
static constexpr char Double = 'd';
static constexpr char String = 's';
static constexpr char NoneType = 'n';
} // namespace D

namespace {

inline bool PyNone_Check(PyObject* o) {
  return o == Py_None;
}

template <typename T>
py::object cast_handle_sequence(std::vector<py::handle> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (const auto i : c10::irange(num_objs)) {
    sequence[i] = py::reinterpret_borrow<py::object>(objs[i]);
  }
  return sequence;
}

void flatten_rec(PyObject* obj, ParsedArgs& args) {
  auto& structure = args.desc.structure;
  if (six::isTuple(obj)) {
    structure.push_back(D::TupleOpen);
    for (auto item : py::reinterpret_borrow<py::tuple>(obj))
      flatten_rec(item.ptr(), args);
    structure.push_back(D::TupleClose);
  } else if (PyList_Check(obj)) {
    structure.push_back(D::ListOpen);
    for (auto item : py::reinterpret_borrow<py::list>(obj))
      flatten_rec(item.ptr(), args);
    structure.push_back(D::ListClose);
  } else if (PyDict_Check(obj)) {
    auto* dict_items = PyDict_Items(obj);
    structure.push_back(D::DictOpen);
    for (auto item : py::reinterpret_borrow<py::list>(dict_items)) {
      flatten_rec(item.ptr(), args);
    }
    structure.push_back(D::DictClose);
    Py_DECREF(dict_items);
  } else if (THPUtils_checkString(obj)) {
    string str = THPUtils_unpackString(obj);
    args.desc.strings.emplace_back(str);
    args.desc.structure.push_back(D::String);
  } else if (THPVariable_Check(obj)) {
    auto& var = THPVariable_Unpack(obj);
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Variable);
  } else if (PyNone_Check(obj)) {
    args.desc.structure.push_back(D::NoneType);
  } else if (PyBool_Check(obj)) { // Wrap bools in Bool tensors
    at::Tensor var = scalar_to_tensor(at::Scalar(THPUtils_unpackBool(obj)));
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Bool);
  } else if (PyLong_Check(obj)) { // Wrap longs in Long tensors
    at::Tensor var = scalar_to_tensor(
        at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(obj))));
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Long);
  } else if (PyFloat_Check(obj)) { // Wrap floats in Double tensors
    at::Tensor var = scalar_to_tensor(THPUtils_unpackDouble(obj));
    args.vars.push_back(var);
    args.desc.metadata.emplace_back(var);
    args.desc.structure.push_back(D::Double);
  } else {
    std::string msg =
        "Only tuples, lists and Variables are supported as JIT inputs/outputs. "
        "Dictionaries and strings are also accepted, but their usage is not "
        "recommended. Here, received an input of unsupported type: ";
    msg += THPUtils_typename(obj);
    throw std::runtime_error(msg);
  }
}

} // anonymous namespace

ParsedArgs flatten(py::handle obj) {
  ParsedArgs args;
  args.desc.grad_enabled = autograd::GradMode::is_enabled();
  flatten_rec(obj.ptr(), args);
  return args;
}

namespace {

template <typename T>
py::object cast_sequence(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  T sequence{num_objs};
  for (const auto i : c10::irange(num_objs)) {
    sequence[i] = std::move(objs[i]);
  }
  return std::move(sequence);
}

py::object cast_dict(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  py::dict sequence = {};
  for (const auto i : c10::irange(num_objs)) {
    py::tuple obj = py::reinterpret_borrow<py::tuple>(objs[i]);
    sequence[obj[0]] = obj[1];
  }
  return std::move(sequence);
}

py::object unflatten_rec(
    ArrayRef<Variable>::iterator& var_it,
    ArrayRef<Variable>::iterator& var_it_end,
    std::string::const_iterator& desc_it,
    std::vector<string>::const_iterator& str_it,
    std::vector<string>::const_iterator& str_it_end) {
  char type = *desc_it++;
  if (type == D::TupleOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::TupleClose)
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    ++desc_it;
    return cast_sequence<py::tuple>(objs);
  } else if (type == D::ListOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::ListClose)
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    ++desc_it;
    return cast_sequence<py::list>(objs);
  } else if (type == D::DictOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::DictClose) {
      objs.push_back(
          unflatten_rec(var_it, var_it_end, desc_it, str_it, str_it_end));
    }
    ++desc_it;
    return cast_dict(objs);
  } else if (type == D::String) {
    if (str_it == str_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto str = *str_it++;
    return py::reinterpret_borrow<py::object>(THPUtils_packString(str));
  } else if (type == D::NoneType) {
    return py::reinterpret_borrow<py::object>(py::none());
  } else {
    // if (type == D::Long || type == D::Double || type == D::Bool ||
    // D::Variable) unwrap variables (D::Variable), or unwrap primitive types
    // (Long, Double, Bool) as variables for tracer.
    if (var_it == var_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto var = *var_it++;
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(var));
  }
}

} // anonymous namespace

PyObject* unflatten(ArrayRef<Variable> vars, const IODescriptor& desc) {
  // NB: We don't do correctness checking on descriptor.
  // It has to be a correct bytes object produced by unflatten.
  auto vars_it = vars.begin();
  auto vars_it_end = vars.end();
  auto desc_it = desc.structure.begin();
  std::vector<std::string>::const_iterator str_it = desc.strings.begin();
  std::vector<std::string>::const_iterator str_end = desc.strings.end();
  auto output = unflatten_rec(vars_it, vars_it_end, desc_it, str_it, str_end);
  if (vars_it != vars_it_end)
    throw std::runtime_error("Too many Variables given to unflatten");
  return output.release().ptr();
}

} // namespace torch::jit::python
