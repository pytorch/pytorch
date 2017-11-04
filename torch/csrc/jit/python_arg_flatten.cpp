#include "python_arg_flatten.h"

namespace torch { namespace jit { namespace python {

// Alphabet used to describe structure of inputs/outputs (D for desc)
namespace D {
static constexpr char ListOpen          = '[';
static constexpr char ListClose         = ']';
static constexpr char TupleOpen         = '(';
static constexpr char TupleClose        = ')';
static constexpr char VariableVolatile  = 'v';
static constexpr char VariableGrad      = 'r';
static constexpr char VariableNoGrad    = 'n';
} // namespace D

struct ParsedArgs {
  // Flat vector of Variables found in arguments
  std::vector<py::handle> vars;
  // Description of argument structure. Variables are replaced with
  // different characters, depending on their flags, beginnings and
  // ends of tuples and lists are denoted by a pair of parenthesis
  // of their corresponding kind. They should always be paired.
  // Example desc: (rn[n(r)r]). Would be (vv[v(v)v]) if **any**
  // input Variable was volatile (even non-volatile ones are marked with v).
  std::string desc;
  // True iff any of vars is volatile
  bool is_volatile = false;
};

namespace {

template<typename T>
py::object cast_handle_sequence(std::vector<py::handle> objs) {
  auto num_objs = objs.size();
  T sequence { num_objs };
  for (std::size_t i = 0; i < num_objs; ++i)
    sequence[i] = py::reinterpret_borrow<py::object>(objs[i]);
  return sequence;
}

void flatten_rec(PyObject* obj, ParsedArgs& args) {
  if (PyTuple_Check(obj)) {
    args.desc.push_back(D::TupleOpen);
    for (auto item : py::reinterpret_borrow<py::tuple>(obj))
      flatten_rec(item.ptr(), args);
    args.desc.push_back(D::TupleClose);
  } else if (PyList_Check(obj)) {
    args.desc.push_back(D::ListOpen);
    for (auto item : py::reinterpret_borrow<py::list>(obj))
      flatten_rec(item.ptr(), args);
    args.desc.push_back(D::ListClose);
  } else if (THPVariable_Check(obj)) {
    auto& var = reinterpret_cast<THPVariable*>(obj)->cdata;
    args.vars.push_back(obj);
    args.is_volatile |= var.is_volatile();
    if (args.is_volatile) {
      args.desc.push_back(D::VariableVolatile);
    } else {
      args.desc.push_back(var.requires_grad() ? D::VariableGrad : D::VariableNoGrad);
    }
  } else {
    std::string msg = "Only tuples, lists and Variables supported as JIT inputs, but got ";
    msg += THPUtils_typename(obj);
    throw std::runtime_error(msg);
  }
}

void mark_all_volatile(std::string& desc) {
  auto desc_size = desc.size();
  for (std::size_t i = 0; i < desc_size; ++i) {
    if (desc[i] == D::VariableGrad || desc[i] == D::VariableNoGrad)
      desc[i] = D::VariableVolatile;
    // Once we find a volatile var, we know that all later ones were marked
    // as volatile too.
    else if (desc[i] == D::VariableVolatile)
      break;
  }
}

} // anonymous namespace

flattened_args flatten(py::handle obj) {
  ParsedArgs args;
  flatten_rec(obj.ptr(), args);
  // We might have put some Variable descriptors in desc before we discovered
  // the first volatile one, so we need to fix it now.
  if (args.is_volatile) {
    mark_all_volatile(args.desc);
  }
  return std::make_tuple(cast_handle_sequence<py::tuple>(args.vars), py::bytes(args.desc), args.is_volatile);
}

namespace {

using tuple_iterator = decltype(std::declval<py::tuple>().begin());

template<typename T>
py::object cast_sequence(std::vector<py::object> objs) {
  auto num_objs = objs.size();
  T sequence { num_objs };
  for (std::size_t i = 0; i < num_objs; ++i)
    sequence[i] = std::move(objs[i]);
  return sequence;
}

py::object unflatten_rec(tuple_iterator& var_it,
                         tuple_iterator& var_it_end,
                         std::string::iterator& desc_it) {
  char type = *desc_it++;
  if (type == D::TupleOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::TupleClose)
      objs.push_back(unflatten_rec(var_it, var_it_end, desc_it));
    ++desc_it;
    return cast_sequence<py::tuple>(objs);
  } else if (type == D::ListOpen) {
    std::vector<py::object> objs;
    while (*desc_it != D::ListClose)
      objs.push_back(unflatten_rec(var_it, var_it_end, desc_it));
    ++desc_it;
    return cast_sequence<py::list>(objs);
  } else {
    if (var_it == var_it_end)
      throw std::runtime_error("Not enough Variables given to unflatten");
    auto var = *var_it++;
    return py::reinterpret_borrow<py::object>(var);
  }
}

} // anonymous namespace

py::object unflatten(py::tuple vars, py::bytes descriptor) {
  // NB: We don't do correctness checking on descriptor.
  // It has to be a correct bytes object produced by unflatten.
  std::string desc = descriptor; // <sigh> we have to make a copy
  auto vars_it = vars.begin();
  auto vars_it_end = vars.end();
  auto desc_it = desc.begin();
  auto output = unflatten_rec(vars_it, vars_it_end, desc_it);
  if (vars_it != vars_it_end)
    throw std::runtime_error("Too many Variables given to unflatten");
  return output;
}

}}} // namespace torch::jit::python
