#include "torch/csrc/toffee/export.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/Exceptions.h"

#include <toffee/toffee.pb.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "torch/csrc/autograd/functions/convolution.h"

#include <fstream>

namespace torch { namespace jit {

std::string node_name(Node* n) {
  return std::to_string(n->unique());
}

// Exports a graph to ToffeeIR
std::string ExportGraph(std::unique_ptr<Graph>& g) {
  toffee::GraphProto p_g;
  torch::autograd::PrimSpecContext ctx;
  ctx.graph = &p_g;
  int temp_next_unique = 0;
  p_g.set_name("torch-jit-export");
  for (auto input : g->inputs()) {
    p_g.add_input(node_name(input));
  }
  for (auto output : g->outputs()) {
    p_g.add_output(node_name(output));
  }
  for (auto node : g->nodes()) {
    if (node->kind() == kSelect) {
      // No select nodes in ToffeeIR: instead we make use
      // of the select invariant
      continue;
    }
    auto generic_node = [&]() {
      toffee::NodeProto* p_n = p_g.add_node();
      for (auto input : node->inputs()) {
        p_n->add_input(node_name(input));
      }
      if (node->type()->kind() == TypeKind::MultiType) {
        for (auto u : node->uses()) {
          p_n->add_output(node_name(u.user));
        }
      } else {
        p_n->add_output(node_name(node));
      }
      return p_n;
    };
    // See https://fb.quip.com/TbPaAzijnd3e
    // TODO: Delete these
    IR_IF2(node, Add)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Add");
    IR_ELSEIF2(Mul)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Mul");
    IR_ELSEIF2(Negate)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Scale");
      toffee::AttributeProto* attr = p_n->add_attribute();
      attr->set_name("scale");
      attr->set_f(-1);
    IR_ELSEIF2(Sigmoid)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Sigmoid");
    IR_ELSEIF2(Tanh)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("TanH");
    IR_ELSEIF2(Constant)
      throw std::runtime_error("Constant not supported yet");
    IR_ELSEIF2(Return)
      JIT_ASSERT(0);
    IR_ELSEIF2(Select)
      JIT_ASSERT(0);
    IR_ELSEIF2(Param)
      JIT_ASSERT(0);
    IR_ELSEIF2(FusionGroup)
      throw std::runtime_error("FusionGroup not supported.  Try exporting before fusing");
    IR_ELSEIF(CppOp)
      if (auto fn = std::dynamic_pointer_cast<autograd::HasPrimSpec>(value->fn)) {
        node_list outputs;
        if (node->type()->kind() == TypeKind::MultiType) {
          for (auto u : node->uses()) {
            outputs.push_back(u.user);
          }
        } else {
          outputs.push_back(node);
        }
        fn->primspec(&ctx, node->inputs(), outputs);
      } else {
        throw std::runtime_error("CppOp doesn't define primspec " + value->name());
      }
    IR_ELSEIF(PythonOp)
      // NB: All inplace designations are dropped

      toffee::NodeProto* p_n = generic_node();

      if (!PyObject_HasAttrString(value->pyobj.get(), "primspec")) {
        throw std::runtime_error("PythonOp doesn't define primspec " + value->name());
      }
      THPObjectPtr primspec_fn(PyObject_GetAttrString(value->pyobj.get(), "primspec"));
      if (!primspec_fn) throw python_error();
      THPObjectPtr py_primspec_args(PyTuple_New(value->cconv.size()));
      // Symbolically represent tensors as longs for now.  Hypothetically,
      // we could pass a Variable in instead, which could allow for
      // "modifications" to the inputs before they get glommed into the
      // Toffee IR.

      // TODO: This is copy-pasted from jit_closure.cpp
      auto node_it = node->inputs().begin();
      auto scalar_it = value->scalar_args.begin();
      Py_ssize_t input_nr = 0;

      for (auto arg_type : value->cconv) {
        PyObject *obj;
        if (arg_type == 's') {
          if (scalar_it == value->scalar_args.end())
            throw std::runtime_error("expected too many scalar args");
          obj = (scalar_it++)->get();
          Py_INCREF(obj);
        } else if (arg_type == 't') {
          if (node_it == node->inputs().end())
            throw std::runtime_error("expected too many inputs");
          // TODO: Send in something more reasonable here
          obj = PyLong_FromLong(node_it - node->inputs().begin());
          node_it++;
        } else {
          throw std::runtime_error("unexpected calling convention");
        }
        PyTuple_SET_ITEM(py_primspec_args.get(), input_nr++, obj);
      }
      THPObjectPtr raw_output(PyObject_CallObject(primspec_fn, py_primspec_args));
      if (!raw_output) {
        throw python_error();
      }
      if (raw_output == Py_None) {
        throw std::runtime_error("PythonOp's primspec returned None, indicating conversion not supported " + value->name());
      }
      if (!PyDict_Check(raw_output.get())) throw std::runtime_error("primspec did not return a dict");

      PyObject* py_op_type = PyDict_GetItemString(raw_output.get(), "name");
      if (!py_op_type) throw std::runtime_error("primspec missing name key");
      if (!THPUtils_checkString(py_op_type)) throw std::runtime_error("primspec returned a non-string name");
      p_n->set_op_type(THPUtils_unpackString(py_op_type));

      PyObject* py_inputs = PyDict_GetItemString(raw_output.get(), "inputs");
      if (py_inputs) {
        if (!PyTuple_Check(py_inputs)) throw std::runtime_error("primspec returned non-tuple inputs");

        p_n->clear_input();
        Py_ssize_t num_inputs = PyTuple_GET_SIZE(py_inputs);
        for (int i = 0; i < num_inputs; i++) {
          // TODO: better error message when at is out of bounds
          p_n->add_input(node_name(node->inputs().at(PyLong_AsLong(PyTuple_GET_ITEM(py_inputs, i)))));
        }
      }
      // otherwise, default to preserving the inputs directly

      PyObject* py_attrs = PyDict_GetItemString(raw_output.get(), "attrs");
      if (py_attrs) {
        if (!PyDict_Check(py_attrs)) throw std::runtime_error("primspec did not return a dict attrs entry");
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(py_attrs, &pos, &key, &value)) {
          if (!THPUtils_checkString(key)) throw std::runtime_error("non-string key in attrs from primspec");
          if (THPUtils_unpackString(key) == "_outputs") {
            // This is a special hack to handle cases when PyTorch supports
            // more outputs than Toffee IR does OR the outputs are in the wrong
            // order.
            // TODO: if we drop an output that is used later, we must FAIL THE
            // EXPORT. Right now I believe we just create a malformed ToffeeIR
            // spec.
            if (node->type()->kind() != TypeKind::MultiType) {
              // NB: Actually, this can never happen because PythonOp is always
              // multi-return (lol)
              throw std::runtime_error("can't use _outputs for a function that doesn't return multiple things");
            }
            p_n->clear_output();
            if (!PyTuple_Check(value)) throw std::runtime_error("_outputs was not tuple");
            Py_ssize_t num_toffee_outputs = PyTuple_GET_SIZE(value);
            for (Py_ssize_t i = 0; i < num_toffee_outputs; i++) {
              PyObject *ix = PyTuple_GET_ITEM(value, i);
              if (!THPUtils_checkLong(ix)) throw std::runtime_error("_outputs entry was not numeric index");
              long l = THPUtils_unpackLong(ix);
              if (l >= 0) {
                p_n->add_output(node_name(node->uses().at(l).user));
              } else {
                // This is an extra Toffee IR output which PyTorch doesn't have.
                // What we will do is just add a dummy output to work around
                // this.
                p_n->add_output("tmp" + std::to_string(temp_next_unique++));
              }
            }
            continue;
          }
          toffee::AttributeProto* attr = p_n->add_attribute();
          attr->set_name(THPUtils_unpackString(key));
          if (THPUtils_checkLong(value)) {
            attr->set_i(THPUtils_unpackLong(value));
          } else if (THPUtils_checkDouble(value)) { // order matters, since all longs are doubles
            attr->set_f(THPUtils_unpackDouble(value)); // TODO: precision?!
          } else if (THPUtils_checkString(value)) {
            // TODO: binary data?!
            attr->set_s(THPUtils_unpackString(value));
          } else if (PyTuple_Check(value)) {
            Py_ssize_t num_value_items = PyTuple_GET_SIZE(value);
            int seen_int = 0;
            int seen_float = 0;
            int seen_string = 0;
            for (Py_ssize_t i = 0; i < num_value_items; i++) {
              PyObject *elem = PyTuple_GET_ITEM(value, i);
              if (THPUtils_checkLong(elem)) {
                attr->add_ints(THPUtils_unpackLong(elem));
                seen_int = 1;
              } else if (THPUtils_checkDouble(elem)) { // order matters, since all longs are doubles
                attr->add_floats(THPUtils_unpackDouble(elem)); // TODO: precision?!
                seen_float = 1;
              } else if (THPUtils_checkString(elem)) {
                // TODO: binary data?!
                attr->add_strings(THPUtils_unpackString(elem));
                seen_string = 1;
              } else {
                // TODO: better message
                throw std::runtime_error("unrecognized type of tuple entry in primspec attrs");
              }
              // TODO: Tensor constants?
            }
            if (seen_int + seen_float + seen_string > 1) {
              throw std::runtime_error("cannot have multiple types in attribute tuple");
            }
          }
        }
      }
    IR_ELSE()
      throw std::runtime_error("Not supported");
    IR_END()
  }
  std::string s;
  google::protobuf::TextFormat::PrintToString(p_g, &s);
  return s; // RVO
}

}}
