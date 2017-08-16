#include "torch/csrc/jit/graph_exporter.h"
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
  // TODO: This needs to be exported from Python
  // int temp_next_unique = 0;
  p_g.set_name("torch-jit-export");
  for (auto input : g->inputs()) {
    p_g.add_input(node_name(input));
  }
  for (auto output : g->outputs()) {
    p_g.add_output(node_name(output));
  }
  for (auto node : g->nodes()) {
    if (node->kind() == NodeKind::Select) {
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
    IR_IF(node, Add)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Add");
    IR_ELSEIF(Mul)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Mul");
    IR_ELSEIF(Negate)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Scale");
      toffee::AttributeProto* attr = p_n->add_attribute();
      attr->set_name("scale");
      attr->set_f(-1);
    IR_ELSEIF(Sigmoid)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("Sigmoid");
    IR_ELSEIF(Tanh)
      toffee::NodeProto* p_n = generic_node();
      p_n->set_op_type("TanH");
    IR_ELSEIF(Constant)
      throw std::runtime_error("Constant not supported yet");
    IR_ELSEIF(Return)
      JIT_ASSERT(0);
    IR_ELSEIF(Select)
      JIT_ASSERT(0);
    IR_ELSEIF(Param)
      JIT_ASSERT(0);
    IR_ELSEIF(FusionGroup)
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
      int num_args = node->inputs().size();
      THPObjectPtr py_primspec_args(PyTuple_New(num_args));
      // Symbolically represent tensors as longs for now.  Hypothetically,
      // we could pass a Variable in instead, which could allow for
      // "modifications" to the inputs before they get glommed into the
      // Toffee IR.
      for (int i = 0; i < num_args; ++i) {
        PyTuple_SET_ITEM(py_primspec_args.get(), i, PyLong_FromLong(i)); // steals!
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
          toffee::AttributeProto* attr = p_n->add_attribute();
          if (!THPUtils_checkString(key)) throw std::runtime_error("non-string key in attrs from primspec");
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
              if (THPUtils_checkLong(value)) {
                attr->add_ints(THPUtils_unpackLong(value));
                seen_int = 1;
              } else if (THPUtils_checkDouble(value)) { // order matters, since all longs are doubles
                attr->add_floats(THPUtils_unpackDouble(value)); // TODO: precision?!
                seen_float = 1;
              } else if (THPUtils_checkString(value)) {
                // TODO: binary data?!
                attr->add_strings(THPUtils_unpackString(value));
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

      /*
        // This is the old code for doing conversions; it should be ported into
        // primspec.

        // I have purposely NOT added these operators to the IR and
        // then transformed them in init_pass, because I don't think
        // we should be in the business of adding every operator
        // to the subclass hierarchy.  See:
        // https://github.com/ezyang/pytorch/issues/36
        //
        // There is something pretty irritating here: many Python classes
        // have default arguments, but when we trace a Python operator,
        // we don't have any visibility into the defaults that would have
        // been picked by the function itself.  See:
        // https://github.com/ezyang/pytorch/issues/40
        if (value->name() == "Threshold" &&
            // Caffe2 backend only supports ReLU, not Threshold
            value->scalar_args.size() >= 2 &&
            THPUtils_unpackLong(value->scalar_args[0]) == 0 && // threshold
            THPUtils_unpackLong(value->scalar_args[1]) == 0) { // value
          p_n->set_op_type("Relu");
        } else if (value->name() == "MaxPool2d" &&
            // TODO: This is very fragile!
            value->scalar_args.size() == 5 &&
            !PyObject_IsTrue(value->scalar_args[4])) {
          // MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
          // e.g., MaxPool2d(3, 2, 0, 1, False) from AlexNet
          // TODO: put these attributes in the standard
          p_n->set_op_type("MaxPool");
          toffee::AttributeProto* attr;
          attr = p_n->add_attribute();
          attr->set_name("kernel");
          attr->set_i(THPUtils_unpackLong(value->scalar_args[0]));

          attr = p_n->add_attribute();
          attr->set_name("stride");
          attr->set_i(THPUtils_unpackLong(value->scalar_args[1]));

          attr = p_n->add_attribute();
          attr->set_name("pad");
          attr->set_i(THPUtils_unpackLong(value->scalar_args[2]));

          attr = p_n->add_attribute();
          attr->set_name("dilation");
          attr->set_i(THPUtils_unpackLong(value->scalar_args[3]));

          // Toffee returns only one arg
          p_n->clear_output();
          p_n->add_output(node_name(node->uses().at(0).user));
        } else if (value->name() == "View") {
          p_n->set_op_type("Reshape");
          toffee::AttributeProto* attr;

          attr = p_n->add_attribute();
          attr->set_name("shape");
          auto& t = value->scalar_args.at(0);
          for (int i = 0, c = PyTuple_Size(t); i < c; i++) {
            attr->add_ints(THPUtils_unpackLong(PyTuple_GET_ITEM(t.get(), i)));
          }

          p_n->add_output("tmp" + std::to_string(temp_next_unique++));
        } else if (value->name() == "Transpose") {
          p_n->set_op_type("Transpose");
          toffee::AttributeProto* attr;

          attr = p_n->add_attribute();
          attr->set_name("axes");
          for (auto& scalar_arg : value->scalar_args) {
            attr->add_ints(THPUtils_unpackLong(scalar_arg.get()));
          }
        } else if (value->name() == "Dropout") {
          // Dropout(0.5, True, False)
          // p, training, inplace (inplace punted)
          p_n->set_op_type("Dropout");
          toffee::AttributeProto* attr;

          attr = p_n->add_attribute();
          attr->set_name("ratio");
          attr->set_f(THPUtils_unpackDouble(value->scalar_args.at(0))); // NB: precision loss

          attr = p_n->add_attribute();
          attr->set_name("is_test");
          // NB: PyTorch's boolean is is_training, which is the inverted sense
          attr->set_i(!PyObject_IsTrue(value->scalar_args.at(1)));

          p_n->add_output("tmp" + std::to_string(temp_next_unique++));
        } else if (value->name() == "Addmm" &&
                   value->scalar_args.size() >= 2 &&
                   // NB: FC doesn't support weights so we have to
                   // exclude these addmm's
                   // TODO: Double check this does the right thing
                   // in terms of numeric precision
                   THPUtils_unpackDouble(value->scalar_args[0]) == 1.0 &&
                   THPUtils_unpackDouble(value->scalar_args[1]) == 1.0) {
          // TODO: handle cases when FC doesn't work.  Addmm supports a 2D bias
          // and FC does not.  We need to detect this case and do something
          // different.
          p_n->set_op_type("FC");
          // Redo the inputs: bias is first for PyTorch and last for Caffe2
          p_n->clear_input();
          p_n->add_input(node_name(node->inputs().at(1)));
          p_n->add_input(node_name(node->inputs().at(2)));
          p_n->add_input(node_name(node->inputs().at(0)));

        } else {
          throw std::runtime_error("PythonOp not supported " + value->name());
        }

      */
    IR_ELSE()
      throw std::runtime_error("Not supported");
    IR_END()
  }
  std::string s;
  google::protobuf::TextFormat::PrintToString(p_g, &s);
  return s; // RVO
}

}}
