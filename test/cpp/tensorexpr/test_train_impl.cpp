#include "test/cpp/tensorexpr/test_train.h"
#include "test/cpp/tensorexpr/test_utils.h"
#include "torch/csrc/jit/tensorexpr/eval.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/ir_printer.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

#include <queue>
#include <set>

std::unordered_map<std::string, VMethod>& getMethodMap() {
  static std::unordered_map<std::string, VMethod> methods_;
  return methods_;
}

RegMethod::RegMethod(
    std::string name,
    VMethod::LowerFn lower,
    VMethod::GradFn grad,
    VMethod::ShapeFn shape,
    size_t num_out) {
  auto& method = getMethodMap()[name];
  method.name = name;
  method.num_outputs = num_out;
  method.lower = lower;
  method.grad = grad;
  method.shape = shape;
}

const VMethod& VMethod::get(const std::string& name) {
  auto method_iter = getMethodMap().find(name);
  TORCH_CHECK(
      method_iter != getMethodMap().end(),
      std::string("Couldn't find method for ") + name);
  auto& method = method_iter->second;
  return method;
}

std::vector<VTensor*> call(
    const std::string& name,
    const std::vector<VTensor*>& vs) {
  TORCH_CHECK(vs.size());
  auto* graph = vs[0]->graph;
  for (const auto& v : vs) {
    TORCH_CHECK(
        v,
        std::string(
            "Invalid input, perhaps an invalid index into the inputs of a grad function that calls ") +
            name);
    TORCH_CHECK(graph == v->graph);
  }
  const auto& method = VMethod::get(name);
  auto op = graph->create_op(name, vs, method.num_outputs);

  size_t index = 0;
  if (!method.shape) {
    std::stringstream ss;
    ss << "method \"" << method.name << "\" has no shape function";
    TORCH_CHECK(method.shape, ss.str());
  }
  const auto& shapes = method.shape(vs);
  for (auto& output : op->outputs) {
    output->shape = shapes[index];
    index++;
  }
  for (auto& v : vs) {
    v->consumers.emplace_back(op);
  }
  return op->outputs;
}

VTensor* grad(VTensor* y, VTensor* x, VTensor* j) {
  std::unordered_set<VTensor*> need_grad;
  need_grad.insert(y);
  std::unordered_set<VTensor*> no_grad;
  using Route = std::unordered_set<VTensor*>;
  std::queue<std::pair<VTensor*, Route>> q;
  // Iterate from X, as most nets work this way
  Route init_route;
  init_route.insert(x);
  q.push(std::make_pair(x, init_route));
  // q contains variables that haven't been
  // traversed.
  while (q.size()) {
    // Take a variable and try to find y,
    // "staying left" (first dep every time).
    //
    //   |
    //   v
    //  dep1  dep2
    //    \   /
    //     var
    //
    // Every time we "stay left," add the other consumers to q
    // If we find y -- add the whole route to need_grad
    // If we can't find y -- add the whole route to no_grad
    VTensor* var;
    std::unordered_set<VTensor*> route;
    std::tie(var, route) = q.front();
    q.pop();
    route.insert(var);

    while (var) {
      if (var == y) {
        need_grad.insert(route.begin(), route.end());
        break;
      }
      // add to q
      std::vector<VTensor*> next;
      for (auto dep : var->consumers) {
        auto i = 0;
        for (auto inp : dep->inputs) {
          if (inp == var) {
            for (const auto& out : dep->outputs) {
              next.emplace_back(out);
            }
          }
          i++;
        }
      }
      if (!next.size()) {
        no_grad.insert(route.begin(), route.end());
        break;
      }
      auto iter = next.begin();
      var = *iter;
      route.insert(var);
      iter++;
      while (iter != next.end()) {
        q.push(std::make_pair(*iter, route));
        iter++;
      }
    }
  }

  // Now calculate the gradients
  std::unordered_map<VTensor*, VTensor*> grad_map;
  // This is the input
  grad_map[y] = j;
  std::vector<VOp*> frontier{y->op};
  std::vector<VOp*> next_frontier;
  // This could be way more efficient
  std::set<VOp*> seen_ops{y->op};
  while (frontier.size()) {
    next_frontier.clear();
    for (const auto& op : frontier) {
      TORCH_CHECK(op, "Invalid operation found!");
      std::vector<VTensor*> grad_inputs;
      for (const auto& op_out : op->outputs) {
        TORCH_CHECK(op_out, "Invalid output");
        TORCH_CHECK(need_grad.find(op_out) != need_grad.end());
        auto grad_inp_iter = grad_map.find(op_out);
        TORCH_CHECK(grad_inp_iter != grad_map.end());
        grad_inputs.emplace_back(grad_inp_iter->second);
      }
      bool run_grad = false;
      for (const auto& input : op->inputs) {
        if (need_grad.find(input) != need_grad.end()) {
          run_grad = true;
          break;
        }
      }
      if (run_grad) {
        const auto& g = op->method->grad;
        if (!g) {
          std::stringstream ss;
          ss << "no known grad for method \"" << op->method->name << "\"";
          TORCH_CHECK(g, ss.str());
        }
        auto g_outs = g(op->inputs, grad_inputs);
        for (auto i = 0U; i < g_outs.size(); ++i) {
          auto input = op->inputs[i];
          if (need_grad.find(input) != need_grad.end()) {
            if (grad_map.find(input) != grad_map.end()) {
              grad_map[input] = call("add", {grad_map[input], g_outs[i]})[0];
            } else {
              grad_map[input] = g_outs[i];
            }
            if (input->op && seen_ops.find(input->op) == seen_ops.end()) {
              next_frontier.emplace_back(input->op);
              seen_ops.insert(input->op);
            }
          }
        }
      }
    }
    frontier = next_frontier;
  }
  TORCH_CHECK(grad_map.find(x) != grad_map.end());
  return grad_map[x];
}

VOp::VOp(
    const std::string& name,
    const std::vector<VTensor*>& inputs_,
    size_t num_outputs,
    VGraph* graph_)
    : inputs(inputs_), graph(graph_) {
  method = &VMethod::get(name);
  for (auto i = 0U; i < num_outputs; ++i) {
    outputs.emplace_back(graph->create_tensor({}));
    outputs.back()->op = this;
  }
}

using namespace torch::jit::tensorexpr;

std::vector<DimArg> get_vars(
    std::vector<std::string> dims,
    const std::map<std::string, torch::jit::tensorexpr::VarHandle>& vbindings) {
  std::vector<DimArg> vars;
  for (auto k : dims) {
    vars.emplace_back(vbindings.at(k));
  }
  if (vars.size() == 0) {
    vars.emplace_back(IntImm::make(1));
  }
  return vars;
}

REGISTER_METHOD(
    add,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 2);
      TORCH_CHECK(vinputs.at(0)->shape.size() == vinputs.at(1)->shape.size());
      auto vars = get_vars(vinputs.at(0)->shape, vbindings);
      Tensor* o = Compute("o", vars, [&](const VarHandle& i) {
        return inputs.at(0)->call(i) + inputs.at(1)->call(i);
      });
      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      return {ginputs[0], ginputs[0]};
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> {
      return {inputs[0]->shape};
    });

REGISTER_METHOD(
    sub,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 2);
      TORCH_CHECK(vinputs.at(0)->shape.size() == vinputs.at(1)->shape.size());
      auto vars = get_vars(vinputs.at(0)->shape, vbindings);
      Tensor* o = Compute("o", vars, [&](const VarHandle& i) {
        return inputs.at(0)->call(i) - inputs.at(1)->call(i);
      });
      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      return {ginputs[0], call("neg", {ginputs[0]})[0]};
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> {
      return {inputs[0]->shape};
    });

REGISTER_METHOD(
    neg,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 1);
      auto vars = get_vars(vinputs.at(0)->shape, vbindings);
      Tensor* o = Compute("o", vars, [&](const VarHandle& i) {
        return FloatImm::make(-1.0f) * inputs.at(0)->call(i);
      });
      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      return call("neg", {ginputs[0]});
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> {
      return {inputs[0]->shape};
    });

REGISTER_METHOD(
    mul,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 2);
      TORCH_CHECK(vinputs.at(0)->shape.size() == vinputs.at(1)->shape.size());
      auto vars = get_vars(vinputs.at(0)->shape, vbindings);
      Tensor* o = Compute("o", vars, [&](const VarHandle& i) {
        return inputs.at(0)->call(i) * inputs.at(1)->call(i);
      });
      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      return {
          call("mul", {ginputs[0], inputs[1]})[0],
          call("mul", {ginputs[0], inputs[0]})[0]};
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> {
      return {inputs[0]->shape};
    });

REGISTER_METHOD(
    div,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 2);
      TORCH_CHECK(vinputs.at(0)->shape.size() == vinputs.at(1)->shape.size());
      auto vars = get_vars(vinputs.at(0)->shape, vbindings);
      Tensor* o = Compute("o", vars, [&](const VarHandle& i) {
        return inputs.at(0)->call(i) / inputs.at(1)->call(i);
      });
      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      auto b_2 = call("mul", {inputs[1], inputs[1]})[0];
      auto a_div_b_2 = call("div", {inputs[0], b_2})[0];
      return {
          call("div", {ginputs[0], inputs[1]})[0],
          call("mul", {ginputs[0], call("neg", {a_div_b_2})[0]})[0]};
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> {
      return {inputs[0]->shape};
    });

REGISTER_METHOD(
    sum,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 1);
      auto vars = get_vars(vinputs.at(0)->shape, vbindings);
      Tensor* o = Reduce(
          "sum",
          {},
          Sum(),
          [=](const VarHandle& i) -> ExprHandle {
            return inputs.at(0)->call(i);
          },
          vars);

      // Tensor* o = Reduce("sum", {}, Sum(), inputs.at(0), vars);
      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      return call("broadcast", {ginputs[0], inputs[0]});
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> { return {{}}; });

REGISTER_METHOD(
    broadcast,
    [](const std::vector<Tensor*>& inputs,
       const std::vector<VTensor*>& vinputs,
       const std::map<std::string, torch::jit::tensorexpr::VarHandle>&
           vbindings) -> std::vector<Tensor*> {
      TORCH_CHECK(inputs.size() == 2);
      auto vars = get_vars(vinputs.at(1)->shape, vbindings);
      Tensor* o = Compute(
          "o", vars, [&](const VarHandle& i) { return inputs.at(0)->call(0); });

      return {o};
    },
    [](const std::vector<VTensor*>& inputs,
       const std::vector<VTensor*>& ginputs) -> std::vector<VTensor*> {
      return call("sum", {ginputs[0]});
    },
    [](const std::vector<VTensor*>& inputs)
        -> std::vector<std::vector<std::string>> {
      return {inputs[1]->shape};
    });

std::string dot(const VGraph& g) {
  std::stringstream ss;
  ss << "digraph {\n";
  for (const auto& op : g.vops) {
    auto name = op.method->name;
    auto id = reinterpret_cast<size_t>(&op);
    for (const auto& o : op.outputs) {
      ss << id << " -> " << reinterpret_cast<size_t>(o) << ";\n";
    }
    for (const auto& i : op.inputs) {
      ss << reinterpret_cast<size_t>(i) << " -> " << id << ";\n";
    }
    ss << id << "[shape=box;label=" << name << "];\n";
  }
  ss << "}\n";
  return ss.str();
}

std::tuple<
    Stmt*,
    std::map<const VTensor*, Placeholder>,
    std::map<const VTensor*, Tensor*>,
    std::map<std::string, VarHandle>>
to_tensorexpr(const VGraph& graph, std::vector<VTensor*> outputs) {
  std::map<size_t, std::string> unique_name_map;
  auto get_name = [&](size_t id) {
    if (!unique_name_map.count(id)) {
      std::stringstream ss;
      auto k = unique_name_map.size() + 1;
      while (k) {
        auto n = k % 26;
        ss << "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n - 1];
        k /= 26;
      }
      auto name = ss.str();
      unique_name_map[id] = name;
    }
    return unique_name_map.at(id);
  };

  auto topo = [](const VGraph& g) {
    std::set<const VOp*> nodes;
    for (auto& vop : g.vops) {
      nodes.insert(&vop);
    }
    std::set<const VOp*> temp;
    std::vector<const VOp*> order;
    std::function<void(const VOp*)> visit = [&](const VOp* n) -> void {
      if (!nodes.count(n)) {
        return;
      }
      if (temp.count(n)) {
        throw std::runtime_error("Cycle in constructed graph");
      }
      temp.insert(n);
      for (auto o : n->outputs) {
        for (auto c : o->consumers) {
          visit(c);
        }
      }
      temp.erase(n);
      nodes.erase(n);
      order.emplace(order.begin(), n);
    };
    while (nodes.size()) {
      visit(*nodes.begin());
    }
    return order;
  };

  std::map<const VTensor*, Placeholder> inputs;
  std::map<const VTensor*, Tensor*> bindings;
  std::map<std::string, torch::jit::tensorexpr::VarHandle> vbindings;

  for (const auto& t : graph.vtensors) {
    auto id = reinterpret_cast<size_t>(&t);
    for (auto d : t.shape) {
      if (!vbindings.count(d)) {
        VarHandle D(d, kInt);
        vbindings[d] = D;
      }
    }
    // input
    if (!t.op) {
      std::vector<DimArg> vars;
      std::vector<ExprHandle> exprs;
      for (auto k : t.shape) {
        vars.emplace_back(vbindings.at(k));
        exprs.emplace_back(vbindings.at(k));
      }
      if (vars.size() == 0) {
        vars.emplace_back(IntImm::make(1));
      }
      Placeholder inpB(BufHandle(get_name(id), exprs, kFloat));
      auto inpT =
          Compute("input" + get_name(id), vars, [&](const VarHandle& i) {
            return Load::make(BufHandle(inpB.data()), {i}, 1);
          });
      inputs.emplace(&t, inpB);
      bindings.emplace(&t, inpT);
    }
  }

  auto order = topo(graph);
  for (auto vop : order) {
    std::vector<Tensor*> inps;
    for (auto i : vop->inputs) {
      inps.emplace_back(bindings.at(i));
    }
    auto outs = vop->method->lower(inps, vop->inputs, vbindings);
    TORCH_CHECK(outs.size() == vop->outputs.size());
    for (auto i = 0U; i < outs.size(); ++i) {
      bindings[vop->outputs[i]] = outs[i];
    }
  }

  std::vector<Tensor*> toutputs;
  if (outputs.size() == 0) {
    for (auto& vtensor : graph.vtensors) {
      if (vtensor.consumers.size() == 0) {
        toutputs.emplace_back(bindings.at(&vtensor));
      }
    }
  } else {
    for (auto vtensor : outputs) {
      toutputs.emplace_back(bindings.at(vtensor));
    }
  }

  LoopNest l(toutputs);
  l.prepareForCodegen();
  Stmt* s = l.root_stmt();
  return std::make_tuple(s, inputs, bindings, vbindings);
}
