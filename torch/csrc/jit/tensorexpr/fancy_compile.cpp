#include <torch/csrc/jit/tensorexpr/fancy_compile.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_opt_limit.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <torch/csrc/jit/tensorexpr/loopnest.h>

#define N 100

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {

static bool isOne(ExprHandle e) {
  auto const& n = e.AsNode<IntImm>();
  if (!n) {
    return false;
  }
  return n->value() == 1;
}

std::vector<ExprHandle> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  auto at = a.rbegin();
  auto bt = b.rbegin();
  std::vector<ExprHandle> ret;
  while (at != a.rend() || bt != b.rend()) {
    if (at == a.rend()) {
      ret.push_back(*bt++);
      continue;
    }
    if (bt == b.rend()) {
      ret.push_back(*at++);
      continue;
    }
    // TODO: if neither *at nor *bt is 1, ensure they are identical
    // expressions.  Nb: `==` doesn't work since that simply produces a new
    // ExprHandle.
    ExprHandle dim = *at;
    if (isOne(*at)) {
      if (!isOne(*bt)) {
        dim = *bt;
      }
    }
    ret.push_back(dim);
    at++;
    bt++;
  }
  std::reverse(ret.begin(), ret.end());
  return ret;
}

std::vector<ExprHandle> broadcastShapes(
    std::vector<std::vector<ExprHandle>> shapes) {
  size_t n = shapes.size();
  if (n == 1) {
    return shapes[0];
  }
  auto res1 = broadcastShapes(shapes[n - 2], shapes[n - 1]);
  shapes[n - 2] = res1;
  shapes.pop_back();
  auto res2 = broadcastShapes(shapes);
  return res2;
}

Tensor* computeNaryOp(
    const std::string& name_hint,
    std::vector<const Buf*> ops,
    std::vector<std::vector<ExprHandle>> shapes,
    const std::function<ExprHandle(const std::vector<ExprHandle>&)>& body) {
  auto out_shape = broadcastShapes(shapes);

  Tensor* r = Compute(
      name_hint,
      c10::fmap<DimArg>(out_shape),
      [ops, body](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs;
        for (auto b : ops) {
          for (int i = 0; i < b->ndim(); i++) {
            if (isOne(ExprHandle(b->dim(i)))) {
              indices[i] = ExprHandle(0);
            } else {
              indices[i] = axes[i];
            }
          }

          inputs.push_back(Load::make(BufHandle(b), indices, IntImm::make(1)));
        }
        return body(inputs);
      });
  return r;
}

void print_vector(const std::vector<ExprHandle>& v) {
  int i = 0;
  std::cerr << "[";
  for (const ExprHandle& e : v) {
    if (i++ > 0) {
      std::cerr << ", ";
    }
    std::cerr << e;
  }
  std::cerr << "]\n";
}

struct ConstantDescr {
  const Buf* buf;
  void* ptr;
};

class AOT_NNC_Compiler {
 public:
  AOT_NNC_Compiler(std::shared_ptr<Graph>& g)
      : original_graph(g), optimized_graph(g) {}

  void compile_for_sizes(const std::vector<int>& sizes) {
    root_stmt = new tensorexpr::Block({});
    optimized_graph = original_graph->copy();

    std::vector<ExprHandle> input_shape;
    for (auto e : sizes) {
      input_shape.push_back(IntImm::make(e));
    }
    auto v = optimized_graph->inputs()[1];
    size_map[v] = input_shape;
    rank_map[v] = input_shape.size();
    const Buf* input_buf =
        new Buf("input", ExprHandleVectorToExprVector(input_shape), kFloat);
    buf_map[v] = input_buf;

    process_block(optimized_graph->block());

    ConstantPropagationImmutableTypes(optimized_graph);
    EliminateDeadCode(optimized_graph->block());
    optimized_graph->dump();

    std::vector<CodeGen::BufferArg> buf_args;
    buf_args.emplace_back(BufHandle(buf_map.at(optimized_graph->inputs()[1])));
    buf_args.emplace_back(BufHandle(buf_map.at(optimized_graph->outputs()[0])));
    std::unordered_set<const Buf*> const_bufs;
    for (const auto& c : constants) {
      buf_args.push_back(BufHandle(c.buf));
      const_bufs.insert(c.buf);
    }
    const Buf* output_buf;
    output_buf = buf_map.at(optimized_graph->outputs()[0]);
    std::unordered_set<const Buf*> interm_bufs;
    auto bufs = NodeFinder<Buf>::find(root_stmt);
    for (auto b : bufs) {
      if (b == input_buf || b == output_buf || const_bufs.count(b))
        continue;
      interm_bufs.insert(b);
    }

    LoopNest l(root_stmt, {output_buf});
    std::cerr << "Original stmt!\n";
    std::cerr << *l.root_stmt() << "\n";
    l.inlineIntermediateBufs(true);
    l.simplify();
    l.prepareForCodegen();
    l.simplify();
    auto preallocated = preallocateTemps(l.root_stmt());
    for (const auto& c : preallocated) {
      buf_args.push_back(BufHandle(c.second));
    }

    std::cerr << "Prepared for codegen!\n";
    std::cerr << *l.root_stmt() << "\n";
    //     codegen_ = CreateCodeGen("simple_ir_eval", l.root_stmt(), buf_args,
    //     c10::kCPU, "qqq");
    codegen_ = CreateCodeGen(
        "llvm_codegen", l.root_stmt(), buf_args, c10::kCPU, "qqq");
    std::cerr << "Codegen created!\n";

    call_args.emplace_back(nullptr);
    call_args.emplace_back(nullptr);
    for (const auto& c : constants) {
      call_args.push_back(c.ptr);
    }
    std::cerr << buf_args.size() << " - " << call_args.size() << "\n";

    int64_t total_size = 1;
    for (auto s : size_map.at(optimized_graph->outputs()[0])) {
      int dim_size = s.AsNode<IntImm>()->value();
      out_sizes.push_back(dim_size);
      total_size *= dim_size;
    }
    output_data = (void*)(new double[total_size]);
    input_tensor = at::ones({1, 3, 224, 224});

    original_graph->eraseInput(0);
    optimized_graph->eraseInput(0);
  }

  at::Tensor call_with_nnc(at::Tensor x) {
    std::cerr << "NNC\n";
    std::chrono::steady_clock::time_point begin, end;
    at::Tensor output_tensor;
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
      call_args[0] = x.data_ptr();
      call_args[1] = output_data;
      codegen_->call(call_args);

      auto options = at::TensorOptions()
                         .dtype(at::kFloat)
                         .layout(at::kStrided)
                         .device(at::kCPU)
                         .requires_grad(false);

      output_tensor = at::from_blob(output_data, out_sizes, options);
    }
    end = std::chrono::steady_clock::now();
    std::cerr << "Result:\n";
    std::cerr << output_tensor.sizes() << "\n";
    std::cerr << at::sum(at::abs(output_tensor)) << "\n";
    //     std::cerr << output_tensor << "\n";
    auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();
    std::cerr << "NNC Time: " << (float)dur / N << " ms/iter\n";
    return output_tensor;
  }

  void call_with_jit() {}

  at::Tensor call_with_jit_optimized_graph(at::Tensor x) {
    std::cerr << "JIT\n";
    //     optimized_graph->dump();
    std::chrono::steady_clock::time_point begin, end;
    at::Tensor output_tensor;
    begin = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
      Stack stack;
      auto input_tensor = x;
      stack.push_back(x);
      Code code(optimized_graph, "qqq");
      InterpreterState(code).run(stack);
      output_tensor = stack[0].toTensor();
    }
    end = std::chrono::steady_clock::now();
    std::cerr << "Result:\n";
    std::cerr << output_tensor.sizes() << "\n";
    std::cerr << at::sum(at::abs(output_tensor)) << "\n";
    //     std::cerr << output_tensor << "\n";
    auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();
    std::cerr << "JIT Time: " << (float)dur / N << " ms/iter\n";
    return output_tensor;
  }

 private:
  std::vector<ExprHandle> sizesForValue(torch::jit::Value* v);
  void handleConstant(Node* n);
  void process_block(torch::jit::Block* bb);
  Stmt* generate_conv(
      torch::jit::Value* v,
      const Buf* inp,
      const Buf* w,
      const Buf* b,
      int sH,
      int sW,
      int pH,
      int pW,
      int dH,
      int dW,
      int groups);
  Tensor* generate_matmul(const Buf* a, const Buf* b);
  Tensor* generate_mm(const Buf* a, const Buf* b);
  void printSizes(torch::jit::Value* v);
  void* preallocateBuf(Stmt* s, const Buf* b);
  std::vector<std::pair<void*, const Buf*>> preallocateTemps(Stmt* s);

  std::shared_ptr<Graph> original_graph;
  std::shared_ptr<Graph> optimized_graph;

  std::unordered_map<torch::jit::Value*, IValue> ival_map;
  std::unordered_map<torch::jit::Value*, int> rank_map;
  std::unordered_map<torch::jit::Value*, std::vector<ExprHandle>> size_map;
  std::unordered_map<torch::jit::Value*, const Buf*> buf_map;

  tensorexpr::Block* root_stmt;
  KernelScope kernel_scope;

  std::vector<ConstantDescr> constants;
  std::vector<CodeGen::CallArg> call_args;
  at::Tensor output_tensor, input_tensor;
  void* output_data;
  std::vector<int64_t> out_sizes;

  std::unique_ptr<CodeGen> codegen_;
  std::vector<at::Tensor> unpacked_constants;
  std::vector<void*> preallocated_temps;
};

void* AOT_NNC_Compiler::preallocateBuf(Stmt* s, const Buf* b) {
  std::cerr << "Preallocate: " << *b->base_handle() << "\n";
  auto sizes = ExprVectorToExprHandleVector(b->dims());
  print_vector(sizes);
  ExprHandle flat_size = IntImm::make(1);
  for (auto s : sizes) {
    flat_size = flat_size * s;
  }
  flat_size = IRSimplifier::simplify(flat_size);

  auto const& n = flat_size.AsNode<IntImm>();
  if (!n) {
    assert(false);
    return nullptr;
  }
  int64_t alloc_size = b->dtype().byte_size() * n->value();
  std::cerr << "Total size: " << alloc_size << "\n";

  void* ptr = malloc(alloc_size);
  preallocated_temps.push_back(ptr);
  constants.push_back({b, ptr});

  Stmt *alloc_stmt = nullptr, *free_stmt = nullptr;
  for (auto ss : dynamic_cast<Block*>(s)->stmts()) {
    if (auto a = dynamic_cast<Allocate*>(ss)) {
      if (a->buffer_var() == b->base_handle()) {
        alloc_stmt = ss;
      }
    }
    if (auto f = dynamic_cast<Free*>(ss)) {
      if (f->buffer_var() == b->base_handle()) {
        free_stmt = ss;
      }
    }
  }
  dynamic_cast<Block*>(s)->remove_stmt(alloc_stmt);
  dynamic_cast<Block*>(s)->remove_stmt(free_stmt);
  return ptr;
}

std::vector<std::pair<void*, const Buf*>> AOT_NNC_Compiler::preallocateTemps(
    Stmt* s) {
  std::vector<std::pair<void*, const Buf*>> ret;
  std::cerr << "preallocate temps\n";
  auto allocs = NodeFinder<Allocate>::find(s);
  std::cerr << allocs.size() << " allocs found\n";
  std::unordered_map<const Var*, const Buf*> var_buf_map;
  auto bufs = NodeFinder<Buf>::find(s);
  for (auto b : bufs) {
    var_buf_map[b->base_handle()] = b;
  }
  for (auto a : allocs) {
    const Buf* buf = var_buf_map.at(dynamic_cast<Allocate*>(a)->buffer_var());
    void* ptr = preallocateBuf(s, buf);
    ret.push_back(std::make_pair(ptr, buf));
  }
  return ret;
}

void AOT_NNC_Compiler::printSizes(torch::jit::Value* v) {
  std::cerr << "%" << v->debugName() << ": ";
  print_vector(size_map[v]);
}

Stmt* AOT_NNC_Compiler::generate_conv(
    torch::jit::Value* v,
    const Buf* inp,
    const Buf* w,
    const Buf* b,
    int sH,
    int sW,
    int pH,
    int pW,
    int dH,
    int dW,
    int groups) {
  Stmt* s = nullptr;

  BufHandle ResultBuf("conv", size_map.at(v), kFloat);
  buf_map[v] = ResultBuf.node();
  s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_conv2d",
      {BufHandle(inp), BufHandle(w), BufHandle(b)},
      {sH, sW, pH, pW, dH, dW, groups});

  return s;
}

Tensor* AOT_NNC_Compiler::generate_matmul(const Buf* a, const Buf* b) {
  Stmt* s = nullptr;
  Tensor* t = nullptr;
  const Buf* out = nullptr;

  auto size_a = ExprVectorToExprHandleVector(a->dims());
  auto size_b = ExprVectorToExprHandleVector(b->dims());
  const IntImm* total_size = dynamic_cast<const IntImm*>(
      IRSimplifier::simplify((size_a[0] * size_a[1] * size_b[1])).node());

  if (total_size && total_size->value() < 3000) {
    t = Reduce(
        "nnc_matmul",
        {{size_a[0], "M"}, {size_b[1], "N"}},
        Sum(),
        [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
          BufHandle ah(a);
          BufHandle bh(b);
          return Load::make(ah, {m, k}, 1) * Load::make(bh, {k, n}, 1);
        },
        {{size_a[1], "K"}});
  } else {
    BufHandle ResultBuf(
        "matmul", {ExprHandle(size_a[0]), ExprHandle(size_b[1])}, kFloat);
    s = ExternalCall::make(
        ResultBuf, "nnc_aten_matmul", {BufHandle(a), BufHandle(b)}, {});
    out = ResultBuf.node();
    t = new Tensor(out, s);
  }

  return t;
}
Tensor* AOT_NNC_Compiler::generate_mm(const Buf* a, const Buf* b) {
  Stmt* s = nullptr;
  Tensor* t = nullptr;
  const Buf* out = nullptr;

  auto size_a = ExprVectorToExprHandleVector(a->dims());
  auto size_b = ExprVectorToExprHandleVector(b->dims());
  const IntImm* total_size = dynamic_cast<const IntImm*>(
      IRSimplifier::simplify((size_a[0] * size_a[1] * size_b[1])).node());

  if (false && total_size && total_size->value() < 3000) {
    t = Reduce(
        "nnc_mm",
        {{size_a[0], "M"}, {size_b[1], "N"}},
        Sum(),
        [&](const ExprHandle& m, const ExprHandle& n, const ExprHandle& k) {
          BufHandle ah(a);
          BufHandle bh(b);
          return Load::make(ah, {m, k}, 1) * Load::make(bh, {k, n}, 1);
        },
        {{size_a[1], "K"}});
  } else {
    BufHandle ResultBuf(
        "mm", {ExprHandle(size_a[0]), ExprHandle(size_b[1])}, kFloat);
    s = ExternalCall::make(
        ResultBuf, "nnc_aten_mm", {BufHandle(a), BufHandle(b)}, {});
    out = ResultBuf.node();
    t = new Tensor(out, s);
  }

  return t;
}

std::vector<ExprHandle> AOT_NNC_Compiler::sizesForValue(torch::jit::Value* v) {
  return size_map[v];
}

void AOT_NNC_Compiler::handleConstant(Node* n) {
  torch::jit::Value* v = n->output();

  ival_map[v] = *toIValue(v);

  if (v->type()->cast<TensorType>()) {
    const auto& tt = v->type()->expect<TensorType>();
    const auto sizes = *tt->sizes().concrete_sizes();
    rank_map[v] = sizes.size();
    std::cerr << "%" << v->debugName() << ": ";
    std::vector<ExprHandle> te_sizes;
    int i = 0;
    for (auto s : sizes) {
      if (i++ > 0) {
        std::cerr << ", ";
      }
      std::cerr << s;
      te_sizes.push_back(IntImm::make(s));
    }
    std::cerr << "; rank = " << sizes.size() << "\n";
    size_map[v] = te_sizes;
    buf_map[v] =
        new Buf(v->debugName(), ExprHandleVectorToExprVector(te_sizes), kFloat);
    auto const_tensor = toIValue(v)->toTensor();
    if (!const_tensor.is_contiguous()) {
      unpacked_constants.push_back(const_tensor.clone().contiguous());
      const_tensor = unpacked_constants.back();
    }
    constants.push_back({buf_map.at(v), const_tensor.data_ptr()});
  } else {
    std::cerr << "%" << v->debugName() << ": ";
    std::cerr << "ivalue " << *toIValue(v) << "\n";
  }
}

void AOT_NNC_Compiler::process_block(torch::jit::Block* bb) {
  Graph* g = bb->owningGraph();
  for (Node* n : bb->nodes()) {
    std::string s = n->kind().toUnqualString();
    bool handled = false;

    auto try_run_results = runNodeIfInputsAreConstant(n, true);
    if (try_run_results) {
      WithInsertPoint p(n);
      for (int i = 0; i < n->outputs().size(); i++) {
        auto iv = (*try_run_results)[i];
        auto new_v = g->insertConstant(iv);
        handleConstant(new_v->node());
        n->output(i)->replaceAllUsesAfterNodeWith(n, new_v);
      }
      handled = true;
    } else if (n->kind() == prim::Constant) {
      handleConstant(n);
      handled = true;
    } else if (n->kind() == prim::If) {
      bool cond = false;
      if (ival_map.count(n->input())) {
        cond = ival_map.at(n->input()).toBool();
      } else {
        cond = toIValue(n->input())->toBool();
      }
      if (cond) {
        process_block(n->blocks()[0]);
      } else {
        process_block(n->blocks()[1]);
      }
      handled = true;
    } else if (n->kind() == aten::addmm) {
      const auto& input1_shape = size_map.at(n->input(1));
      const auto& input2_shape = size_map.at(n->input(2));
      std::vector<ExprHandle> shape = {input1_shape[0], input2_shape[1]};
      size_map[n->output()] = shape;
      printSizes(n->output());

      Stmt* s = nullptr;
      auto v = n->output();
      BufHandle ResultBuf("addmm", size_map.at(v), kFloat);
      buf_map[v] = ResultBuf.node();
      s = ExternalCall::make(
          ResultBuf,
          "nnc_aten_addmm",
          {BufHandle(buf_map.at(n->input(0))),
           BufHandle(buf_map.at(n->input(1))),
           BufHandle(buf_map.at(n->input(2)))},
          {1, 1});

      std::cerr << "Addmm stmt:\n[QQ] " << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

      std::cerr << "[QQ] Input0 shape: ";
      print_vector(size_map.at(n->input(0)));
      std::cerr << "[QQ] Input1 shape: ";
      print_vector(input1_shape);
      std::cerr << "[QQ] Input2 shape: ";
      print_vector(input2_shape);

    } else if (n->kind() == aten::matmul) {
      const auto& input1_shape = size_map.at(n->input(0));
      const auto& input2_shape = size_map.at(n->input(1));
      std::vector<ExprHandle> shape = {input1_shape[0], input2_shape[1]};
      size_map[n->output()] = shape;
      printSizes(n->output());

      Tensor* t =
          generate_matmul(buf_map.at(n->input(0)), buf_map.at(n->input(1)));

      Stmt* s = nullptr;
      s = t->stmt();
      auto v = n->output();
      buf_map[v] = t->buf();

      std::cerr << "matmul stmt:\n[QQ] " << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

      std::cerr << "[QQ] Input1 shape: ";
      print_vector(input1_shape);
      std::cerr << "[QQ] Input2 shape: ";
      print_vector(input2_shape);
    } else if (n->kind() == aten::mm) {
      const auto& input1_shape = size_map.at(n->input(0));
      const auto& input2_shape = size_map.at(n->input(1));
      std::vector<ExprHandle> shape = {input1_shape[0], input2_shape[1]};
      size_map[n->output()] = shape;
      printSizes(n->output());

      Tensor* t =
          generate_mm(buf_map.at(n->input(0)), buf_map.at(n->input(1)));

      Stmt* s = nullptr;
      s = t->stmt();
      auto v = n->output();
      buf_map[v] = t->buf();

      std::cerr << "mm stmt:\n[QQ] " << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

      std::cerr << "[QQ] Input1 shape: ";
      print_vector(input1_shape);
      std::cerr << "[QQ] Input2 shape: ";
      print_vector(input2_shape);
    } else if (n->kind() == aten::adaptive_avg_pool2d) {
      auto sizes_arg = toIValue(n->input(1))->toIntVector();
      std::vector<ExprHandle> shape;
      const auto& input_shape = size_map.at(n->input(0));
      for (int i = 0; i < input_shape.size() - 2; i++) {
        shape.push_back(input_shape[i]);
      }
      for (auto e : sizes_arg) {
        shape.push_back(IntImm::make(e));
      }
      size_map[n->output()] = shape;
      printSizes(n->output());

      int64_t oH = sizes_arg[0];
      int64_t oW = sizes_arg[1];

      Stmt* s = nullptr;

      auto v = n->output();
      BufHandle ResultBuf("avgpool", size_map.at(v), kFloat);
      buf_map[v] = ResultBuf.node();
      s = ExternalCall::make(
          ResultBuf,
          "nnc_aten_adaptive_avg_pool2d",
          {BufHandle(buf_map.at(n->input(0)))},
          {oH, oW});

      std::cerr << "AvgPool stmt:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::expand_as) {
      auto out_shape = size_map.at(n->input(1));
      auto inp_shape = size_map.at(n->input(0));
      size_map[n->output()] = out_shape;
      printSizes(n->output());

      Tensor* t = Compute(
          std::string("expand_as_op"),
          c10::fmap<DimArg>(out_shape),
          [&](const std::vector<VarHandle>& axes) {
            int i = 0;
            std::vector<ExprHandle> input_indices;
            for (const auto& s : size_map.at(n->input(0))) {
              if (isOne(inp_shape[i])) {
                input_indices.push_back(IntImm::make(0));
              } else {
                input_indices.push_back(axes[i]);
              }
              i++;
            }

            std::vector<ExprHandle> indices(axes.begin(), axes.end());

            std::cerr << "===========================\n";
            print_vector(indices);
            print_vector(input_indices);

            return Load::make(
                BufHandle(buf_map.at(n->input(0))),
                input_indices,
                IntImm::make(1));
          });

      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "EXPAND_AS:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::flatten) {
      int start_dim = toIValue(n->input(1))->toInt();
      int end_dim = toIValue(n->input(2))->toInt();
      auto inp_shape = size_map.at(n->input(0));
      if (end_dim == -1) {
        end_dim = inp_shape.size();
      }

      std::vector<ExprHandle> out_shape;
      std::vector<ExprHandle> multipliers;
      std::vector<ExprHandle> mods;
      int idx = 0;
      for (idx = 0; idx < start_dim; idx++) {
        out_shape.push_back(inp_shape[idx]);
//         strides.push_back(IntImm::make(1));
      }
      ExprHandle flattened_shape = IntImm::make(1);
      for (; idx < end_dim; idx++) {
        flattened_shape = flattened_shape * inp_shape[idx];
      }
      out_shape.push_back(flattened_shape);
      for (; idx < inp_shape.size(); idx++) {
        out_shape.push_back(inp_shape[idx]);
      }
      size_map[n->output()] = out_shape;
      printSizes(n->output());

//       Tensor* t = Compute(
//           std::string("flatten_op"),
//           c10::fmap<DimArg>(out_shape),
//           [&](const std::vector<VarHandle>& axes) {
//             int i = 0;
//             std::vector<ExprHandle> input_indices;
//             for (const auto& s : size_map.at(n->output(0))) {
//               if (isOne(inp_shape[i])) {
//                 input_indices.push_back(IntImm::make(0));
//               } else {
//                 input_indices.push_back(axes[i]);
//               }
//               i++;
//             }
//
//             std::vector<ExprHandle> indices(axes.begin(), axes.end());
//
//             std::cerr << "===========================\n";
//             print_vector(indices);
//             print_vector(input_indices);
//
//             return Load::make(
//                 BufHandle(buf_map.at(n->input(0))),
//                 input_indices,
//                 IntImm::make(1));
//           });
//
//       buf_map[n->output()] = t->buf();
//       Stmt* s = t->stmt();
//       std::cerr << "FLATTEN:\n" << *s << "\n";
//       handled = true;
//       root_stmt->append_stmt(s);

    } else if (n->kind() == aten::dropout) {
      size_map[n->output()] = size_map.at(n->input(0));
      printSizes(n->output());
    } else if (n->kind() == aten::mean) {
      const auto& input_shape = size_map.at(n->input(0));
      auto dims_arg = toIValue(n->input(1))->toIntVector();
      std::unordered_set<int64_t> dims_set(dims_arg.begin(), dims_arg.end());
      auto keepdims_arg = toIValue(n->input(2))->toBool();
      std::vector<ExprHandle> shape;
      for (int i = 0; i < input_shape.size(); i++) {
        if (dims_set.count(i)) {
          if (keepdims_arg) {
            shape.push_back(IntImm::make(1));
          }
        } else {
          shape.push_back(input_shape[i]);
        }
      }
      size_map[n->output()] = shape;
      printSizes(n->output());

      Stmt* s = nullptr;

      auto v = n->output();
      BufHandle ResultBuf("mean", size_map.at(v), kFloat);
      buf_map[v] = ResultBuf.node();
      s = ExternalCall::make(
          ResultBuf,
          "nnc_aten_mean",
          {BufHandle(buf_map.at(n->input(0)))},
          {dims_arg[0]});

      std::cerr << "Mean stmt:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::reshape || n->kind() == aten::view) {
      auto sizes_arg = toIValue(n->input(1))->toIntVector();
      std::vector<ExprHandle> shape;
      for (auto e : sizes_arg) {
        shape.push_back(IntImm::make(e));
      }
      size_map[n->output()] = shape;
      printSizes(n->output());

      std::vector<DimArg> dims;
      for (int64_t dim : sizes_arg) {
        dims.emplace_back(IntImm::make(dim), "i");
      }
      Tensor* t = Compute(
          std::string("view_op"),
          dims,
          [&](const std::vector<VarHandle>& axes) {
            ExprHandle mult = IntImm::make(1);
            ExprHandle flattened_idx = IntImm::make(0);
            int i = 0;
            for (const auto& d : dims) {
              flattened_idx = flattened_idx + axes[i] * mult;
              mult = mult * IntImm::make(sizes_arg[i]);
              i++;
            }
            flattened_idx =
                ExprHandle(IRSimplifier::simplify(flattened_idx.node()));
            std::vector<ExprHandle> multipliers;
            std::vector<ExprHandle> input_indices;
            multipliers.push_back(IntImm::make(1));
            i = 0;
            for (const auto& s : size_map.at(n->input(0))) {
              multipliers.push_back(multipliers[i] * s);
              ExprHandle e = flattened_idx / multipliers[i] % s;
              input_indices.push_back(
                  ExprHandle(IRSimplifier::simplify(e.node())));
              i++;
            }

            std::vector<ExprHandle> indices(axes.begin(), axes.end());

            std::cerr << "===========================\n";
            print_vector(indices);
            std::cerr << sizes_arg << "\n";
            std::cerr << flattened_idx << "\n";
            print_vector(input_indices);

            return Load::make(
                BufHandle(buf_map.at(n->input(0))),
                input_indices,
                IntImm::make(1));
          });

      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "VIEW:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::dim) {
      const auto& input_shape = size_map.at(n->input(0));

      auto iv = IValue((int64_t)input_shape.size());
      ival_map[n->output()] = iv;
      WithInsertPoint p(n);
      auto new_v = g->insertConstant(iv);
      handleConstant(new_v->node());
      n->output()->replaceAllUsesAfterNodeWith(n, new_v);
      std::cerr << "%" << n->output()->debugName() << ": ";
      std::cerr << "ivalue " << iv << "\n";
      handled = true;
    } else if (n->kind() == aten::size) {
      c10::List<int64_t> l;
      for (const ExprHandle& e : size_map.at(n->input())) {
        l.push_back(e.AsNode<IntImm>()->value());
      }
      auto iv = IValue(l);
      std::cerr << iv << "\n";
      ival_map[n->output()] = iv;
      WithInsertPoint p(n);
      auto new_v = g->insertConstant(iv);
      handleConstant(new_v->node());
      n->output()->replaceAllUsesAfterNodeWith(n, new_v);
      std::cerr << "%" << n->output()->debugName() << ": ";
      std::cerr << "ivalue " << iv << "\n";
      handled = true;
    } else if (n->schema().name() == "aten::hardtanh_") {
      size_map[n->output()] = sizesForValue(n->input(0));
      printSizes(n->output());

      Tensor* t = nullptr;
      std::vector<const Buf*> bufs = {buf_map.at(n->input(0))};
      t = computeNaryOp(
          "hardtanh_op",
          bufs,
          {size_map.at(n->output())},
          [&](const std::vector<ExprHandle>& ops) {
            auto min_val = FloatImm::make(0.0);
            auto max_val = FloatImm::make(6.0);
            auto q = ops[0];
            auto mm = CompareSelect::make(
                tensorexpr::cast<float>(q), min_val, min_val, q, kLT);
            return CompareSelect::make(mm, max_val, max_val, mm, kGT);
          });
      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "HARDTANH:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->schema().name() == "aten::relu_") {
      size_map[n->output()] = sizesForValue(n->input(0));
      printSizes(n->output());

      Tensor* t = nullptr;
      std::vector<const Buf*> bufs = {buf_map.at(n->input(0))};
      t = computeNaryOp(
          "relu",
          bufs,
          {size_map.at(n->output())},
          [&](const std::vector<ExprHandle>& ops) {
            auto zero = FloatImm::make(0.0);
            auto op = tensorexpr::cast<float>(ops[0]);
            return CompareSelect::make(op, zero, zero, op, kLT);
          });
      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "RELU:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);
    } else if (n->kind() == aten::mul) {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 2; idx++) {
        torch::jit::Value* inp = n->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      size_map[n->output()] = broadcastShapes(shapes);
      printSizes(n->output());

      Tensor* t = nullptr;
      if (buf_map.count(n->input(1))) {
        std::vector<const Buf*> bufs = {
            buf_map.at(n->input(0)), buf_map.at(n->input(1))};
        t = computeNaryOp(
            "mul", bufs, shapes, [](const std::vector<ExprHandle>& ops) {
              return ops[0] * ops[1];
            });
      } else {
        std::vector<const Buf*> bufs = {buf_map.at(n->input(0))};
        t = computeNaryOp(
            "mul", bufs, shapes, [&](const std::vector<ExprHandle>& ops) {
            double vv = 1.0;
            if( ival_map.at(n->input(1)).isInt()) {
              vv = ival_map.at(n->input(1)).toInt();
            } else  if( ival_map.at(n->input(1)).isDouble()) {
              vv = ival_map.at(n->input(1)).toDouble();
            }
              return ops[0] *
                  tensorexpr::cast<float>(vv);
            });
      }
      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "MUL:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::div) {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 2; idx++) {
        torch::jit::Value* inp = n->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      size_map[n->output()] = broadcastShapes(shapes);
      printSizes(n->output());

      Tensor* t = nullptr;
      if (buf_map.count(n->input(1))) {
        std::vector<const Buf*> bufs = {
            buf_map.at(n->input(0)), buf_map.at(n->input(1))};
        t = computeNaryOp(
            "div", bufs, shapes, [](const std::vector<ExprHandle>& ops) {
              return ops[0] / ops[1];
            });
      } else {
        std::vector<const Buf*> bufs = {buf_map.at(n->input(0))};
        t = computeNaryOp(
            "div", bufs, shapes, [&](const std::vector<ExprHandle>& ops) {
              return ops[0] /
                  tensorexpr::cast<float>(ival_map.at(n->input(1)).toDouble());
            });
      }
      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "DIV:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::add) {
      std::vector<std::vector<ExprHandle>> shapes;
      for (size_t idx = 0; idx < 2; idx++) {
        torch::jit::Value* inp = n->input(idx);
        shapes.push_back(sizesForValue(inp));
      }
      size_map[n->output()] = broadcastShapes(shapes);
      printSizes(n->output());

      Tensor* t = nullptr;
      if (buf_map.count(n->input(1))) {
        std::vector<const Buf*> bufs = {
            buf_map.at(n->input(0)), buf_map.at(n->input(1))};
        t = computeNaryOp(
            "add", bufs, shapes, [](const std::vector<ExprHandle>& ops) {
              return ops[0] + ops[1];
            });
      } else {
        std::vector<const Buf*> bufs = {buf_map.at(n->input(0))};
        t = computeNaryOp(
            "add", bufs, shapes, [&](const std::vector<ExprHandle>& ops) {
              return ops[0] +
                  tensorexpr::cast<float>(ival_map.at(n->input(1)).toDouble());
            });
      }
      buf_map[n->output()] = t->buf();
      Stmt* s = t->stmt();
      std::cerr << "ADD:\n" << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);

    } else if (n->kind() == aten::conv2d) {
      std::cerr << *n << "\n";
      auto input_shape = sizesForValue(n->input(0));
      auto weight_shape = sizesForValue(n->input(1));
      auto bias_shape = sizesForValue(n->input(2));

      int sH, sW;
      auto strides_iv = *toIValue(n->input(3));
      if (strides_iv.isIntList()) {
        sH = strides_iv.toIntList()[0];
        sW = strides_iv.toIntList()[1];
      } else {
        sH = sW = strides_iv.toInt();
      }
      int pH, pW;
      auto padding_iv = *toIValue(n->input(4));
      if (padding_iv.isIntList()) {
        pH = padding_iv.toIntList()[0];
        pW = padding_iv.toIntList()[1];
      } else {
        pH = pW = padding_iv.toInt();
      }
      int dH, dW;
      auto dil_iv = *toIValue(n->input(5));
      if (dil_iv.isIntList()) {
        dH = dil_iv.toIntList()[0];
        dW = dil_iv.toIntList()[1];
      } else {
        dH = dW = dil_iv.toInt();
      }
      int groups = toIValue(n->input(6))->toInt();

      ExprHandle oH = IRSimplifier::simplify(tensorexpr::cast<int>(
          (input_shape[2] + 2 * pH - (weight_shape[2] - 1) * dH - 1) / sH + 1));
      ExprHandle oW = IRSimplifier::simplify(tensorexpr::cast<int>(
          (input_shape[3] + 2 * pW - (weight_shape[3] - 1) * dW - 1) / sW + 1));
      size_map[n->output()] = {input_shape[0], weight_shape[0], oH, oW};
      printSizes(n->output());

      Stmt* s = generate_conv(
          n->output(),
          buf_map.at(n->input(0)),
          buf_map.at(n->input(1)),
          buf_map.at(n->input(2)),
          sH,
          sW,
          pH,
          pW,
          dH,
          dW,
          groups);
      std::cerr << "Conv stmt:\n[QQ] " << *s << "\n";
      handled = true;
      root_stmt->append_stmt(s);
      std::cerr << "[QQ] Input shape: ";
      print_vector(input_shape);
      std::cerr << "[QQ] Weight shape: ";
      print_vector(weight_shape);
      std::cerr << "[QQ] Bias shape: ";
      print_vector(bias_shape);
      std::cerr << "[QQ] sH, sW = {" << sH << ", " << sW << "}, pH, pW = {"
                << pH << ", " << pW << "}, dH, dW = {" << dH << ", " << dW
                << "}, groups = " << groups << "\n";
    }
    if (n->outputs().size() == 1 && n->kind() != prim::Constant) {
      if (!JIT_OPT_ALLOWED) {
        optimized_graph->outputs()[0]->replaceAllUsesWith(n->output());
        const auto end_it = n->reverseIterator();
        auto it = optimized_graph->return_node()->reverseIterator();
        it++;
        for (; it != end_it;) {
          auto next = it;
          next++;
          (*it)->destroy();
          it = next;
        }

        EliminateDeadCode(optimized_graph->block());
        return;
      }
    }

    if (!handled) {
      std::cerr << "Unhandled node type: " << s << "\n";
      std::cerr << *n << "\n";
    }
  }
}

void fancy_compile(std::shared_ptr<Graph>& g, const std::vector<int>& sizes) {
  KernelScope kernel_scope;
  std::cerr << "Hi from fancy_compile!\nSizes: ";
  for (auto s : sizes) {
    std::cerr << s << ", ";
  }
  std::cerr << "\n";

  AOT_NNC_Compiler aot(g);
  aot.compile_for_sizes(sizes);
  //   at::Tensor input_tensor = at::ones({1, 3, 224, 224});
  at::Tensor input_tensor = at::randn({1, 3, 224, 224}) * 1000.0;
  auto a = aot.call_with_nnc(input_tensor);
  auto b = aot.call_with_jit_optimized_graph(input_tensor);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
