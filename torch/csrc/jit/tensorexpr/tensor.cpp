#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/Logging.h>
#include <torch/csrc/jit/tensorexpr/dim_arg.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch {
namespace jit {
namespace tensorexpr {

namespace {
class LowerReduceX : public IRMutator {
 public:
  LowerReduceX(const std::vector<const Expr*> output_args)
      : IRMutator(), output_args_(output_args)
    {}

  Stmt* mutate(const Store* v) override {
    const ReduceXOp* reduceX = dynamic_cast<const ReduceXOp*>(v->value());
    if (!reduceX) {
      return const_cast<Store*>(v);
    }
    // Turn this store:
    //     base[indices] = ReduceX(body, rvars, rdims, reducer)
    // into:
    //     base[indices] = reducer.init
    //     for (rvar in rdims):
    //         base[indices] = Reduce(
    //             base,
    //             reducer.apply(base[indices], body),
    //             output_args,
    //             rvars,
    //             reducer)
    auto const& reducer = reduceX->reducer();
    const Expr* init_expr = new Cast(reduceX->body()->dtype(), reducer.initializer());
    Store* init = new Store(v->buf(), v->indices(), init_expr, v->mask());
    Expr* reduce = reducer(v->buf(), reduceX->body(), output_args_, reduceX->rvars());
    Stmt* update = new Store(v->buf(), v->indices(), reduce, v->mask());

    auto const& rvars = reduceX->rvars();
    auto const& rdims = reduceX->rdims();
    size_t reduce_ndim = rdims.size();
    for (size_t i = 0; i < reduce_ndim; i++) {
      size_t dim_index = reduce_ndim - i - 1;
      update = new For(rvars[dim_index], new IntImm(0), rdims[dim_index], update);
    }
    return new Block({init, update});
  }

 private:
  std::vector<const Expr*> output_args_;
};
}

// Transform ReduceX into For loops surrounding an ordinary Reduce.
static Stmt* lowerReduceX(Stmt* s, const std::vector<const Var*> output_args) {
  LowerReduceX lowering(c10::fmap<const Expr*>(output_args));
  return s->accept_mutator(&lowering);
}

Stmt* Tensor::constructStmt(
    const std::vector<const Var*>& args,
    const Expr* body,
    const std::vector<const Expr*>& reduce_dims,
    const std::vector<const Var*>& reduce_args) const {
  std::vector<const Expr*> indices(args.begin(), args.end());

  const Expr* mask = new IntImm(1);
  Stmt* s = new Store(buf_, indices, body, mask);

  size_t ndim = buf()->ndim();
  size_t reduce_ndim = reduce_dims.size();

  if (ndim == 0 && reduce_ndim == 0) {
    return s;
  }

  const Expr* init_expr = buf()->initializer();

  if (reduce_ndim > 0) {
    for (size_t i = 0; i < reduce_ndim; i++) {
      // Going in reverse order: from innermost loop to the outermost
      size_t dim_index = reduce_ndim - i - 1;
      s = new For(
          reduce_args[dim_index], new IntImm(0), reduce_dims[dim_index], s);
    }
    if (init_expr) {
      Store* init_stmt = new Store(buf(), indices, init_expr, new IntImm(1));
      s = new Block({init_stmt, s});
    }
  }

  for (size_t i = 0; i < ndim; i++) {
    // Going in reverse order: from innermost loop to the outermost
    size_t dim_index = ndim - i - 1;
    s = new For(args[dim_index], new IntImm(0), buf()->dim(dim_index), s);
  }

  s = lowerReduceX(s, args);
  return s;
}

Tensor* Compute(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func) {
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(VarVectorToVarHandleVector(args)).node();
  const Buf* buf = new Buf(name, dims, body->dtype());
  return new Tensor(buf, args, body);
}

Tensor* Compute(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&)>& body_func) {
  if (dim_args.size() != 1) {
    throw malformed_input("mismatch between body and arg size (1)");
  }

  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(VarHandle(args[0])).node();
  const Buf* buf = new Buf(name, dims, body->dtype());
  return new Tensor(buf, args, body);
}

Tensor* Compute(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func) {
  if (dim_args.size() != 2) {
    throw malformed_input("mismatch between body and arg size (2)");
  }
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(VarHandle(args[0]), VarHandle(args[1])).node();
  const Buf* buf = new Buf(name, dims, body->dtype());
  return new Tensor(buf, args, body);
}

Tensor* Compute(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func) {
  if (dim_args.size() != 3) {
    throw malformed_input("mismatch between body and arg size (3)");
  }
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body =
      body_func(VarHandle(args[0]), VarHandle(args[1]), VarHandle(args[2]))
          .node();
  const Buf* buf = new Buf(name, dims, body->dtype());
  return new Tensor(buf, args, body);
}

Tensor* Compute(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func) {
  if (dim_args.size() != 4) {
    throw malformed_input("mismatch between body and arg size (4)");
  }
  std::vector<const Expr*> dims;
  std::vector<const Var*> args;
  unpack_dim_args(dim_args, &dims, &args);
  const Expr* body = body_func(
                         VarHandle(args[0]),
                         VarHandle(args[1]),
                         VarHandle(args[2]),
                         VarHandle(args[3]))
                         .node();
  const Buf* buf = new Buf(name, dims, body->dtype());
  return new Tensor(buf, args, body);
}

Tensor* Reduce(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    const Placeholder& buffer,
    const std::vector<DimArg>& reduce_args) {
  return Reduce(
      name,
      dim_args,
      reducer,
      [&](ParameterList& p) { return buffer.load(p); },
      reduce_args);
}

Tensor* Reduce(
    const std::string& name,
    const std::vector<DimArg>& dim_args,
    const Reducer& reducer,
    Tensor* tensor,
    const std::vector<DimArg>& reduce_args) {
  return Reduce(
      name,
      dim_args,
      reducer,
      [&](ParameterList& p) { return tensor->call(p); },
      reduce_args);
}

// Tensor* ComputeX(
//     const std::string& name,
//     const std::vector<DimArg>& dim_args,
//     const std::function<Stmt*(const std::vector<VarHandle>&)>& body_func) {
//   std::vector<const Expr*> dims;
//   std::vector<const Var*> vars;
//   unpack_dim_args(dim_args, &dims, &vars);


// }

Expr* SumX(
    const std::vector<DimArg>& rdim_args,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body) {
  std::vector<const Expr*> rdims;
  std::vector<const Var*> rvars;
  unpack_dim_args(rdim_args, &rdims, &rvars);

  auto rvarhs = c10::fmap<VarHandle>(rvars);
  return new ReduceXOp(body(rvarhs).node(), rvars, rdims, Sum());
}


} // namespace tensorexpr
} // namespace jit
} // namespace torch
