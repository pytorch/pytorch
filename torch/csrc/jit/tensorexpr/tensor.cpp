#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/tensorexpr/reduction.h>

namespace torch::jit::tensorexpr {

StmtPtr Tensor::constructStmt(
    const std::vector<VarPtr>& args,
    ExprPtr body,
    const std::vector<ExprPtr>& reduce_dims,
    const std::vector<VarPtr>& reduce_args) const {
  std::vector<ExprPtr> indices(args.begin(), args.end());

  size_t ndim = buf()->ndim();
  size_t reduce_ndim = reduce_dims.size();
  auto reduce_op = to<ReduceOp>(body);
  auto acc_buf = reduce_ndim > 0 ? reduce_op->getAccBuf() : nullptr;

  StmtPtr s = alloc<Store>(buf_, indices, body);
  if (reduce_ndim > 0) {
    TORCH_INTERNAL_ASSERT(reduce_op != nullptr);
    if (acc_buf != nullptr) {
      auto reducer = reduce_op->reducer();
      std::vector<ExprPtr> output_args(args.begin(), args.end());
      ExprPtr new_reduce_op = reducer(
          to<Buf>(acc_buf),
          alloc<Cast>(acc_buf->dtype(), reduce_op->getRiOperand()),
          output_args,
          reduce_args);
      new_reduce_op->set_dtype(acc_buf->dtype());
      s = alloc<Store>(to<Buf>(acc_buf), indices, new_reduce_op);
    }
  }

  if (ndim == 0 && reduce_ndim == 0) {
    return s;
  }

  if (reduce_ndim > 0) {
    TORCH_INTERNAL_ASSERT(reduce_op != nullptr);

    for (const auto i : c10::irange(reduce_ndim)) {
      // Going in reverse order: from innermost loop to the outermost
      size_t dim_index = reduce_ndim - i - 1;
      auto const& dim = reduce_dims[dim_index];
      s = alloc<For>(reduce_args[dim_index], immLike(dim, 0), dim, s);
    }
    s = alloc<Block>(std::vector<StmtPtr>({s}));

    BufPtr init_buf = acc_buf ? to<Buf>(acc_buf) : buf();
    ExprPtr init_expr =
        acc_buf ? to<Buf>(acc_buf)->initializer() : buf()->initializer();
    if (init_expr) {
      StorePtr init_stmt = alloc<Store>(init_buf, indices, init_expr);
      to<Block>(s)->prepend_stmt(init_stmt);
    }

    if (acc_buf != nullptr) {
      LoadPtr load_acc = alloc<Load>(acc_buf, indices);
      auto cast = alloc<Cast>(buf()->dtype(), load_acc);
      StorePtr post_stmt = alloc<Store>(buf(), indices, cast);
      to<Block>(s)->append_stmt(post_stmt);
    }
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      buf_->is_contiguous() ||
      buf_->is_contiguous(at::MemoryFormat::ChannelsLast) ||
      buf_->is_contiguous(at::MemoryFormat::ChannelsLast3d) ||
      buf_->is_channels_last_1d_contiguous());

  auto loop_order_fn = [&]() {
    std::vector<int32_t> loop_order;
    if (buf_->is_contiguous()) {
      for (int32_t i = args.size() - 1; i >= 0; i--) {
        loop_order.push_back(i);
      }
    } else if (buf_->is_contiguous(c10::MemoryFormat::ChannelsLast)) {
      loop_order = {1, 3, 2, 0};
    } else if (buf_->is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
      loop_order = {1, 4, 3, 2, 0};
    } else {
      loop_order = {1, 2, 0};
    }

    return loop_order;
  };

  auto loop_order = loop_order_fn();
  for (auto dim_index : loop_order) {
    auto const& dim = buf()->dim(dim_index);
    s = alloc<For>(args[dim_index], immLike(dim, 0), dim, s);
  }
  return s;
}

Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func) {
  std::vector<VarHandle> args = create_index_vars(dims);
  ExprHandle body = body_func(args);
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  return Tensor(buf, args, body);
}
Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const std::vector<VarHandle>&)>& body_func) {
  return Compute(name, dims, c10::nullopt, body_func);
}

Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&)>& body_func) {
  if (dims.size() != 1) {
    throw malformed_input("mismatch between body and arg size (1)");
  }

  std::vector<VarHandle> args = create_index_vars(dims);
  ExprHandle body = body_func(args[0]);
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  return Tensor(buf, args, body);
}
Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&)>& body_func) {
  return Compute(name, dims, c10::nullopt, body_func);
}

Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func) {
  if (dims.size() != 2) {
    throw malformed_input("mismatch between body and arg size (2)");
  }
  std::vector<VarHandle> args = create_index_vars(dims);
  ExprHandle body = body_func(args[0], args[1]);
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  return Tensor(buf, args, body);
}
Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(const VarHandle&, const VarHandle&)>&
        body_func) {
  return Compute(name, dims, c10::nullopt, body_func);
}

Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func) {
  if (dims.size() != 3) {
    throw malformed_input("mismatch between body and arg size (3)");
  }
  std::vector<VarHandle> args = create_index_vars(dims);
  ExprHandle body = body_func(args[0], args[1], args[2]);
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  return Tensor(buf, args, body);
}
Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<
        ExprHandle(const VarHandle&, const VarHandle&, const VarHandle&)>&
        body_func) {
  return Compute(name, dims, c10::nullopt, body_func);
}

Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func) {
  if (dims.size() != 4) {
    throw malformed_input("mismatch between body and arg size (4)");
  }
  std::vector<VarHandle> args = create_index_vars(dims);
  ExprHandle body = body_func(args[0], args[1], args[2], args[3]);
  BufHandle buf = Buf::make(name, dims, body.dtype(), c10::nullopt, strides);
  return Tensor(buf, args, body);
}
Tensor Compute(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const std::function<ExprHandle(
        const VarHandle&,
        const VarHandle&,
        const VarHandle&,
        const VarHandle&)>& body_func) {
  return Compute(name, dims, c10::nullopt, body_func);
}

Tensor Reduce(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    const BufHandle& buffer,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(
      name,
      dims,
      strides,
      reducer,
      [&](ParameterList& p) { return buffer.load(p); },
      reduce_dims);
}
Tensor Reduce(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    const BufHandle& buffer,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(name, dims, c10::nullopt, reducer, buffer, reduce_dims);
}

Tensor Reduce(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    c10::optional<std::vector<ExprHandle>> strides,
    const Reducer& reducer,
    Tensor tensor,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(
      name,
      dims,
      strides,
      reducer,
      [&](ParameterList& p) { return tensor.load(p); },
      reduce_dims);
}
Tensor Reduce(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    const Reducer& reducer,
    Tensor tensor,
    const std::vector<ExprHandle>& reduce_dims) {
  return Reduce(name, dims, c10::nullopt, reducer, tensor, reduce_dims);
}

} // namespace torch::jit::tensorexpr
