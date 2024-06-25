#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

int64_t normalizeAndCheckIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }

  if (idx < 0 || idx >= list_size) {
    AT_ERROR("Invalid index ", idx, " for list_size", list_size);
  }
  return idx;
}

// Convert boolean to integer, if needed.
ExprHandle boolToInteger(const ExprHandle& x) {
  return x.dtype().scalar_type() == ScalarType::Bool ? cast<int>(x) : x;
}

ExprHandle promoteToDtype(ExprHandle e, ScalarType dt) {
  if (e.dtype().scalar_type() == dt) {
    return e;
  }

  switch (dt) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    e = cast<Type>(e);        \
    break;
    AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::QUInt8:
      e = cast<c10::quint8>(e);
      break;
    case ScalarType::QInt8:
      e = cast<c10::qint8>(e);
      break;
    default:
      throw unsupported_dtype();
  }
  return e;
}

static bool checkTypes(const ScalarType highType, const int typeConstraints) {
  if (typeConstraints == kAllTypes) {
    return true;
  }

  if (c10::isIntegralType(highType, false)) {
    return (typeConstraints & kIntegralTypes) != 0;
  } else if (c10::isFloatingType(highType)) {
    return (typeConstraints & kFloatingPointTypes) != 0;
  } else if (highType == ScalarType::Bool) {
    return (typeConstraints & kBoolType) != 0;
  }

  // assume JIT not supporting complex and qint yet
  TORCH_INTERNAL_ASSERT(
      (typeConstraints & (kQintTypes | kComplexTypes)) == 0,
      buildErrorMessage(
          "Qint and Complex types are not supported in the fuser."));
  return false;
}

static bool isScalar(ExprHandle e) {
  auto n = e.node();
  return n->isConstant() || to<Var>(n);
}

ExprHandle promoteHalfToFloat(const ExprHandle& e) {
  auto scalarType = static_cast<c10::ScalarType>(e.dtype().scalar_type());
  auto floatType = static_cast<c10::ScalarType>(tensorexpr::ScalarType::Float);
  if (c10::isFloatingType(scalarType) &&
      (c10::elementSize(scalarType) < c10::elementSize(floatType))) {
    return Cast::make(
        Dtype(tensorexpr::ScalarType::Float, e.dtype().lanes()), e);
  } else {
    return e;
  }
}

void promoteInputs(std::vector<ExprHandle>& inputs, const int typeConstraints) {
  if (inputs.empty()) {
    return;
  }

  // Find the highest type among the inputs.
  ScalarType highType = inputs[0].dtype().scalar_type();
  for (const auto& input : inputs) {
    auto inputType = input.dtype().scalar_type();
    if (isScalar(input)) {
      if (isIntegralType(highType, false) && isFloatingType(inputType)) {
        highType = c10::get_default_dtype_as_scalartype();
      } else if (highType == c10::kBool) {
        highType = inputType;
      }
    } else {
      highType = promoteTypes(highType, inputType);
    }
  }

  if (!checkTypes(highType, typeConstraints)) {
    throw unsupported_dtype();
  }

  for (ExprHandle& e : inputs) {
    e = promoteToDtype(e, highType);
  }
}

ExprHandle promoteIntegerToDefaultType(const ExprHandle& e) {
  auto scalarType = static_cast<c10::ScalarType>(e.dtype().scalar_type());
  if (!c10::isIntegralType(scalarType, /*includeBool*/ true)) {
    return e;
  }

  auto defaultType = c10::typeMetaToScalarType(c10::get_default_dtype());

  // We intend to promote Integers to floating-point types
  TORCH_INTERNAL_ASSERT(
      !c10::isIntegralType(defaultType, /*includeBool*/ true));

  return Cast::make(
      Dtype(
          static_cast<tensorexpr::ScalarType>(defaultType), e.dtype().lanes()),
      e);
}

ExprHandle demoteOutput(
    const ExprHandle& e,
    const std::optional<ScalarType> type) {
  if (!type.has_value()) {
    return e;
  }
  if (*type == e.dtype().scalar_type()) {
    return e;
  }

  switch (*type) {
// NOLINTNEXTLINE
#define TYPE_CASE(Type, Name) \
  case ScalarType::Name:      \
    return cast<Type>(e);
    AT_FORALL_SCALAR_TYPES_AND2(Half, BFloat16, TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::Bool:
      return cast<bool>(e);
    default:
      throw unsupported_dtype();
  }

  return e;
}

std::optional<TensorInfo> getTensorInfo(BufHandle b) {
  std::vector<int64_t> dims;
  for (auto dim : b.dims()) {
    auto val = intValue(dim.node());
    if (!val) {
      return c10::nullopt;
    }
    dims.push_back(*val);
  }
  return TensorInfo{dims, static_cast<at::ScalarType>(b.dtype().scalar_type())};
}

ExprHandle clamp(
    const ExprHandle& cmin,
    const ExprHandle& cmax,
    const ExprHandle& input) {
  auto mm = CompareSelect::make(input, cmin, cmin, input, kLT);
  return CompareSelect::make(mm, cmax, cmax, mm, kGT);
}

static bool isOne(ExprHandle e) {
  auto const& n = intValue(e);
  if (!n) {
    return false;
  }
  return *n == 1;
}

static std::pair<std::vector<ExprHandle>, bool> broadcastShapesImpl(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  auto at = a.rbegin();
  auto bt = b.rbegin();
  std::vector<ExprHandle> ret;
  bool hasBroadcast = false;
  while (at != a.rend() || bt != b.rend()) {
    if (at == a.rend()) {
      hasBroadcast = true;
      ret.push_back(*bt++);
      continue;
    }
    if (bt == b.rend()) {
      hasBroadcast = true;
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
        hasBroadcast = true;
      }
    }
    ret.push_back(dim);
    at++;
    bt++;
  }
  std::reverse(ret.begin(), ret.end());
  return {ret, hasBroadcast};
}

static std::pair<std::vector<ExprHandle>, bool> broadcastShapesImpl(
    std::vector<std::vector<ExprHandle>> shapes) {
  size_t n = shapes.size();
  if (n == 1) {
    return {shapes[0], false};
  }
  auto res1 = broadcastShapesImpl(shapes[n - 2], shapes[n - 1]);
  shapes[n - 2] = res1.first;
  shapes.pop_back();
  auto res2 = broadcastShapesImpl(shapes);
  return {res2.first, (res1.second || res2.second)};
}

std::vector<ExprHandle> broadcastShapes(
    std::vector<std::vector<ExprHandle>> shapes) {
  return broadcastShapesImpl(shapes).first;
}

std::vector<ExprHandle> broadcastShapes(
    const std::vector<ExprHandle>& a,
    const std::vector<ExprHandle>& b) {
  return broadcastShapesImpl(a, b).first;
}

std::vector<ExprHandle> valueShape(const ArgValue& v) {
  if (auto b = std::get_if<tensorexpr::BufHandle>(&v)) {
    return b->dims();
  }
  return {};
}

ExprHandle tensorOrConstant(
    const ArgValue& v,
    const std::vector<ExprHandle>& axes) {
  if (auto b = std::get_if<BufHandle>(&v)) {
    return broadcast(*b, axes);
  }
  return constant(v);
}

ExprHandle scalarOrConstant(const ArgValue& v) {
  if (auto vh = std::get_if<VarHandle>(&v)) {
    return *vh;
  }
  return constant(v);
}

ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes) {
  return b.load(computeIndicesToBroadcast(axes, b.dims()));
}

ExprHandle constant(const ArgValue& v) {
  if (auto s = std::get_if<tensorexpr::VarHandle>(&v)) {
    return *s;
  } else if (auto d = std::get_if<double>(&v)) {
    return DoubleImm::make(*d);
  } else if (auto i = std::get_if<int64_t>(&v)) {
    return LongImm::make(*i);
  } else if (auto b = std::get_if<bool>(&v)) {
    return BoolImm::make(*b);
  } else if (std::get_if<ArgNone>(&v)) {
    // This is just a placeholder so we don't throw.  None-handling
    // is operator-specific and should be handled properly in
    // the operator-specific lowering code.
    return IntImm::make(0);
  } else {
    throw unsupported_dtype("Trying to convert unsupported dtype to constant");
  }
}

std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<ExprHandle>& outputAxes,
    const std::vector<ExprHandle>& inputSizes) {
  if (outputAxes.size() < inputSizes.size()) {
    throw malformed_input("Cannot broadcast to a lower rank tensor");
  }
  std::vector<ExprHandle> bcast;
  auto axisIt = outputAxes.rbegin();
  auto sizeIt = inputSizes.rbegin();
  while (sizeIt != inputSizes.rend()) {
    auto const& size = intValue(*sizeIt);
    if (size && *size == 1) {
      bcast.emplace_back(LongImm::make(0));
    } else {
      bcast.emplace_back(*axisIt);
    }
    ++axisIt;
    ++sizeIt;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

Tensor computeChunk(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  return Compute(
      "prim_constantchunk",
      outputShape,
      [inputs](const std::vector<VarHandle>& axes) {
        const auto& b = std::get<BufHandle>(inputs[0]);
        int64_t chunkIdx = std::get<int64_t>(inputs[1]);
        int64_t dim = std::get<int64_t>(inputs[2]);
        int64_t chunks = std::get<int64_t>(inputs[3]);
        std::vector<ExprHandle> indices(axes.begin(), axes.end());

        auto norm_dim = normalizeAndCheckIndex(dim, indices.size());
        auto buf_info = getTensorInfo(b);
        size_t step = buf_info->dims[norm_dim] / chunks;

        std::vector<ExprHandle> new_indices;
        for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); ++i) {
          if (i == norm_dim) {
            new_indices.push_back(
                indices[i] + ExprHandle(immLike(indices[i], chunkIdx * step)));
          } else {
            new_indices.push_back(indices[i]);
          }
        }

        return b.load(new_indices);
      });
}

Tensor computeTranspose(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  auto A = std::get<BufHandle>(inputs[0]);
  // Trivial case of 0-dim and 1-dim tensors: transpose is just a copy
  if (A.ndim() <= 1) {
    return Compute(
        "aten_transpose", outputShape, [&](std::vector<VarHandle> axes) {
          TORCH_INTERNAL_ASSERT(
              axes.size() <= 1,
              buildErrorMessage("Invalid axes size in transpose"));
          return A.load(axes);
        });
  }
  // Usual case where transpose actually swaps dimensions
  auto start_dim = at::maybe_wrap_dim(std::get<int64_t>(inputs[1]), A.ndim());
  auto to_dim = at::maybe_wrap_dim(std::get<int64_t>(inputs[2]), A.ndim());
  return Compute(
      "aten_transpose", outputShape, [&](std::vector<VarHandle> axes) {
        std::swap(axes[start_dim], axes[to_dim]);
        return A.load(axes);
      });
}

Tensor computeExpand(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  auto A = std::get<BufHandle>(inputs[0]);
  return Compute(
      "aten_expand", outputShape, [&](const std::vector<VarHandle>& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        return broadcast(A, indices);
      });
}

Tensor computeReshape(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  auto A = std::get<BufHandle>(inputs[0]);
  if (A.ndim() == 0) {
    return Compute(
        "aten_view", outputShape, [&](const std::vector<VarHandle>& axes) {
          std::vector<ExprHandle> empty_indices;
          return A.load(empty_indices);
        });
  }
  return Compute(
      "aten_reshape", outputShape, [&](const std::vector<VarHandle>& axes) {
        std::vector<VarHandle> new_axes;
        assert(outputShape.size() == axes.size());
        /*
        Example for the index transformation. Assume we have a tensor A and
        its view B:
          A.size() = [6,2,3]
          B = A.view(2,1,9,1,2)

        In TE IR we would want to represent B as the following loopnest:
          for (i1 in 0..2)
            for (i2 in 0..1)
              for (i3 in 0..9)
                for (i4 in 0..1)
                  for (i5 in 0..2)
                    idx = i5 + i4*2 + i3*2 + i2*18 + i1*18
                    B[i1,i2,i3,i4,i5] = A[idx/(3*2), (idx/3)%2, idx%3]
        */
        std::vector<ExprPtr> dims, indices;
        for (size_t idx = 0; idx < outputShape.size(); idx++) {
          dims.push_back(outputShape[idx].node());
          indices.push_back(axes[idx].node());
        }

        auto ndim = dims.size();
        std::vector<ExprPtr> strides(ndim);
        strides[ndim - 1] = immLike(dims[ndim - 1], 1);
        for (size_t i = 1; i < ndim; i++) {
          strides[ndim - 1 - i] = alloc<Mul>(strides[ndim - i], dims[ndim - i]);
        }

        ExprHandle flat_idx = ExprHandle(flatten_index(dims, indices, strides));
        std::vector<ExprHandle> orig_buf_indexes(A.ndim(), ExprHandle(0));
        ExprHandle stride = ExprHandle(immLike(flat_idx, 1));
        for (size_t idx = 0; idx < A.ndim(); idx++) {
          size_t dim_idx = A.ndim() - idx - 1;
          // We don't need to generate mod-div for the first dimension -
          // ideally IRSimplifier would get rid of that for us, but for now
          // let's just avoid generating it in the first place.
          if (dim_idx > 0) {
            orig_buf_indexes[dim_idx] = flat_idx / stride % A.dim(dim_idx);
          } else {
            orig_buf_indexes[dim_idx] = flat_idx / stride;
          }
          // In the example above the stride is initially 1 for dim_idx = 2,
          // then it's 3 for dim_idx = 1, and then it's 3*2 for dim_idx = 0.
          stride = stride * A.dim(dim_idx);
        }
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        return A.load(orig_buf_indexes);
      });
}

Tensor computeFlatten(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  std::vector<int64_t> outputShapeVec;
  for (const auto dim : c10::irange(outputShape.size())) {
    outputShapeVec.push_back(outputShape[dim].AsNode<LongImm>()->value());
  }
  std::vector<ArgValue> reshapeInputs;
  reshapeInputs.push_back(inputs[0]);
  reshapeInputs.emplace_back(outputShapeVec);
  return computeReshape(
      reshapeInputs, outputShape, outputStrides, outputType, device);
}

static std::pair<ScalarType, std::vector<BufHandle>> processCatList(
    const std::vector<BufHandle>& bufList) {
  if (bufList.empty()) {
    throw std::runtime_error("Empty input list is passed to aten::cat");
  }
  std::vector<BufHandle> bufInputs;
  std::vector<BufHandle> nonEmptyInputs;
  for (auto buf : bufList) {
    bufInputs.push_back(buf);
    TORCH_INTERNAL_ASSERT(
        !buf.node()->dims().empty(), buildErrorMessage("Invalid buf rank"));
    // Ignore buffers that are 0-sized on any dimension.
    bool hasEmptyDims = false;
    for (const auto& dim : buf.dims()) {
      if (dim.AsNode<LongImm>() && immediateAs<int64_t>(dim) == 0ll) {
        hasEmptyDims = true;
        break;
      }
    }
    if (!hasEmptyDims) {
      nonEmptyInputs.push_back(buf);
    }
  }
  ScalarType highType = bufInputs[0].dtype().scalar_type();
  for (const auto& input : bufInputs) {
    auto maybe_dtype = input.dtype().scalar_type();
    highType = promoteTypes(highType, maybe_dtype);
  }
  return {highType, nonEmptyInputs};
}

static Tensor computeCatWoConditionals(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto input_list = std::get<BufList>(inputs[0]);
  auto arg_dim = inputs[1];
  auto cat_info = processCatList(input_list);
  ScalarType high_type = cat_info.first;
  std::vector<BufHandle> non_empty_inputs = cat_info.second;

  // Now we build one loop per input:
  //
  // for i
  //   for j
  //     for k
  //       output[i,j,k] = inp1[i,j,k]
  // for i
  //   for j
  //     for k
  //       output[i,j+l1,k] = inp2[i,j,k]
  // for i
  //   for j
  //     for k
  //       output[i,j+l2,k] = inp3[i,j,k]

  auto output_sizes_expr = ExprHandleVectorToExprVector(outputShape);
  auto output_strides_expr = ExprHandleVectorToExprVector(outputStrides);
  auto output_buf = alloc<Buf>(
      "aten_cat",
      output_sizes_expr,
      ToDtype(high_type),
      nullptr,
      output_strides_expr);
  if (non_empty_inputs.empty()) {
    return Tensor(
        output_buf, alloc<tensorexpr::Block>(std::vector<StmtPtr>({})));
  }

  int64_t concat_dim = std::get<int64_t>(arg_dim);
  auto norm_concat_dim = normalizeAndCheckIndex(concat_dim, outputShape.size());

  auto loop_order_fn = [&](const BufPtr& buf_) {
    std::vector<int32_t> loop_order;
    if (buf_->is_contiguous()) {
      for (int32_t i = buf_->ndim() - 1; i >= 0; i--) {
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

  auto gen_code_for_input = [&](const BufHandle& inp,
                                size_t inp_pos,
                                ExprPtr concat_dim_size,
                                const std::vector<ExprHandle>& dims) {
    std::vector<VarPtr> for_vars(dims.size());
    std::vector<ExprPtr> load_indices(dims.size());
    std::vector<ExprPtr> store_indices(dims.size());
    for (int64_t i = 0; i < static_cast<int64_t>(dims.size()); ++i) {
      for_vars[i] = alloc<Var>(
          "i" + std::to_string(inp_pos) + "_" + std::to_string(i),
          dims[i].dtype());
      load_indices[i] = for_vars[i];
      if (i == norm_concat_dim) {
        store_indices[i] = alloc<Add>(for_vars[i], concat_dim_size);
      } else {
        store_indices[i] = for_vars[i];
      }
    }
    auto inp_buf = inp.node();
    auto load_expr = alloc<Load>(inp_buf, load_indices);
    auto load_promoted = promoteToDtype(ExprHandle(load_expr), high_type);
    StmtPtr st = alloc<Store>(output_buf, store_indices, load_promoted.node());

    auto loop_order = loop_order_fn(inp.node());
    for (auto dim_index : loop_order) {
      st = alloc<For>(
          for_vars[dim_index],
          immLike(dims[dim_index], 0),
          dims[dim_index].node(),
          st);
    }

    return st;
  };

  ExprPtr concat_dim_size = nullptr;
  auto block = alloc<tensorexpr::Block>(std::vector<StmtPtr>({}));
  for (size_t i = 0; i < non_empty_inputs.size(); ++i) {
    auto input_dims =
        ExprVectorToExprHandleVector(non_empty_inputs[i].node()->dims());
    if (concat_dim_size == nullptr) {
      concat_dim_size = immLike(input_dims[norm_concat_dim], 0);
    }
    block->append_stmt(gen_code_for_input(
        non_empty_inputs[i], i, concat_dim_size, input_dims));
    concat_dim_size =
        alloc<Add>(concat_dim_size, input_dims[norm_concat_dim].node());
  }
  return Tensor(output_buf, IRSimplifier::simplify(block));
}

Tensor computeCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  if (device == at::kCPU && getCatWoConditionals()) {
    return computeCatWoConditionals(inputs, outputShape, outputStrides);
  }
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto inputList = std::get<BufList>(inputs[0]);
  auto argDim = inputs[1];
  auto catInfo = processCatList(inputList);
  ScalarType highType = catInfo.first;
  std::vector<BufHandle> nonEmptyInputs = catInfo.second;
  return Compute(
      "aten_cat",
      outputShape,
      outputStrides,
      [&](const std::vector<VarHandle>& axes) {
        if (nonEmptyInputs.empty()) {
          return ExprHandle(0);
        }

        int64_t dim_ = std::get<int64_t>(argDim);
        auto dim = normalizeAndCheckIndex(dim_, axes.size());
        // Promote input types.
        // Note that we need to consider all inputs, including empty - they
        // also affect the resultant dtype.

        // Now we know the final dtype, we know what inputs are non-empty,
        // and we know that there is at least one such an input. With all
        // that we construct a tensor expression performing the
        // concatenation.
        // The expression we build here is a cascading if-then-else that
        // essentially represents:
        //
        //              inp1[i, j, k]         if 0   < i < l1,
        // out[i,j,k] = inp2[i, j-l1, k]      if l1 =< i < l1 + l2,
        //              ...
        //              inpN[i, j-l_N_1, k]   if l1+l2+...l_N_1  < i
        // where l_i is the corresponding size of the i-th input.
        std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
        ExprHandle load = promoteToDtype(
            tensorOrConstant(nonEmptyInputs[0], newAxes), highType);
        auto offset = ExprHandle(nonEmptyInputs[0].node()->dim(dim));
        newAxes[dim] = newAxes[dim] - offset;

        for (size_t ii = 1; ii < nonEmptyInputs.size(); ++ii) {
          auto input = nonEmptyInputs[ii];
          load = ifThenElse(
              CompareSelect::make(axes[dim], offset, kLT),
              load,
              promoteToDtype(tensorOrConstant(input, newAxes), highType));

          offset = offset + ExprHandle(input.node()->dim(dim));
          newAxes[dim] = axes[dim] - offset;
        }

        return load;
      });
}

Tensor computeEmbedding(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  BufHandle ResultBuf("emb", outputShape, dtype);
  const BufHandle& w = std::get<BufHandle>(inputs[0]);
  const BufHandle& indices = std::get<BufHandle>(inputs[1]);

  StmtPtr s =
      ExternalCall::make(ResultBuf, "nnc_aten_embedding", {w, indices}, {});
  return Tensor(ResultBuf.node(), s);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
