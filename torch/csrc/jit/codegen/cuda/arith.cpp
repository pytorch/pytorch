#include <torch/csrc/jit/codegen/cuda/arith.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <cfloat>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// Will return a new value of type val with the DataType dtype.
Val* newScalar(ValType vtype, DataType dtype) {
  switch (vtype) {
    case (ValType::NamedScalar):
    case (ValType::Scalar):
      switch (dtype) {
        case DataType::Bool:
          return new Bool();
        case DataType::Double:
        case DataType::Float:
        case DataType::Half:
          return new Double();
        case DataType::Int:
          return new Int();
        default:
          break;
      }
    default:
      break;
  }

  TORCH_CHECK(
      false,
      "Cannot handle ValType: ",
      vtype,
      " with DataType:",
      dtype,
      " in newScalar.");
}

TensorView* newOutputTV(const std::vector<Val*>& vals, DataType dtype) {
  std::vector<TensorView*> tvs;
  for (auto val : vals)
    if (val->getValType() == ValType::TensorView)
      tvs.push_back(val->as<TensorView>());

  TORCH_CHECK(
      !tvs.empty(),
      "Tried to create new output TensorView but received empty list.");

  std::vector<IterDomain*> out_domain(
      TensorDomain::noReductions(tvs[0]->getRootDomain()).size(), nullptr);

  // For the start and stop offsets, take the maximum of input axes.
  // For now, the offsets of both start and stop are always integer
  // constant, so we can statically compute them. It is unclear
  // whether we would need to support dynamic offsetting, e.g.,
  // shifting by a dynamic offset.
  std::vector<int64_t> start_offsets(out_domain.size(), 0);
  std::vector<int64_t> stop_offsets(out_domain.size(), 0);
  std::vector<Val*> extent_vals(out_domain.size(), nullptr);
  std::vector<IterType> iter_types(out_domain.size(), IterType::Iteration);

  for (auto tv : tvs) {
    auto dom = TensorDomain::noReductions(tv->getRootDomain());
    TORCH_INTERNAL_ASSERT(
        dom.size() == out_domain.size(),
        "Invalid tensor view found while producing and output, it has ",
        dom.size(),
        " dimensions but expected ",
        out_domain.size());
    for (const auto i : c10::irange(dom.size())) {
      if (dom[i]->isBroadcast()) {
        continue;
      }
      if (extent_vals[i] == nullptr) {
        extent_vals[i] = dom[i]->extent();
        iter_types[i] = dom[i]->getIterType();
      }
      auto start_offset = dom[i]->start()->as<Int>();
      auto stop_offset = dom[i]->stopOffset()->as<Int>();
      // Currently, start is always constant
      TORCH_INTERNAL_ASSERT(
          start_offset->isConst(), "Invalid IterDomain start: ", start_offset);
      TORCH_INTERNAL_ASSERT(
          stop_offset->isConst(),
          "Invalid IterDomain stop offset: ",
          stop_offset);
      start_offsets[i] =
          std::max(start_offsets[i], start_offset->value().value());
      stop_offsets[i] = std::max(stop_offsets[i], stop_offset->value().value());
    }
  }
  for (const auto dim_i : c10::irange(out_domain.size())) {
    if (extent_vals[dim_i] != nullptr) {
      out_domain[dim_i] = new IterDomain(
          new Int(start_offsets[dim_i]),
          extent_vals[dim_i],
          new Int(stop_offsets[dim_i]),
          ParallelType::Serial,
          iter_types[dim_i]);
    } else {
      IterType itype = IterType::BroadcastWithoutStride;
      for (const auto tv : tvs) {
        auto dim = TensorDomain::noReductions(tv->getRootDomain())[dim_i];
        // If there's an unresolved bcast dim and it came from a strided dim,
        // assume output of it should be strided too
        if (dim->getIterType() == IterType::BroadcastWithStride) {
          itype = IterType::BroadcastWithStride;
          break;
        }
      }
      out_domain[dim_i] =
          new IterDomain(new Int(0), new Int(1), ParallelType::Serial, itype);
    }
  }

  return new TensorView(
      new TensorDomain(out_domain, std::vector<bool>(out_domain.size(), true)),
      dtype);
}

std::vector<Val*> maybeBroadcast(const std::vector<Val*>& vals) {
  std::vector<Val*> out_vals(vals.size(), nullptr);
  size_t n_dims = 0;
  for (auto val : vals) {
    if (val->getValType().value() == ValType::TensorView) {
      n_dims = std::max(
          n_dims,
          TensorDomain::noReductions(val->as<TensorView>()->getRootDomain())
              .size());
    }
  }

  for (const auto i : c10::irange(vals.size())) {
    if (vals[i]->getValType().value() == ValType::TensorView) {
      auto tv = vals[i]->as<TensorView>();
      size_t tv_dims = TensorDomain::noReductions(tv->getRootDomain()).size();
      if (tv_dims < n_dims) {
        std::vector<bool> bcast_flags(n_dims, false);
        for (size_t j = 0; j < n_dims - tv_dims; j++) {
          bcast_flags[j] = true;
        }
        out_vals[i] = broadcast(tv, bcast_flags);
      } else {
        out_vals[i] = vals[i];
      }
    } else {
      out_vals[i] = vals[i];
    }
  }
  return out_vals;
}

Val* newValLike(Val* val, DataType dtype) {
  TORCH_CHECK(
      dtype != DataType::Null, "Invalid datatype provided for new value.");

  const ValType vtype = val->getValType().value();

  if (vtype == ValType::TensorView)
    return newOutputTV({val}, dtype);

  return newScalar(vtype, dtype);
}

} // namespace

Val* castOp(DataType dtype, Val* v1) {
  if (v1->getDataType().value() == dtype)
    return v1;

  if (cast_func_str(std::make_pair(v1->getDataType().value(), dtype)) ==
      c10::nullopt) {
    TORCH_CHECK(
        false,
        "Illegal Cast value from  DataType: ",
        v1->getDataType().value(),
        " to DataType: ",
        dtype);
  }

  Val* out = newValLike(v1, dtype);
  new UnaryOp(UnaryOpType::Cast, out, v1);
  return out;
}

TensorView* castOp(DataType dtype, TensorView* v1) {
  return castOp(dtype, v1->as<Val>())->as<TensorView>();
}

// UNARY OPERATIONS

Val* unaryOp(UnaryOpType type, Val* v1) {
  TORCH_INTERNAL_ASSERT(
      type != UnaryOpType::Address,
      "The reference operator & is not accessible in the Fusion IR");
  Val* out = newValLike(v1, v1->getDataType().value());
  // TODO: We should add the following, but we need to go through shchedulers
  // and make sure all calls to "fusion->inputs" includes the output of RandLike
  //
  //  If rand like, there isn't a real dependency on the input value, so map it
  //  to a dummy scalar. if
  //
  // (type == UnaryOpType::RandLike) {
  //   v1 = new NamedScalar("__rnd", v1->getDataType().value());
  // }

  new UnaryOp(type, out, v1);
  return out;
}

TensorView* unaryOp(UnaryOpType type, TensorView* v1) {
  return unaryOp(type, v1->as<Val>())->as<TensorView>();
}

Val* neg(Val* v) {
  return unaryOp(UnaryOpType::Neg, v);
}

TensorView* neg(TensorView* v) {
  return unaryOp(UnaryOpType::Neg, v);
}

// BINARY OPERATIONS

namespace {
// Helper function to reduce repetitive code
template <typename T1, typename T2>
TensorView* arithOpOverloads(Val* (*func)(Val*, Val*), T1* v1, T2* v2) {
  return func(v1->template as<Val>(), v2->template as<Val>())
      ->template as<TensorView>();
}

template <typename T1, typename T2>
TensorView* arithOpOverloads(BinaryOpType type, T1* v1, T2* v2) {
  return binaryOp(type, v1->template as<Val>(), v2->template as<Val>())
      ->template as<TensorView>();
}

template <typename T1, typename T2, typename T3>
TensorView* arithOpOverloads(
    Val* (*func)(Val*, Val*, Val*),
    T1* v1,
    T2* v2,
    T3* v3) {
  auto vals = maybeBroadcast({v1, v2, v3});
  return func(
             vals[0]->template as<Val>(),
             vals[1]->template as<Val>(),
             vals[2]->template as<Val>())
      ->template as<TensorView>();
}

template <typename T1, typename T2, typename T3, typename T4>
TensorView* arithOpOverloads(
    Val* (*func)(Val*, Val*, Val*, Val*),
    T1* v1,
    T2* v2,
    T3* v3,
    T4* v4) {
  auto vals = maybeBroadcast({v1, v2, v3, v4});
  return func(
             vals[0]->template as<Val>(),
             vals[1]->template as<Val>(),
             vals[2]->template as<Val>(),
             vals[3]->template as<Val>())
      ->template as<TensorView>();
}

namespace {
enum class Category { Scalar, ZeroDimTensor, DimTensor };

inline Category getCategory(const Val* v) {
  if (v->isA<TensorView>()) {
    if (v->as<TensorView>()->nDims() > 0) {
      return Category::DimTensor;
    } else {
      return Category::ZeroDimTensor;
    }
  } else {
    return Category::Scalar;
  }
}

// replicated logic from Aten/native/TypeProperties.cpp, minus complex support
DataType getCommonType(DataType higher, DataType lower) {
  if (isFloatingPointType(higher)) {
    return higher;
  }
  if (higher == DataType::Bool || isFloatingPointType(lower)) {
    return promote_type(higher, lower);
  }
  if (higher != DataType::Null) {
    return higher;
  }
  return lower;
}
} // namespace

// Type promotion logic for binary operators
DataType getOutputType(BinaryOpType op_type, Val* v1, Val* v2) {
  DataType v1_dtype = v1->getDataType().value();
  DataType v2_dtype = v2->getDataType().value();

  const bool floating_input =
      isFloatingPointType(v1_dtype) || isFloatingPointType(v2_dtype);

  const bool integer_input =
      isIntegralType(v1_dtype) || isIntegralType(v2_dtype);

  const bool all_integer_input =
      isIntegralType(v1_dtype) && isIntegralType(v2_dtype);

  if (all_integer_input) {
    TORCH_INTERNAL_ASSERT(
        !(noFullIntegerSupport(op_type)) || (v1->isScalar() && v2->isScalar()),
        "unsupported op with all integer tensor inputs");
  }

  // Combine categories
  const auto v1_cat = getCategory(v1);
  const auto v2_cat = getCategory(v2);
  if (v1_cat != v2_cat) {
    const DataType higher = v1_cat > v2_cat ? v1_dtype : v2_dtype;
    const DataType lower = v1_cat > v2_cat ? v2_dtype : v1_dtype;
    const DataType common_type = getCommonType(higher, lower);
    v1_dtype = common_type;
    v2_dtype = common_type;
  }

  if (isIntegerOp(op_type) || (alsoBooleanOperator(op_type) && integer_input)) {
    // If integer op or maybe bool op with integer inputs meaning binary op
    if (integer_input && all_integer_input) {
      return promote_type(v1_dtype, v2_dtype);
    } else if (integer_input && !all_integer_input) {
      TORCH_CHECK(
          !floating_input,
          "Operator ",
          op_type,
          " not supported with floating point inputs.");
      return isIntegralType(v1_dtype) ? v1_dtype : v2_dtype;
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Currently no support for float inputs to int operations. ",
          "Inputs should be manually casted first.");
    }
  } else if (isLogicalOp(op_type)) {
    return DataType::Bool;
  } else if (alsoBooleanOperator(op_type)) {
    // If boolean op that can't have floating inputs (& or |)
    TORCH_CHECK(
        !floating_input,
        "Operator ",
        op_type,
        " not supported with floating point inputs.");
    return DataType::Bool;
  } else {
    // Otherwise do normal type promotion
    return promote_type(v1_dtype, v2_dtype);
  }
}

} // namespace

TORCH_CUDA_CU_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2) {
  const auto out_dtype = getOutputType(type, v1, v2);
  const auto out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  auto vals = maybeBroadcast({v1, v2});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = newOutputTV(vals, out_dtype);
  } else {
    out = newScalar(out_vtype, out_dtype);
  }
  new BinaryOp(type, out, vals[0], vals[1]);
  return out;
}

TensorView* binaryOp(BinaryOpType type, TensorView* v1, Val* v2) {
  return arithOpOverloads(type, v1, v2);
}

TensorView* binaryOp(BinaryOpType type, Val* v1, TensorView* v2) {
  return arithOpOverloads(type, v1, v2);
}

TensorView* binaryOp(BinaryOpType type, TensorView* v1, TensorView* v2) {
  return arithOpOverloads(type, v1, v2);
}

// add
Val* add(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Add, v1, v2);
}
TensorView* add(TensorView* v1, Val* v2) {
  return arithOpOverloads(add, v1, v2);
}
TensorView* add(Val* v1, TensorView* v2) {
  return arithOpOverloads(add, v1, v2);
}
TensorView* add(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(add, v1, v2);
}

// sub
Val* sub(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Sub, v1, v2);
}
TensorView* sub(TensorView* v1, Val* v2) {
  return arithOpOverloads(sub, v1, v2);
}
TensorView* sub(Val* v1, TensorView* v2) {
  return arithOpOverloads(sub, v1, v2);
}
TensorView* sub(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(sub, v1, v2);
}

// mul
Val* mul(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Mul, v1, v2);
}
TensorView* mul(TensorView* v1, Val* v2) {
  return arithOpOverloads(mul, v1, v2);
}
TensorView* mul(Val* v1, TensorView* v2) {
  return arithOpOverloads(mul, v1, v2);
}
TensorView* mul(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(mul, v1, v2);
}

// div
Val* div(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Div, v1, v2);
}
TensorView* div(TensorView* v1, Val* v2) {
  return arithOpOverloads(div, v1, v2);
}
TensorView* div(Val* v1, TensorView* v2) {
  return arithOpOverloads(div, v1, v2);
}
TensorView* div(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(div, v1, v2);
}

// mod
Val* mod(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Mod, v1, v2);
}
TensorView* mod(TensorView* v1, Val* v2) {
  return arithOpOverloads(mod, v1, v2);
}
TensorView* mod(Val* v1, TensorView* v2) {
  return arithOpOverloads(mod, v1, v2);
}
TensorView* mod(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(mod, v1, v2);
}

// lt
Val* lt(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::LT, v1, v2);
}
TensorView* lt(TensorView* v1, Val* v2) {
  return arithOpOverloads(lt, v1, v2);
}
TensorView* lt(Val* v1, TensorView* v2) {
  return arithOpOverloads(lt, v1, v2);
}
TensorView* lt(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(lt, v1, v2);
}

// gt
Val* gt(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::GT, v1, v2);
}
TensorView* gt(TensorView* v1, Val* v2) {
  return arithOpOverloads(gt, v1, v2);
}
TensorView* gt(Val* v1, TensorView* v2) {
  return arithOpOverloads(gt, v1, v2);
}
TensorView* gt(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(gt, v1, v2);
}
// eq
Val* eq(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::Eq, v1, v2);
}
TensorView* eq(TensorView* v1, Val* v2) {
  return arithOpOverloads(eq, v1, v2);
}
TensorView* eq(Val* v1, TensorView* v2) {
  return arithOpOverloads(eq, v1, v2);
}
TensorView* eq(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(eq, v1, v2);
}

// ceilDiv
Val* ceilDiv(Val* v1, Val* v2) {
  return binaryOp(BinaryOpType::CeilDiv, v1, v2);
}
TensorView* ceilDiv(TensorView* v1, Val* v2) {
  return arithOpOverloads(ceilDiv, v1, v2);
}
TensorView* ceilDiv(Val* v1, TensorView* v2) {
  return arithOpOverloads(ceilDiv, v1, v2);
}
TensorView* ceilDiv(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(ceilDiv, v1, v2);
}

// andOp
Val* andOp(Val* v1, Val* v2) {
  TORCH_CHECK(
      !isFloatingPointType(v1->getDataType().value()),
      "Input1 should not be a floating point type, but received: ",
      v1->getDataType().value());
  TORCH_CHECK(
      !isFloatingPointType(v2->getDataType().value()),
      "Input2 should not be a floating point type, but received: ",
      v2->getDataType().value());
  return binaryOp(BinaryOpType::And, v1, v2);
}
TensorView* andOp(TensorView* v1, Val* v2) {
  return arithOpOverloads(andOp, v1, v2);
}
TensorView* andOp(Val* v1, TensorView* v2) {
  return arithOpOverloads(andOp, v1, v2);
}
TensorView* andOp(TensorView* v1, TensorView* v2) {
  return arithOpOverloads(andOp, v1, v2);
}

// REDUCTION OPERATIONS

// TODO: How do we adjust this so we can reduce to a single scalar value?
static TensorView* newForReduction(
    TensorView* tv,
    const std::vector<unsigned int>& axes,
    DataType data_type = DataType::Null) {
  auto orig_domain = TensorDomain::noReductions(tv->getRootDomain());
  std::set<unsigned int> axes_set(axes.begin(), axes.end());

  std::vector<IterDomain*> new_domain;

  TORCH_INTERNAL_ASSERT(
      !axes_set.empty(),
      "Asked for ouput of reduction, but no reduction axis provided.");

  TORCH_INTERNAL_ASSERT(
      (*(axes_set.rbegin())) < orig_domain.size(),
      "Error setting up reduction, reduction axis is outside nDims. Keep in mind reductions are relative to root domains, not modified views.");

  auto axis_iter = axes_set.begin();
  for (const auto dim : c10::irange(orig_domain.size())) {
    bool isReduction = false;
    if (axis_iter != axes_set.end() && *axis_iter == dim) {
      isReduction = true;
      axis_iter++;
    }

    const IterDomain* id = orig_domain[dim];

    TORCH_CHECK(
        !(isReduction && id->isBroadcast() && !id->isImplicitBroadcast()),
        "Cannot reduce an axis that is marked as broadcasted as it has an undetermined size. Tried to reduce ID = ",
        id,
        " of tensor ",
        tv);

    new_domain.push_back(new IterDomain(
        id->start(),
        id->extent(),
        id->stopOffset(),
        ParallelType::Serial,
        isReduction ? IterType::Reduction : id->getIterType()));
  }

  TensorDomain* td =
      new TensorDomain(new_domain, std::vector<bool>(new_domain.size(), true));

  data_type =
      data_type == DataType::Null ? tv->getDataType().value() : data_type;
  return new TensorView(td, data_type);
}

TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* tv,
    bool keep_dim /*=false*/) {
  TORCH_CHECK(
      init->isConstScalar(),
      "Cannot create a reduction operation where the initial value is not a const scalar.");

  TORCH_CHECK(
      TensorDomain::sameAs(tv->getRootDomain(), tv->domain()->domain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. Please set reductions before calling split/merge/computeAt.");

  TORCH_CHECK(tv->nDims() > 0, "Tried to reduce a 0-dim tensor");

  TORCH_CHECK(axes.size() > 0, "No reduction axis specified");

  std::vector<unsigned int> uint_axes;
  for (int axis : axes) {
    if (axis < 0)
      axis += int(tv->nDims());

    TORCH_CHECK(
        axis >= 0 && (unsigned int)axis < tv->nDims(),
        "Reduction on invalid axis, recieved: ",
        axis,
        " however tensor view only has ",
        tv->nDims(),
        " dims.");

    uint_axes.push_back((unsigned int)axis);
  }

  TensorView* out = newForReduction(tv, uint_axes);
  const auto out_type = out->getDataType().value();
  const auto init_type = init->getDataType().value();
  TORCH_CHECK(
      (isFloatingPointType(out_type) && isFloatingPointType(init_type)) ||
          (isIntegralType(out_type) && isIntegralType(init_type)) ||
          (out_type == DataType::Bool && init_type == DataType::Bool),
      "Types should match for reduction ops but received: ",
      out_type,
      " and ",
      init_type);
  new ReductionOp(reduction_op_type, init, out, tv);

  if (keep_dim) {
    auto tv_root = TensorDomain::noReductions(tv->getRootDomain());
    std::vector<bool> is_broadcast(tv_root.size(), false);
    for (int axis : axes) {
      is_broadcast[axis] = true;
    }

    out = broadcast(out, is_broadcast);
  }
  return out;
}

TensorView* sum(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/) {
  Val* init = nullptr;
  auto dtype = v1->getDataType().value();
  if (isFloatingPointType(dtype)) {
    init = new Double(0.0);
  } else if (isIntegralType(dtype)) {
    init = new Int(0);
  } else {
    TORCH_CHECK(
        false,
        "Could not generate a sum op for tensor with type: ",
        v1->getDataType().value());
  }

  return reductionOp(BinaryOpType::Add, axes, init, v1, keep_dim);
}

TensorView* max(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/) {
  Val* init = nullptr;
  switch (v1->getDataType().value()) {
    case (DataType::Double):
      init = new Double(DBL_MIN);
      break;
    case (DataType::Float):
      init = new Double(FLT_MIN);
      break;
    case (DataType::Int):
      init = new Int(INT_MIN);
      break;
    default:
      TORCH_CHECK(
          false,
          "Could not generate a max op for tensor with type: ",
          v1->getDataType().value());
  }

  return reductionOp(BinaryOpType::Max, axes, init, v1, keep_dim);
}

TensorView* min(
    TensorView* v1,
    const std::vector<int>& axes,
    bool keep_dim /*=false*/) {
  Val* init = nullptr;
  switch (v1->getDataType().value()) {
    case (DataType::Double):
      init = new Double(DBL_MAX);
      break;
    case (DataType::Float):
      init = new Double(FLT_MAX);
      break;
    case (DataType::Int):
      init = new Int(INT_MAX);
      break;
    default:
      TORCH_CHECK(
          false,
          "Could not generate a min op for tensor with type: ",
          v1->getDataType().value());
  }

  return reductionOp(BinaryOpType::Min, axes, init, v1, keep_dim);
}

TensorView* broadcast(
    TensorView* inp,
    const std::vector<bool>& is_broadcast_dim) {
  auto nBCastDims = is_broadcast_dim.size();
  // Validate is_broadcast_dim
  unsigned int n_broadcasts = 0;
  for (auto ent : is_broadcast_dim)
    if (ent)
      n_broadcasts++;
  TORCH_CHECK(
      nBCastDims - n_broadcasts ==
          TensorDomain::noReductions(inp->getRootDomain()).size(),
      "Invalid broadcast, number of false entries in is_broadcast_dim expected to be ",
      TensorDomain::noReductions(inp->getRootDomain()).size(),
      " but received ",
      nBCastDims - n_broadcasts);

  if (n_broadcasts == 0) {
    auto identity = unaryOp(UnaryOpType::Set, inp);
    TORCH_INTERNAL_ASSERT(
        identity->getValType().value() == ValType::TensorView,
        "Expected identity op, but didn't get a TensorView back.");
    return identity->as<TensorView>();
  }

  std::vector<IterDomain*> out_domain;
  // Don't propagate reduction IDs through arith ops.
  auto inp_domain = TensorDomain::noReductions(inp->getRootDomain());
  size_t iinp = 0, ibdim = 0;
  while (ibdim < is_broadcast_dim.size()) {
    if (is_broadcast_dim[ibdim]) {
      out_domain.push_back(new IterDomain(
          new Int(0),
          new Int(1),
          ParallelType::Serial,
          IterType::BroadcastWithoutStride));
    } else {
      out_domain.push_back(inp_domain[iinp]->clone());
      iinp++;
    }
    ibdim++;
  }

  TensorView* out_tensor = new TensorView(
      new TensorDomain(out_domain, std::vector<bool>(out_domain.size(), true)),
      inp->getDataType().value());
  new BroadcastOp(out_tensor, inp, is_broadcast_dim);
  return out_tensor;
}

WelfordResult Welford(
    TensorView* tv,
    const std::vector<int>& axes,
    TensorView* init_avg,
    TensorView* init_var,
    Int* init_N) {
  TORCH_CHECK(
      TensorDomain::sameAs(tv->getRootDomain(), tv->domain()->domain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. Please set reductions before calling split/merge/computeAt.");

  TORCH_CHECK(tv->nDims() > 0, "Tried to reduce a 0-dim tensor");
  TORCH_CHECK(axes.size() > 0, "No reduction axis specified");

  // Initial values for welford op are tensors, so their dims have to match the
  // output dim,
  // i.e. original_dims - dims_to_be_reduced
  Val* init_avg_val = nullptr;
  Val* init_var_val = nullptr;
  if (!init_N->isZeroInt()) {
    TORCH_CHECK(
        init_avg != nullptr && init_var != nullptr && init_N != nullptr,
        "welford op: all init values need to be provided");
    TORCH_CHECK(
        (axes.size() + init_avg->getRootDomain().size()) ==
            tv->getRootDomain().size(),
        "welford op: initial tensor mismatch");
    TORCH_CHECK(
        (axes.size() + init_var->getRootDomain().size()) ==
            tv->getRootDomain().size(),
        "welford op: initial tensor mismatch");
    init_avg_val = init_avg;
    init_var_val = init_var;
  } else {
    init_avg_val = new Double(0);
    init_var_val = new Double(0);
  }

  // Check and collect reduction axes
  std::vector<unsigned int> uint_axes;
  for (int axis : axes) {
    if (axis < 0)
      axis += int(tv->nDims());

    TORCH_CHECK(
        axis >= 0 && (unsigned int)axis < tv->nDims(),
        "Reduction on invalid axis, recieved: ",
        axis,
        " however tensor view only has ",
        tv->nDims(),
        " dims.");

    uint_axes.push_back((unsigned int)axis);
  }

  // Create tensor outputs
  TensorView* out_avg = newForReduction(tv, uint_axes);
  TensorView* out_var = newForReduction(tv, uint_axes);
  TensorView* out_N = newForReduction(tv, uint_axes, DataType::Int);

  new WelfordOp(
      out_avg,
      out_var,
      out_N, /*out var/avg/count */
      init_avg_val,
      init_var_val,
      init_N, /*init var/avg/count */
      tv,
      nullptr,
      new Int(1)); /*in var/avg/count */

  return WelfordResult(out_avg, out_var, out_N);
}

WelfordResult::WelfordResult(
    TensorView* in_avg,
    TensorView* in_var_sum,
    TensorView* in_n)
    : avg(in_avg), var_sum(in_var_sum), n(in_n) {
  TORCH_INTERNAL_ASSERT(avg->definition()->sameAs(var_sum->definition()));
  TORCH_INTERNAL_ASSERT(avg->definition()->sameAs(n->definition()));
}

WelfordResult WelfordResult::rFactor(const std::vector<int>& axes) {
  auto o_tv = avg->definition()->as<WelfordOp>()->out()->as<TensorView>();
  return o_tv->rFactor(axes, avg, var_sum, n);
}

TensorView* transpose(
    TensorView* inp,
    const std::unordered_map<int, int>& old2new) {
  auto inp_domain = TensorDomain::noReductions(inp->getRootDomain());
  std::vector<IterDomain*> out_domain(inp_domain.size());

  auto new2old = ir_utils::normalizeOld2New(old2new, inp_domain.size());

  for (size_t i = 0; i < out_domain.size(); ++i) {
    auto in_id = inp_domain[new2old[i]];
    out_domain[i] = in_id->clone();
  }

  TensorView* out_tensor = new TensorView(
      new TensorDomain(out_domain, std::vector<bool>(out_domain.size(), true)),
      inp->getDataType().value());
  new TransposeOp(out_tensor, inp, new2old);
  return out_tensor;
}

// COMPOUND OPERATIONS

// add_alpha
Val* add_alpha(Val* v1, Val* v2, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  auto vals = maybeBroadcast({v1, v2, s});
  Val* intrm = binaryOp(BinaryOpType::Mul, vals[1], vals[2]);
  return binaryOp(BinaryOpType::Add, vals[0], intrm);
}
TensorView* add_alpha(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
TensorView* add_alpha(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
TensorView* add_alpha(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(add_alpha, v1, v2, v3);
}
// sub_alpha
Val* sub_alpha(Val* v1, Val* v2, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  auto vals = maybeBroadcast({v1, v2, s});
  Val* intrm = binaryOp(BinaryOpType::Mul, vals[1], vals[2]);
  return binaryOp(BinaryOpType::Sub, vals[0], intrm);
}
TensorView* sub_alpha(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
TensorView* sub_alpha(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
TensorView* sub_alpha(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(sub_alpha, v1, v2, v3);
}
// lerp
TORCH_CUDA_CU_API Val* lerp(Val* start, Val* end, Val* weight) {
  auto vals = maybeBroadcast({start, end, weight});
  Val* intrm1 = binaryOp(BinaryOpType::Sub, vals[1], vals[0]);
  Val* intrm2 = binaryOp(BinaryOpType::Mul, vals[2], intrm1);
  return binaryOp(BinaryOpType::Add, vals[0], intrm2);
}
TensorView* lerp(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(Val* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(TensorView* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(Val* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
TensorView* lerp(TensorView* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(lerp, v1, v2, v3);
}
// addcmul
Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s) {
  TORCH_CHECK(
      s->getValType().value() == ValType::Scalar,
      "Alpha value should be a Scalar Valtype and not ",
      s->getValType().value());

  auto vals = maybeBroadcast({v1, v2, v3, s});
  Val* intrm1 = binaryOp(BinaryOpType::Mul, vals[2], vals[3]);
  Val* intrm2 = binaryOp(BinaryOpType::Mul, vals[1], intrm1);
  return binaryOp(BinaryOpType::Add, vals[0], intrm2);
}
TensorView* addcmul(TensorView* v1, Val* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(Val* v1, TensorView* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(Val* v1, Val* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(TensorView* v1, TensorView* v2, Val* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(TensorView* v1, Val* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(Val* v1, TensorView* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}
TensorView* addcmul(TensorView* v1, TensorView* v2, TensorView* v3, Val* v4) {
  return arithOpOverloads(addcmul, v1, v2, v3, v4);
}

// TERNARY OPERATIONS
// where (c ? v1 : v2)
Val* where(Val* c, Val* v1, Val* v2) {
  TORCH_CHECK(
      c->getDataType().value() == DataType::Bool,
      "Condition should be of DataType Bool, not ",
      c->getDataType().value());

  // Not actually an add, but need to send a binary op to get output type
  auto out_dtype = getOutputType(BinaryOpType::Add, v1, v2);
  auto out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  auto vals = maybeBroadcast({c, v1, v2});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = newOutputTV(vals, out_dtype);
  } else {
    out = newScalar(out_vtype, out_dtype);
  }
  new TernaryOp(TernaryOpType::Where, out, vals[0], vals[1], vals[2]);
  return out;
}

TensorView* where(TensorView* v1, Val* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(Val* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(Val* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(TensorView* v1, TensorView* v2, Val* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(TensorView* v1, Val* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(Val* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}
TensorView* where(TensorView* v1, TensorView* v2, TensorView* v3) {
  return arithOpOverloads(where, v1, v2, v3);
}

// TERNARY OPERATIONS

Val* threshold(Val* in, Val* thresh, Val* value) {
  const auto in_type = in->getDataType().value();
  const auto thresh_type = thresh->getDataType().value();
  const auto value_type = value->getDataType().value();
  if (isFloatingPointType(in_type)) {
    TORCH_CHECK(
        isFloatingPointType(thresh_type) && isFloatingPointType(value_type),
        "All input DataType values should match the input type ",
        in_type,
        " vs ",
        thresh_type,
        " and ",
        value_type);
  } else if (isIntegralType(in_type)) {
    TORCH_CHECK(
        isIntegralType(thresh_type) && isIntegralType(value_type),
        "All input DataType values should match the input ",
        in_type,
        " vs ",
        thresh_type,
        " and ",
        value_type);
  }
  TORCH_CHECK(
      (thresh->getValType().value() == ValType::Scalar ||
       thresh->getValType().value() == ValType::NamedScalar) &&
          (value->getValType().value() == ValType::Scalar ||
           value->getValType().value() == ValType::NamedScalar),
      "For Threshold operation: Thresh and Value values should be Scalars.");

  Val* out = newValLike(in, in_type);

  new TernaryOp(TernaryOpType::Threshold, out, in, thresh, value);
  return out;
}

TensorView* threshold(TensorView* in, Val* thresh, Val* value) {
  return threshold(in->as<Val>(), thresh, value)->as<TensorView>();
}

Val* clamp(Val* in, Val* min_val, Val* max_val) {
  const auto in_type = in->getDataType().value();
  const auto min_type = min_val->getDataType().value();
  const auto max_type = max_val->getDataType().value();
  if (isFloatingPointType(in_type)) {
    TORCH_CHECK(
        isFloatingPointType(min_type) && isFloatingPointType(max_type),
        "All input DataType values should match the input type ",
        in_type,
        " vs ",
        min_type,
        " and ",
        max_type);
  } else if (isIntegralType(in_type)) {
    TORCH_CHECK(
        isIntegralType(min_type) && isIntegralType(max_type),
        "All input DataType values should match the input ",
        in_type,
        " vs ",
        min_type,
        " and ",
        max_type);
  }
  TORCH_CHECK(
      (min_val->getValType().value() == ValType::Scalar ||
       min_val->getValType().value() == ValType::NamedScalar) &&
          (max_val->getValType().value() == ValType::Scalar ||
           max_val->getValType().value() == ValType::NamedScalar),
      "For Threshold operation: Thresh and Value values should be Scalars.");

  Val* out = newValLike(in, in_type);

  new TernaryOp(TernaryOpType::Clamp, out, in, min_val, max_val);
  return out;
}

TensorView* clamp(TensorView* in, Val* min_val, Val* max_val) {
  return clamp(in->as<Val>(), min_val, max_val)->as<TensorView>();
}

// sum_to operator

TensorView* sum_to(TensorView* in, const std::vector<Int*>& sum_to_size) {
  const auto& root = TensorDomain::noReductions(in->getRootDomain());

  TORCH_CHECK(
      root.size() >= sum_to_size.size(),
      "sum_to: Error trying to reduce",
      in,
      "into a shape of size",
      sum_to_size.size());

  // If no reduction is needed sum_to returns the input tv
  TensorView* out = in;

  const int64_t leading_dims = root.size() - sum_to_size.size();

  // Generate reduction axes for leading dims
  std::vector<int> reduce_dims(leading_dims);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);

  // Generate reduction axes for dims within sum_to_size
  std::vector<bool> inner_red_dims(sum_to_size.size(), false);
  bool reduction_within_shape = false;

  // Reduce rest of the dims with keep_dim
  for (int i = leading_dims; i < int(root.size()); i++) {
    if (sum_to_size[i - leading_dims]->isOneInt() &&
        !root[i]->extent()->isOneInt()) {
      inner_red_dims[i - leading_dims] = true;
      reduce_dims.push_back(i);
      reduction_within_shape = true;
    }
  }

  // Reduction step
  if (!reduce_dims.empty()) {
    out = sum(in, reduce_dims);
  }

  // Broadcast back reduced dims within shape
  if (reduction_within_shape) {
    out = broadcast(out, inner_red_dims);
  }

  return out;
}

TensorView* sum_to(TensorView* in, const std::vector<int64_t>& sum_to_size) {
  const auto& root = TensorDomain::noReductions(in->getRootDomain());

  TORCH_CHECK(
      root.size() >= sum_to_size.size(),
      "sum_to: Error trying to reduce",
      in,
      "into a shape of size",
      sum_to_size.size());

  // If no reduction is needed sum_to returns the input tv
  TensorView* out = in;

  const int64_t leading_dims = root.size() - sum_to_size.size();

  // Generate reduction axes for leading dims
  std::vector<int> reduce_dims(leading_dims);
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);

  // Generate reduction axes for dims within sum_to_size
  std::vector<bool> inner_red_dims(sum_to_size.size(), false);
  bool reduction_within_shape = false;

  // Reduce rest of the dims with keep_dim
  for (int i = leading_dims; i < int(root.size()); i++) {
    if (sum_to_size[i - leading_dims] == 1 && !root[i]->extent()->isOneInt()) {
      inner_red_dims[i - leading_dims] = true;
      reduce_dims.push_back(i);
      reduction_within_shape = true;
    }
  }

  // Reduction step
  if (!reduce_dims.empty()) {
    out = sum(in, reduce_dims);
  }

  // Broadcast back reduced dims within shape
  if (reduction_within_shape) {
    out = broadcast(out, inner_red_dims);
  }

  return out;
}

TensorView* shift(TensorView* inp, const std::vector<int>& offsets, bool pad) {
  TORCH_CHECK(
      TensorDomain::noReductions(inp->getRootDomain()).size() == offsets.size(),
      "Invalid shift offsets, number of entries in offsets expected to be ",
      TensorDomain::noReductions(inp->getRootDomain()).size(),
      " but received ",
      offsets.size());

  TensorView* out = nullptr;

  if (pad) {
    out = newValLike(inp, inp->getDataType().value())->as<TensorView>();
  } else {
    auto inp_dom = TensorDomain::noReductions(inp->getRootDomain());
    const auto ndims = inp_dom.size();
    std::vector<IterDomain*> out_dom;
    for (size_t i = 0; i < ndims; ++i) {
      const auto inp_axis = inp_dom[i];
      const auto offset = offsets[i];
      if (offset == 0) {
        out_dom.push_back(inp_axis->clone());
        continue;
      }

      Int* current_start_offset = dynamic_cast<Int*>(inp_axis->start());
      TORCH_INTERNAL_ASSERT(
          current_start_offset != nullptr && current_start_offset->isConst(),
          "Invalid IterDomain start value:",
          current_start_offset);

      Int* current_stop_offset = dynamic_cast<Int*>(inp_axis->stopOffset());
      TORCH_INTERNAL_ASSERT(
          current_stop_offset != nullptr && current_stop_offset->isConst(),
          "Invalid IterDomain stop offset value:",
          current_stop_offset);

      const auto cur_start_offset_value = current_start_offset->value().value();
      const auto cur_stop_offset_value = current_stop_offset->value().value();

      Val* out_start_offset = nullptr;
      Val* out_stop_offset = nullptr;

      if (offset > 0) {
        // shift to right; extent remains the same, start and stop
        // positions are moved right
        out_start_offset = new Int(cur_start_offset_value + offset);
        out_stop_offset =
            new Int(std::max(cur_stop_offset_value - offset, int64_t(0)));
      } else {
        // shift to left; extent remains the same, start and stop
        // positions are moved left
        out_start_offset =
            new Int(std::max(cur_start_offset_value + offset, int64_t(0)));
        out_stop_offset = new Int(cur_stop_offset_value - offset);
      }

      out_dom.push_back(new IterDomain(
          out_start_offset,
          inp_axis->extent(),
          out_stop_offset,
          ParallelType::Serial,
          inp_axis->getIterType()));
    }

    out = new TensorView(
        new TensorDomain(out_dom, std::vector<bool>(out_dom.size(), true)),
        inp->getDataType().value());
  }

  new ShiftOp(out, inp, offsets, pad);
  return out;
}

namespace {
std::vector<Int*> convertToIntVector(const std::vector<int>& x) {
  std::vector<Int*> converted;
  std::transform(x.begin(), x.end(), std::back_inserter(converted), [](int x) {
    return new Int(x);
  });
  return converted;
}
} // namespace

TensorView* gather(
    TensorView* inp,
    const std::vector<int>& window_shape,
    const std::vector<std::vector<int>>& pad_width) {
  std::vector<Int*> window_shape_int = convertToIntVector(window_shape);
  std::vector<std::vector<Int*>> pad_width_int;
  std::transform(
      pad_width.begin(),
      pad_width.end(),
      std::back_inserter(pad_width_int),
      [](const std::vector<int>& x) { return convertToIntVector(x); });
  return gather(inp, window_shape_int, pad_width_int);
}

TensorView* gather(
    TensorView* inp,
    const std::vector<Int*>& window_shape,
    const std::vector<std::vector<Int*>>& pad_width) {
  auto inp_dom = TensorDomain::noReductions(inp->getRootDomain());
  const auto ndims = inp_dom.size();

  TORCH_CHECK(
      ndims == window_shape.size(),
      "Invalid window shape: number of entries expected to be ",
      ndims,
      " but received ",
      window_shape.size());

  TORCH_CHECK(
      ndims == pad_width.size(),
      "Invalid pad width: number of entries expected to be ",
      ndims,
      " but received ",
      pad_width.size());

  std::for_each(pad_width.begin(), pad_width.end(), [](const auto& p) {
    TORCH_CHECK(
        p.size() == 2,
        "Each entry of pad_width must have two non-negative integers.");
  });

  std::vector<IterDomain*> out_dom;
  std::vector<IterDomain*> out_gather_dom;

  for (size_t i = 0; i < ndims; ++i) {
    const auto inp_axis = inp_dom[i];
    const auto window_dim = window_shape[i];
    const auto pad_left = pad_width[i][0];
    const auto pad_right = pad_width[i][1];
    TORCH_INTERNAL_ASSERT(inp_axis->start()->isZeroInt());
    Val* out_axis_dim = nullptr;
    if (window_dim->isConst() && pad_left->isConst() && pad_right->isConst()) {
      const int64_t extent_adjustment =
          -(-window_dim->value().value() + 1 + pad_left->value().value() +
            pad_right->value().value());
      out_axis_dim = extent_adjustment == 0
          ? inp_axis->extent()
          : sub(inp_axis->extent(), new Int(extent_adjustment));
    } else {
      out_axis_dim =
          add(add(sub(inp_axis->extent(), window_dim), new Int(1)),
              add(pad_left, pad_right));
    }
    out_dom.push_back(new IterDomain(
        new Int(0),
        out_axis_dim,
        ParallelType::Serial,
        inp_axis->getIterType()));
    // create a new axis for the gathered domain
    out_gather_dom.push_back(new IterDomain(
        new Int(0), window_dim, ParallelType::Serial, IterType::Gather));
  }

  out_dom.insert(out_dom.end(), out_gather_dom.begin(), out_gather_dom.end());

  auto out = new TensorView(
      new TensorDomain(out_dom, std::vector<bool>(out_dom.size(), true)),
      inp->getDataType().value());

  new GatherOp(out, inp, window_shape, pad_width);
  return out;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
