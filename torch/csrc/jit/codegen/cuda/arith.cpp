#include <torch/csrc/jit/codegen/cuda/arith.h>

#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <torch/csrc/jit/codegen/cuda/type_promotion.h>
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
          return IrBuilder::create<Bool>();
        case DataType::Double:
        case DataType::Float:
        case DataType::Half:
        case DataType::BFloat16:
          return IrBuilder::create<Double>();
        case DataType::Int32:
        case DataType::Int:
          return IrBuilder::create<Int>();
        case DataType::ComplexFloat:
        case DataType::ComplexDouble:
          return IrBuilder::create<ComplexDouble>();
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
      TensorDomain::noReductions(tvs[0]->getMaybeRFactorDomain()).size(),
      nullptr);

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
    auto dom = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
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
      out_domain[dim_i] = IrBuilder::create<IterDomain>(
          IrBuilder::create<Int>(start_offsets[dim_i]),
          extent_vals[dim_i],
          IrBuilder::create<Int>(stop_offsets[dim_i]),
          ParallelType::Serial,
          iter_types[dim_i]);
    } else {
      IterType itype = IterType::BroadcastWithoutStride;
      for (const auto tv : tvs) {
        auto dim =
            TensorDomain::noReductions(tv->getMaybeRFactorDomain())[dim_i];
        // If there's an unresolved bcast dim and it came from a strided dim,
        // assume output of it should be strided too
        if (dim->getIterType() == IterType::BroadcastWithStride) {
          itype = IterType::BroadcastWithStride;
          break;
        }
      }
      out_domain[dim_i] = IrBuilder::create<IterDomain>(
          FusionGuard::getCurFusion()->zeroVal(),
          FusionGuard::getCurFusion()->oneVal(),
          ParallelType::Serial,
          itype);
    }
  }

  return IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, std::vector<bool>(out_domain.size(), true)),
      dtype);
}

std::vector<Val*> maybeBroadcast(const std::vector<Val*>& vals) {
  std::vector<Val*> out_vals(vals.size(), nullptr);
  size_t n_dims = 0;
  for (auto val : vals) {
    if (val->getValType().value() == ValType::TensorView) {
      n_dims = std::max(
          n_dims,
          TensorDomain::noReductions(
              val->as<TensorView>()->getMaybeRFactorDomain())
              .size());
    }
  }

  for (const auto i : c10::irange(vals.size())) {
    if (vals[i]->getValType().value() == ValType::TensorView) {
      auto tv = vals[i]->as<TensorView>();
      size_t tv_dims =
          TensorDomain::noReductions(tv->getMaybeRFactorDomain()).size();
      if (tv_dims < n_dims) {
        std::vector<bool> bcast_flags(n_dims, false);
        for (const auto j : c10::irange(n_dims - tv_dims)) {
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
  if (v1->getDataType().value() == dtype) {
    return set(v1);
  }

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
  IrBuilder::create<UnaryOp>(UnaryOpType::Cast, out, v1);
  return out;
}

TensorView* castOp(DataType dtype, TensorView* v1) {
  return castOp(dtype, v1->as<Val>())->as<TensorView>();
}

Val* unaryOp(UnaryOpType type, Val* v1) {
  TORCH_INTERNAL_ASSERT(
      type != UnaryOpType::Address,
      "The reference operator & is not accessible in the Fusion IR");

  // TODO: We should add the following, but we need to go through schedulers
  // and make sure all calls to "fusion->inputs" includes the output of RandLike
  //
  //  If rand like, there isn't a real dependency on the input value, so map it
  //  to a dummy scalar. if
  //
  // (type == UnaryOpType::RandLike) {
  //   v1 = new NamedScalar("__rnd", v1->getDataType().value());
  // }

  Val* out = newValLike(v1, v1->getDataType().value());
  IrBuilder::create<UnaryOp>(type, out, v1);
  return out;
}

TensorView* unaryOp(UnaryOpType type, TensorView* v1) {
  return unaryOp(type, v1->as<Val>())->as<TensorView>();
}

Val* unaryOp(UnaryOpType type, Val* v1, const TypePromotionConfig& config) {
  auto casted_v1 = promoteValues(config, {v1}).front();
  return unaryOp(type, casted_v1);
}

TensorView* unaryOp(
    UnaryOpType type,
    TensorView* v1,
    const TypePromotionConfig& config) {
  auto casted_v1 = promoteValues(config, {v1}).front();
  return unaryOp(type, casted_v1)->as<TensorView>();
}

// UNARY OPERATIONS

#define NVFUSER_DEFINE_UNARY_OP(op_name, op_type) \
  Val* op_name(Val* v) {                          \
    return unaryOp(UnaryOpType::op_type, v);      \
  }                                               \
  TensorView* op_name(TensorView* tv) {           \
    return unaryOp(UnaryOpType::op_type, tv);     \
  }

NVFUSER_DEFINE_UNARY_OP(set, Set)
NVFUSER_DEFINE_UNARY_OP(randlike, RandLike)
NVFUSER_DEFINE_UNARY_OP(notOp, Not)
NVFUSER_DEFINE_UNARY_OP(ceil, Ceil)
NVFUSER_DEFINE_UNARY_OP(floor, Floor)
NVFUSER_DEFINE_UNARY_OP(frac, Frac)
NVFUSER_DEFINE_UNARY_OP(neg, Neg)
NVFUSER_DEFINE_UNARY_OP(relu, Relu)
NVFUSER_DEFINE_UNARY_OP(round, Round)
NVFUSER_DEFINE_UNARY_OP(silu, Silu)
NVFUSER_DEFINE_UNARY_OP(trunc, Trunc)
#undef NVFUSER_DEFINE_UNARY_OP

// The output of abs(complex_tensor) are real numbers
Val* abs(Val* v) {
  if (v->getDataType() == DataType::ComplexDouble) {
    Val* out = newValLike(v, DataType::Double);
    IrBuilder::create<UnaryOp>(UnaryOpType::Abs, out, v);
    return out;
  }
  if (v->getDataType() == DataType::ComplexFloat) {
    Val* out = newValLike(v, DataType::Float);
    IrBuilder::create<UnaryOp>(UnaryOpType::Abs, out, v);
    return out;
  }
  return unaryOp(UnaryOpType::Abs, v);
}

TensorView* abs(TensorView* tv) {
  return abs(tv->as<Val>())->as<TensorView>();
}

// UNARY FLOAT CAST OPERATIONS

#define NVFUSER_DEFINE_UNARY_FLOAT_OP(op_name, op_type)                       \
  Val* op_name(Val* v) {                                                      \
    return unaryOp(UnaryOpType::op_type, v, TypePromotion::float_op_config);  \
  }                                                                           \
  TensorView* op_name(TensorView* tv) {                                       \
    return unaryOp(UnaryOpType::op_type, tv, TypePromotion::float_op_config); \
  }

NVFUSER_DEFINE_UNARY_FLOAT_OP(acos, Acos)
NVFUSER_DEFINE_UNARY_FLOAT_OP(asin, Asin)
NVFUSER_DEFINE_UNARY_FLOAT_OP(atan, Atan)
NVFUSER_DEFINE_UNARY_FLOAT_OP(atanh, Atanh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(cos, Cos)
NVFUSER_DEFINE_UNARY_FLOAT_OP(cosh, Cosh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(exp, Exp)
NVFUSER_DEFINE_UNARY_FLOAT_OP(expm1, Expm1)
NVFUSER_DEFINE_UNARY_FLOAT_OP(erf, Erf)
NVFUSER_DEFINE_UNARY_FLOAT_OP(erfc, Erfc)
NVFUSER_DEFINE_UNARY_FLOAT_OP(lgamma, Lgamma)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log, Log)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log10, Log10)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log1p, Log1p)
NVFUSER_DEFINE_UNARY_FLOAT_OP(log2, Log2)
NVFUSER_DEFINE_UNARY_FLOAT_OP(reciprocal, Reciprocal)
NVFUSER_DEFINE_UNARY_FLOAT_OP(rsqrt, Rsqrt)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sigmoid, Sigmoid)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sin, Sin)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sinh, Sinh)
NVFUSER_DEFINE_UNARY_FLOAT_OP(sqrt, Sqrt)
NVFUSER_DEFINE_UNARY_FLOAT_OP(tan, Tan)
NVFUSER_DEFINE_UNARY_FLOAT_OP(tanh, Tanh)
#undef NVFUSER_DEFINE_UNARY_FLOAT_OP

// BINARY OPERATIONS

namespace {
// Helper function to reduce repetitive code
template <typename T1, typename T2>
TensorView* arithOpOverloads(Val* (*func)(Val*, Val*), T1* v1, T2* v2) {
  return func(v1->template as<Val>(), v2->template as<Val>())
      ->template as<TensorView>();
}

template <typename T1, typename T2>
TensorView* arithOpOverloads(
    BinaryOpType type,
    T1* v1,
    T2* v2,
    DataType common_dtype) {
  return binaryOp(
             type, v1->template as<Val>(), v2->template as<Val>(), common_dtype)
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

// Output type promotion logic for binary operators
DataType getOutputType(
    BinaryOpType op_type,
    Val* v1,
    Val* v2,
    DataType common_dtype) {
  if (isLogicalOp(op_type)) {
    return DataType::Bool;
  } else if (common_dtype == DataType::Null) {
    return promote_type(v1->getDataType().value(), v2->getDataType().value());
  } else {
    return common_dtype;
  }
}

} // namespace

Val* binaryOp(BinaryOpType type, Val* v1, Val* v2, DataType common_dtype) {
  const auto out_dtype = getOutputType(type, v1, v2, common_dtype);
  const auto out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  auto vals = maybeBroadcast({v1, v2});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = newOutputTV(vals, out_dtype);
  } else {
    out = newScalar(out_vtype, out_dtype);
  }
  IrBuilder::create<BinaryOp>(type, out, vals[0], vals[1]);
  return out;
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    DataType common_dtype) {
  return arithOpOverloads(type, v1, v2, common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    DataType common_dtype) {
  return arithOpOverloads(type, v1, v2, common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    DataType common_dtype) {
  return arithOpOverloads(type, v1, v2, common_dtype);
}

Val* binaryOp(
    BinaryOpType type,
    Val* v1,
    Val* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto casted_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type, casted_values.front(), casted_values.back(), common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto casted_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type,
      casted_values.front()->as<TensorView>(),
      casted_values.back(),
      common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto casted_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type,
      casted_values.front(),
      casted_values.back()->as<TensorView>(),
      common_dtype);
}

TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2,
    const TypePromotionConfig& config) {
  std::vector<Val*> operands = {v1, v2};
  auto common_dtype = computeTypes(config, operands);
  auto casted_values = promoteValues(operands, common_dtype);
  return binaryOp(
      type,
      casted_values.front()->as<TensorView>(),
      casted_values.back()->as<TensorView>(),
      common_dtype);
}

#define NVFUSER_DEFINE_BINARY_FLOAT_OP(op_name, op_type)                \
  Val* op_name(Val* v1, Val* v2) {                                      \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }                                                                     \
  TensorView* op_name(TensorView* v1, Val* v2) {                        \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }                                                                     \
  TensorView* op_name(Val* v1, TensorView* v2) {                        \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v2, v2, TypePromotion::float_op_config); \
  }                                                                     \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                 \
    return binaryOp(                                                    \
        BinaryOpType::op_type, v1, v2, TypePromotion::float_op_config); \
  }

NVFUSER_DEFINE_BINARY_FLOAT_OP(div, Div)
NVFUSER_DEFINE_BINARY_FLOAT_OP(atan2, Atan2)
#undef NVFUSER_DEFINE_BINARY_FLOAT_OP

#define NVFUSER_DEFINE_BINARY_CAST_OP(op_name, op_type)                   \
  Val* op_name(Val* v1, Val* v2) {                                        \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, Val* v2) {                          \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(Val* v1, TensorView* v2) {                          \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }                                                                       \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                   \
    return binaryOp(                                                      \
        BinaryOpType::op_type, v1, v2, TypePromotion::default_op_config); \
  }

// Integer binary ops
NVFUSER_DEFINE_BINARY_CAST_OP(mod, Mod)
NVFUSER_DEFINE_BINARY_CAST_OP(ceilDiv, CeilDiv)

NVFUSER_DEFINE_BINARY_CAST_OP(add, Add)
NVFUSER_DEFINE_BINARY_CAST_OP(fmod, Fmod)
NVFUSER_DEFINE_BINARY_CAST_OP(mul, Mul)
NVFUSER_DEFINE_BINARY_CAST_OP(pow, Pow)
NVFUSER_DEFINE_BINARY_CAST_OP(remainder, Remainder)
NVFUSER_DEFINE_BINARY_CAST_OP(sub, Sub)
NVFUSER_DEFINE_BINARY_CAST_OP(lshift, Lshift)
NVFUSER_DEFINE_BINARY_CAST_OP(rshift, Rshift)
NVFUSER_DEFINE_BINARY_CAST_OP(andOp, And)
NVFUSER_DEFINE_BINARY_CAST_OP(orOp, Or)
NVFUSER_DEFINE_BINARY_CAST_OP(xorOp, Xor)
#undef NVFUSER_DEFINE_BINARY_CAST_OP

#define NVFUSER_DEFINE_BINARY_COMPARE_OP(op_name, op_type)                   \
  Val* op_name(Val* v1, Val* v2) {                                           \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }                                                                          \
  TensorView* op_name(TensorView* v1, Val* v2) {                             \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }                                                                          \
  TensorView* op_name(Val* v1, TensorView* v2) {                             \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }                                                                          \
  TensorView* op_name(TensorView* v1, TensorView* v2) {                      \
    return binaryOp(                                                         \
        BinaryOpType::op_type, v1, v2, TypePromotion::comparison_op_config); \
  }

// Logical binary ops
NVFUSER_DEFINE_BINARY_COMPARE_OP(eq, Eq)
NVFUSER_DEFINE_BINARY_COMPARE_OP(ge, GE)
NVFUSER_DEFINE_BINARY_COMPARE_OP(gt, GT)
NVFUSER_DEFINE_BINARY_COMPARE_OP(le, LE)
NVFUSER_DEFINE_BINARY_COMPARE_OP(lt, LT)
NVFUSER_DEFINE_BINARY_COMPARE_OP(ne, NE)
#undef NVFUSER_DEFINE_BINARY_COMPARE_OP

// REDUCTION OPERATIONS

// TODO: How do we adjust this so we can reduce to a single scalar value?
static TensorView* newForReduction(
    TensorView* tv,
    const std::vector<unsigned int>& axes,
    DataType data_type = DataType::Null) {
  auto orig_domain = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  std::set<unsigned int> axes_set(axes.begin(), axes.end());

  std::vector<IterDomain*> new_domain;

  TORCH_INTERNAL_ASSERT(
      !axes_set.empty(),
      "Asked for ouput of reduction, but no reduction axis provided.");

  TORCH_INTERNAL_ASSERT(
      (*(axes_set.rbegin())) < orig_domain.size(),
      "Error setting up reduction, reduction axis (",
      *(axes_set.rbegin()),
      ") is outside nDims (",
      orig_domain.size(),
      "). Keep in mind reductions are relative to root domains, not modified views.");

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

    new_domain.push_back(IrBuilder::create<IterDomain>(
        id->start(),
        id->extent(),
        id->stopOffset(),
        ParallelType::Serial,
        isReduction ? IterType::Reduction : id->getIterType()));
  }

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      new_domain, std::vector<bool>(new_domain.size(), true));

  data_type =
      data_type == DataType::Null ? tv->getDataType().value() : data_type;
  return IrBuilder::create<TensorView>(td, data_type);
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
      TensorDomain::sameAs(tv->getMaybeRFactorDomain(), tv->domain()->domain()),
      "Reducing a tensor once it's gone under transformations is not permitted at this time. Please set reductions before calling split/merge/computeAt.");

  TORCH_CHECK(tv->nDims() > 0, "Tried to reduce a 0-dim tensor");

  TORCH_CHECK(axes.size() > 0, "No reduction axis specified");

  std::vector<unsigned int> uint_axes;
  const int ndims = tv->domain()->noReductions().size();
  for (int axis : axes) {
    if (axis < 0) {
      axis += ndims;
    }

    TORCH_CHECK(
        axis >= 0 && axis < ndims,
        "Reduction on invalid axis, recieved: ",
        axis,
        " however tensor view only has ",
        ndims,
        " non-reduction dims.");

    uint_axes.push_back((unsigned int)axis);
  }

  TensorView* out = newForReduction(tv, uint_axes);
  const auto out_type = out->getDataType().value();
  const auto init_type = init->getDataType().value();
  TORCH_CHECK(
      (isFloatingPointType(out_type) && isFloatingPointType(init_type)) ||
          (isComplexType(out_type) && isComplexType(init_type)) ||
          (isIntegralType(out_type) && isIntegralType(init_type)) ||
          (isBooleanType(out_type) && isBooleanType(init_type)),
      "Types should match for reduction ops but received: ",
      out_type,
      " and ",
      init_type);
  IrBuilder::create<ReductionOp>(reduction_op_type, init, out, tv);

  if (keep_dim) {
    auto tv_root = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
    std::vector<bool> is_broadcast(tv_root.size(), false);
    for (auto axis : uint_axes) {
      is_broadcast.at(axis) = true;
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
    init = IrBuilder::create<Double>(0.0);
  } else if (isComplexType(dtype)) {
    init = IrBuilder::create<ComplexDouble>(c10::complex<double>(0.0, 0.0));
  } else if (isIntegralType(dtype)) {
    init = FusionGuard::getCurFusion()->zeroVal();
  } else if (isBooleanType(dtype)) {
    v1 = castOp(DataType::Int, v1);
    init = FusionGuard::getCurFusion()->zeroVal();
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
      init =
          IrBuilder::create<Double>(-std::numeric_limits<double>::infinity());
      break;
    case (DataType::Float):
      init = IrBuilder::create<Double>(-std::numeric_limits<float>::infinity());
      break;
    case (DataType::Int):
      init = IrBuilder::create<Int>(std::numeric_limits<int64_t>::lowest());
      break;
    case (DataType::Int32):
      init = IrBuilder::create<Int>(std::numeric_limits<int32_t>::lowest());
      break;
    case (DataType::Bool):
      init = IrBuilder::create<Bool>(false);
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
      init = IrBuilder::create<Double>(std::numeric_limits<double>::infinity());
      break;
    case (DataType::Float):
      init = IrBuilder::create<Double>(std::numeric_limits<float>::infinity());
      break;
    case (DataType::Int):
      init = IrBuilder::create<Int>(std::numeric_limits<int64_t>::max());
      break;
    case (DataType::Int32):
      init = IrBuilder::create<Int>(std::numeric_limits<int32_t>::max());
      break;
    case (DataType::Bool):
      init = IrBuilder::create<Bool>(true);
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
          TensorDomain::noReductions(inp->getMaybeRFactorDomain()).size(),
      "Invalid broadcast, number of false entries in is_broadcast_dim expected to be ",
      TensorDomain::noReductions(inp->getMaybeRFactorDomain()).size(),
      " but received ",
      nBCastDims - n_broadcasts);

  if (n_broadcasts == 0) {
    auto identity = set(inp);
    TORCH_INTERNAL_ASSERT(
        identity->getValType().value() == ValType::TensorView,
        "Expected identity op, but didn't get a TensorView back.");
    return identity->as<TensorView>();
  }

  std::vector<IterDomain*> out_domain;
  // Don't propagate reduction IDs through arith ops.
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  size_t iinp = 0, ibdim = 0;
  while (ibdim < is_broadcast_dim.size()) {
    if (is_broadcast_dim[ibdim]) {
      out_domain.push_back(IrBuilder::create<IterDomain>(
          FusionGuard::getCurFusion()->zeroVal(),
          FusionGuard::getCurFusion()->oneVal(),
          ParallelType::Serial,
          IterType::BroadcastWithoutStride));
    } else {
      out_domain.push_back(IrBuilder::create<IterDomain>(
          inp_domain[iinp]->start(),
          inp_domain[iinp]->extent(),
          inp_domain[iinp]->stopOffset(),
          inp_domain[iinp]->getParallelType(),
          inp_domain[iinp]->getIterType()));
      iinp++;
    }
    ibdim++;
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, std::vector<bool>(out_domain.size(), true)),
      inp->getDataType().value());
  IrBuilder::create<BroadcastOp>(out_tensor, inp, is_broadcast_dim);
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

  if (init_N == nullptr) {
    init_N = FusionGuard::getCurFusion()->zeroVal();
  }

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
    init_avg_val = IrBuilder::create<Double>(0);
    init_var_val = IrBuilder::create<Double>(0);
  }

  // Check and collect reduction axes
  std::vector<unsigned int> uint_axes;
  const int ndims = tv->domain()->noReductions().size();
  for (int axis : axes) {
    if (axis < 0) {
      axis += ndims;
    }

    TORCH_CHECK(
        axis >= 0 && axis < ndims,
        "Reduction on invalid axis, recieved: ",
        axis,
        " however tensor view only has ",
        ndims,
        " non-reduction dims.");

    uint_axes.push_back((unsigned int)axis);
  }

  // Create tensor outputs
  TensorView* out_avg = newForReduction(tv, uint_axes);
  TensorView* out_var = newForReduction(tv, uint_axes);
  TensorView* out_N = newForReduction(tv, uint_axes, DataType::Index);

  IrBuilder::create<WelfordOp>(
      out_avg,
      out_var,
      out_N, /*out var/avg/count */
      init_avg_val,
      init_var_val,
      init_N, /*init var/avg/count */
      tv,
      nullptr,
      FusionGuard::getCurFusion()->oneVal()); /*in var/avg/count */

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
  auto inp_domain = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  std::vector<IterDomain*> out_domain(inp_domain.size());

  auto new2old = ir_utils::normalizeOld2New(old2new, inp_domain.size());

  for (const auto i : c10::irange(out_domain.size())) {
    auto in_id = inp_domain[new2old[i]];
    out_domain[i] = in_id->clone();
  }

  TensorView* out_tensor = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_domain, std::vector<bool>(out_domain.size(), true)),
      inp->getDataType().value());
  IrBuilder::create<TransposeOp>(out_tensor, inp, new2old);
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
  Val* intrm = mul(vals[1], vals[2]);
  return add(vals[0], intrm);
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
  Val* intrm = mul(vals[1], vals[2]);
  return sub(vals[0], intrm);
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
Val* lerp(Val* start, Val* end, Val* weight) {
  auto vals = maybeBroadcast({start, end, weight});
  Val* intrm1 = sub(vals[1], vals[0]);
  Val* intrm2 = mul(vals[2], intrm1);
  return add(vals[0], intrm2);
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
  Val* intrm1 = mul(vals[2], vals[3]);
  Val* intrm2 = mul(vals[1], intrm1);
  return add(vals[0], intrm2);
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

  auto casted_values =
      promoteValues(TypePromotion::default_op_config, {v1, v2});
  v1 = casted_values[0];
  v2 = casted_values[1];

  TORCH_CHECK(c->getDataType().value() == DataType::Bool);
  auto out_dtype =
      promote_type(v1->getDataType().value(), v2->getDataType().value());
  auto out_vtype =
      promote_type(v1->getValType().value(), v2->getValType().value());
  auto vals = maybeBroadcast({c, v1, v2});
  Val* out = nullptr;
  if (out_vtype == ValType::TensorView) {
    out = newOutputTV(vals, out_dtype);
  } else {
    out = newScalar(out_vtype, out_dtype);
  }
  IrBuilder::create<TernaryOp>(
      TernaryOpType::Where, out, vals[0], vals[1], vals[2]);
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
  TORCH_CHECK(
      (thresh->getValType().value() == ValType::Scalar ||
       thresh->getValType().value() == ValType::NamedScalar) &&
          (value->getValType().value() == ValType::Scalar ||
           value->getValType().value() == ValType::NamedScalar),
      "For Threshold operation: Thresh and Value values should be Scalars.");

  thresh = optionalCast(in->getDataType().value(), thresh);
  value = optionalCast(in->getDataType().value(), value);
  Val* out = newValLike(in, in->getDataType().value());

  IrBuilder::create<TernaryOp>(
      TernaryOpType::Threshold, out, in, thresh, value);
  return out;
}

TensorView* threshold(TensorView* in, Val* thresh, Val* value) {
  return threshold(in->as<Val>(), thresh, value)->as<TensorView>();
}

Val* clamp(Val* in, Val* min_val, Val* max_val) {
  TORCH_CHECK(
      (min_val->getValType().value() == ValType::Scalar ||
       min_val->getValType().value() == ValType::NamedScalar) &&
          (max_val->getValType().value() == ValType::Scalar ||
           max_val->getValType().value() == ValType::NamedScalar),
      "For Clamp operation: Min and Max values should be Scalars.");

  min_val = optionalCast(in->getDataType().value(), min_val);
  max_val = optionalCast(in->getDataType().value(), max_val);
  Val* out = newValLike(in, in->getDataType().value());

  IrBuilder::create<TernaryOp>(TernaryOpType::Clamp, out, in, min_val, max_val);
  return out;
}

TensorView* clamp(TensorView* in, Val* min_val, Val* max_val) {
  return clamp(in->as<Val>(), min_val, max_val)->as<TensorView>();
}

// sum_to operator

TensorView* sum_to(TensorView* in, const std::vector<Int*>& sum_to_size) {
  const auto& root = TensorDomain::noReductions(in->getMaybeRFactorDomain());

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
  for (const auto i : c10::irange(leading_dims, root.size())) {
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
  const auto& root = TensorDomain::noReductions(in->getMaybeRFactorDomain());

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
  for (const auto i : c10::irange(leading_dims, root.size())) {
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
  // When pad is false, no padding is given. When it is true, padding
  // sizes are set so that output domains have the same extents as
  // input domains.
  std::vector<int> pad_width(offsets.size(), 0);
  if (pad) {
    for (const auto i : c10::irange(offsets.size())) {
      pad_width[i] = std::abs(offsets[i]);
    }
  }
  return shift(inp, offsets, pad_width);
}

TensorView* shift(
    TensorView* inp,
    const std::vector<int>& offsets,
    const std::vector<int>& pad_width_param) {
  auto inp_dom = TensorDomain::noReductions(inp->getRootDomain());
  const auto ndims = inp_dom.size();

  auto pad_width = pad_width_param;
  // Default padding is set so that the extent is kept unchanged
  if (pad_width.empty()) {
    pad_width = offsets;
    for (auto& p : pad_width) {
      p = std::abs(p);
    }
  }

  TORCH_CHECK(
      ndims == offsets.size(),
      "Invalid shift offsets, number of entries in offsets expected to be ",
      ndims,
      " but received ",
      offsets.size());

  TORCH_CHECK(
      ndims == pad_width.size(),
      "Invalid padding width list, number of entries in pad_width expected to be ",
      ndims,
      " but received ",
      pad_width.size());

  std::for_each(pad_width.begin(), pad_width.end(), [](const auto& pad) {
    TORCH_CHECK(pad >= 0, "Padding width must be >= 0: ", pad);
  });

  TensorView* out = nullptr;

  std::vector<IterDomain*> out_dom;
  for (const auto i : c10::irange(ndims)) {
    const auto inp_axis = inp_dom[i];
    const auto offset = offsets[i];
    const auto pad = pad_width[i];

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

    int64_t out_start_offset = 0;
    int64_t out_stop_offset = 0;

    if (offset > 0) {
      // shift to right; extent remains the same, start and stop
      // positions are moved right
      out_start_offset = cur_start_offset_value + offset - pad;
      out_stop_offset = std::max(cur_stop_offset_value - offset, int64_t(0));
      // If pad > offset, the extent of the output ID could be larger than the
      // input, and the start offset of the output domain could become
      // negative, which is not supported.
      TORCH_CHECK(
          out_start_offset >= 0,
          "Invalid shift offset and padding. Padding must not be larger than the absolute extent of shift offset. Padding: ",
          pad,
          ". Shift: ",
          offset,
          ".");
    } else {
      // shift to left; extent remains the same, start and stop
      // positions are moved left
      out_start_offset = std::max(cur_start_offset_value + offset, int64_t(0));
      out_stop_offset = cur_stop_offset_value - offset - pad;
      // Similar to the above case whwere offset is positive, if pad >
      // -offset (note offset is negative), the extent of the output
      // ID could be larger than the input, and the stop offset of the
      // output domain could become negative.
      TORCH_CHECK(
          out_stop_offset >= 0,
          "Invalid shift offset and padding. Padding must not be larger than the absolute extent of shift offset. Padding: ",
          pad,
          ". Shift: ",
          offset,
          ".");
    }

    out_dom.push_back(IrBuilder::create<IterDomain>(
        IrBuilder::create<Int>(out_start_offset),
        inp_axis->extent(),
        IrBuilder::create<Int>(out_stop_offset),
        ParallelType::Serial,
        inp_axis->getIterType()));
  }

  out = IrBuilder::create<TensorView>(
      IrBuilder::create<TensorDomain>(
          out_dom, std::vector<bool>(out_dom.size(), true)),
      inp->getDataType().value());

  IrBuilder::create<ShiftOp>(out, inp, offsets, pad_width);
  return out;
}

namespace {

// Return a new TensorDomain with given root domains. Apply
// strides if necessary. With non-unit strides, strided domains become an
// rfactor domain.
TensorDomain* generateTensorDomainWithStrides(
    const std::vector<IterDomain*>& root_domains,
    const std::vector<int>& strides,
    bool skip_unit_stride) {
  std::vector<IterDomain*> strided_domains;

  // If strides are just unit strides, don't apply striding
  if (strides.empty() ||
      (skip_unit_stride &&
       std::all_of(
           strides.begin(), strides.end(), [](int s) { return s == 1; }))) {
    return IrBuilder::create<TensorDomain>(
        root_domains, std::vector<bool>(root_domains.size(), true));
  }

  for (const auto i : c10::irange(root_domains.size())) {
    auto root_dom = root_domains.at(i);

    if (i >= strides.size() || (skip_unit_stride && strides[i] == 1)) {
      strided_domains.push_back(root_dom);
      continue;
    }

    // Split the root domain by the stride
    auto split_out = root_dom->stridedSplit(strides[i]);
    strided_domains.push_back(split_out.first);
    strided_domains.push_back(split_out.second);
  }

  auto contig_vector_size = strided_domains.size();

  auto strided_td = IrBuilder::create<TensorDomain>(
      root_domains,
      strided_domains,
      strided_domains,
      std::vector<bool>(contig_vector_size, true));

  return strided_td;
}

} // namespace

TensorView* gather(
    TensorView* inp,
    const std::vector<int>& window_shape,
    const std::vector<std::vector<int>>& pad_width,
    const std::vector<int>& strides,
    bool trim_out_of_bounds) {
  auto inp_dom = TensorDomain::noReductions(inp->getMaybeRFactorDomain());
  const auto ndims = inp_dom.size();

  TORCH_CHECK(
      ndims == window_shape.size(),
      "Invalid window shape: number of entries expected to be ",
      ndims,
      " but received ",
      window_shape.size());

  std::for_each(window_shape.begin(), window_shape.end(), [](const auto& w) {
    TORCH_CHECK(w > 0, "Window size must be > 0: ", w);
  });

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
    std::for_each(p.begin(), p.end(), [](const auto& p_left_or_right) {
      TORCH_CHECK(
          p_left_or_right >= 0, "Padding must be >= 0: ", p_left_or_right);
    });
  });

  TORCH_CHECK(
      strides.empty() || ndims == strides.size(),
      "Invalid strides: number of entries expected to be ",
      ndims,
      " but received ",
      strides.size());

  std::for_each(strides.begin(), strides.end(), [](const auto& s) {
    TORCH_CHECK(s > 0, "Stride must be > 0: ", s);
  });

  std::vector<IterDomain*> out_root_domains;
  std::vector<IterDomain*> out_gather_dom;

  for (const auto i : c10::irange(ndims)) {
    const auto inp_axis = inp_dom[i];
    const auto window_dim = window_shape[i];
    const auto pad_left = pad_width[i][0];
    const auto pad_right = pad_width[i][1];
    // This may be over-conservative
    TORCH_INTERNAL_ASSERT(inp_axis->start()->isZeroInt());
    const auto inp_stop_offset = inp_axis->stopOffset()->getInt();
    TORCH_INTERNAL_ASSERT(
        inp_stop_offset.has_value(),
        "Dynamic stop offset not supported: ",
        inp_axis);
    const auto extent_adjustment = window_dim - 1 - pad_left - pad_right;
    TORCH_CHECK(
        extent_adjustment >= 0,
        "Invalid gather window and padding as output extent would be larger than input.",
        " Window: ",
        window_dim,
        ". Padding left: ",
        pad_left,
        ". Padding right: ",
        pad_right);
    const auto out_stop_offset = inp_stop_offset.value() + extent_adjustment;
    out_root_domains.push_back(IrBuilder::create<IterDomain>(
        FusionGuard::getCurFusion()->zeroVal(),
        inp_axis->extent(),
        IrBuilder::create<Int>(out_stop_offset),
        ParallelType::Serial,
        inp_axis->getIterType()));
    // create a new axis for the gathered domain
    out_gather_dom.push_back(IrBuilder::create<IterDomain>(
        FusionGuard::getCurFusion()->zeroVal(),
        IrBuilder::create<Int>(window_dim),
        ParallelType::Serial,
        IterType::Gather));
  }

  out_root_domains.insert(
      out_root_domains.end(), out_gather_dom.begin(), out_gather_dom.end());

  TensorDomain* out_td = nullptr;

  if (trim_out_of_bounds) {
    // If no stride vector is given, just use stride 1. It does not do
    // any striding effect, but out-of-bounds values are trimmed.
    auto s = strides.empty() ? std::vector<int>(ndims, 1) : strides;
    out_td = generateTensorDomainWithStrides(out_root_domains, strides, false);
  } else {
    out_td = generateTensorDomainWithStrides(out_root_domains, strides, true);
  }

  auto out_tv =
      IrBuilder::create<TensorView>(out_td, inp->getDataType().value());

  IrBuilder::create<GatherOp>(out_tv, inp, window_shape, pad_width);
  return out_tv;
}

namespace {

//! Create new output for mma
static TensorView* newForMma(
    TensorView* tv_a,
    TensorView* tv_b,
    const std::vector<unsigned int>& axes,
    DataType data_type = DataType::Float) {
  auto orig_domain_a =
      TensorDomain::noReductions(tv_a->getMaybeRFactorDomain());
  auto orig_domain_b =
      TensorDomain::noReductions(tv_b->getMaybeRFactorDomain());

  TORCH_INTERNAL_ASSERT(
      orig_domain_a.size() == orig_domain_b.size(),
      "MMA op: need matching dim input");

  std::set<unsigned int> axes_set(axes.begin(), axes.end());
  std::vector<IterDomain*> new_domain;

  TORCH_INTERNAL_ASSERT(
      !axes_set.empty(),
      "Asked for ouput of reduction, but no reduction axis provided.");

  TORCH_INTERNAL_ASSERT(
      (*(axes_set.rbegin())) < orig_domain_a.size(),
      "Error setting up reduction, reduction axis (",
      *(axes_set.rbegin()),
      ") is outside nDims (",
      orig_domain_a.size(),
      "). Keep in mind reductions are relative to root domains, not modified views.");

  auto axis_iter = axes_set.begin();
  for (const auto dim : c10::irange(orig_domain_a.size())) {
    bool isReduction = false;
    if (axis_iter != axes_set.end() && *axis_iter == dim) {
      isReduction = true;
      axis_iter++;
    }

    const IterDomain* id = orig_domain_a[dim]->isBroadcast()
        ? orig_domain_b[dim]
        : orig_domain_a[dim];

    TORCH_CHECK(
        !(isReduction && id->isBroadcast() && !id->isImplicitBroadcast()),
        "Cannot reduce an axis that is marked as broadcasted as it has an undetermined size. Tried to reduce ID = ",
        id,
        " of tensor ",
        tv_a,
        "and",
        tv_b);

    new_domain.push_back(IrBuilder::create<IterDomain>(
        id->start(),
        id->extent(),
        id->stopOffset(),
        ParallelType::Serial,
        isReduction ? IterType::Reduction : id->getIterType()));
  }

  TensorDomain* td = IrBuilder::create<TensorDomain>(
      new_domain, std::vector<bool>(new_domain.size(), true));

  return IrBuilder::create<TensorView>(td, data_type);
}

} // namespace

TensorView* fusedMultiplySum(
    TensorView* tv_a,
    TensorView* tv_b,
    const std::vector<int>& axes,
    Val* init) {
  if (init == nullptr) {
    init = IrBuilder::create<Double>(0);
  }

  // TODO:
  //  We will want to support initialize and rfactor with
  //  mma as well, for maybe fusing bias in prolog.
  // TODO: check init type if given a tv,
  //  not supported currently though.
  TORCH_CHECK(
      init->isConstScalar(),
      "Cannot create a reduction operation where the initial value is not a const scalar.");

  // TODO:
  //  Validate axis relationships between a and b
  TORCH_CHECK(tv_a->nDims() > 0, "Tried to reduce a 0-dim tensor");

  // TODO:
  //  Add tf32 and other mma data types
  //  Add fallback path for non-mma data types.
  TORCH_CHECK(tv_a->getDataType().value() == DataType::Half);
  TORCH_CHECK(tv_b->getDataType().value() == DataType::Half);

  TORCH_CHECK(axes.size() > 0, "No reduction axis specified");

  // TODO:
  //  will lift this in a follow up when we have a
  //  more generic axes matching.
  TORCH_CHECK(
      axes.size() == 1, "Single axis reduction only for mma op instantiation.")

  std::vector<unsigned int> uint_axes;
  const int ndims = tv_a->domain()->noReductions().size();
  for (int axis : axes) {
    if (axis < 0) {
      axis += ndims;
    }

    TORCH_CHECK(
        axis >= 0 && axis < ndims,
        "Reduction on invalid axis, recieved: ",
        axis,
        " however tensor view only has ",
        ndims,
        " non-reduction dims.");

    uint_axes.push_back((unsigned int)axis);
  }

  TensorView* out = newForMma(tv_a, tv_b, uint_axes);
  IrBuilder::create<MmaOp>(out, tv_a, tv_b, init);

  return out;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
