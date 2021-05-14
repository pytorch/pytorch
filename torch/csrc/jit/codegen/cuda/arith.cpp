#include <torch/csrc/jit/codegen/cuda/arith.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

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
        case DataType::Float:
          return new Float();
        case DataType::Half:
          return new Half();
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
      "Was expecting a scalar type, but received ValType: ",
      vtype,
      " with DataType:",
      dtype);
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

  for (auto tv : tvs) {
    auto dom = TensorDomain::noReductions(tv->getRootDomain());
    TORCH_INTERNAL_ASSERT(
        dom.size() == out_domain.size(),
        "Invalid tensor view found while producing and output, it has ",
        dom.size(),
        " dimensions but expected ",
        out_domain.size());
    for (size_t i = 0; i < dom.size(); i++) {
      if (out_domain[i] != nullptr)
        continue;
      if (dom[i]->isBroadcast())
        continue;
      out_domain[i] = new IterDomain(dom[i]->start(), dom[i]->extent());
    }
  }
  for (size_t dim_i = 0; dim_i < out_domain.size(); dim_i++) {
    if (out_domain[dim_i] == nullptr) {
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

  for (size_t i = 0; i < vals.size(); i++) {
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

Val* newOutputVal(const std::vector<Val*>& vals) {
  ValType out_vtype = vals[0]->getValType().value();
  DataType out_dtype = vals[0]->getDataType().value();

  for (auto val : vals) {
    TORCH_CHECK(val->isVal(), "Invalid statement found during promotion.");
    TORCH_CHECK(
        val->getDataType().value() != DataType::Null,
        "Invalid datatype found during prmotion.");
    out_vtype = promote_type(out_vtype, val->getValType().value());
    out_dtype = promote_type(out_dtype, val->getDataType().value());
  }

  if (out_vtype == ValType::TensorView)
    return newOutputTV(vals, out_dtype);

  return newScalar(out_vtype, out_dtype);
}

Val* newValLike(Val* val, DataType dtype) {
  TORCH_CHECK(val->isVal(), "Invalid statement provided to create new value.");
  TORCH_CHECK(
      dtype != DataType::Null, "Invalid datatype provided for new value.");

  ValType vtype = val->getValType().value();

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
  Val* out = newOutputVal({v1});
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
} // namespace

TORCH_CUDA_CU_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2) {
  auto vals = maybeBroadcast({v1, v2});
  Val* out = newOutputVal({vals[0], vals[1]});
  if (is_logical_op(type)) {
    if (out->getDataType().value() != DataType::Bool)
      out = newValLike(out, DataType::Bool);
  } else if (type >= BinaryOpType::Mod) {
    if (out->getDataType().value() != DataType::Int)
      out = newValLike(out, DataType::Int);
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
      v1->getDataType().value() == DataType::Bool,
      "Input1 should be of type bool, not ",
      v1->getDataType().value());
  TORCH_CHECK(
      v2->getDataType().value() == DataType::Bool,
      "Input2 should be of type bool, not ",
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
    const std::vector<unsigned int>& axes) {
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
  for (size_t dim = 0; dim < orig_domain.size(); dim++) {
    bool isReduction = false;
    if (axis_iter != axes_set.end() && *axis_iter == dim) {
      isReduction = true;
      axis_iter++;
    }

    const IterDomain* id = orig_domain[dim];

    TORCH_CHECK(
        !(isReduction && id->isBroadcast()),
        "Cannot reduce an axis that is marked as broadcasted as it has an undetermined size. Tried to reduce ID = ",
        id,
        " of tensor ",
        tv);

    new_domain.push_back(new IterDomain(
        id->start(),
        id->extent(),
        ParallelType::Serial,
        isReduction ? IterType::Reduction : id->getIterType()));
  }

  TensorDomain* td =
      new TensorDomain(new_domain, std::vector<bool>(new_domain.size(), true));
  return new TensorView(td, tv->getDataType().value());
}

TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* tv) {
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
  if (init->getDataType().value() != tv->getDataType().value())
    init = castOp(tv->getDataType().value(), init);
  new ReductionOp(reduction_op_type, init, out, tv);
  return out;
}

TensorView* sum(TensorView* v1, const std::vector<int>& axes) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Val* init;
  switch (v1->getDataType().value()) {
    case (DataType::Float):
      init = new Float(0.0);
      break;
    case (DataType::Int):
      init = new Int(0);
      break;
    default:
      TORCH_CHECK(
          false,
          "Could not generate a sum op for tensor with type: ",
          v1->getDataType().value());
  }

  return reductionOp(BinaryOpType::Add, axes, init, v1);
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
      // Don't propagate reduction IDs through arith ops.
      out_domain.push_back(inp_domain[iinp]);
      iinp++;
    }
    ibdim++;
  }

  TensorView* out_tensor = new TensorView(
      new TensorDomain(out_domain, std::vector<bool>(out_domain.size(), true)),
      inp->getDataType().value());
  new BroadcastOp(out_tensor, inp);
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
// where
Val* where(Val* c, Val* v1, Val* v2) {
  TORCH_CHECK(
      c->getDataType().value() == DataType::Bool,
      "Condition should be of DataType Bool, not ",
      c->getDataType().value());

  auto vals = maybeBroadcast({c, v1, v2});
  Val* out = newOutputVal({vals[1], vals[2]});
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
  TORCH_CHECK(
      in->getDataType().value() == thresh->getDataType().value() &&
          in->getDataType().value() == value->getDataType().value(),
      "All input DataType values should match the input ",
      in->getDataType().value());
  TORCH_CHECK(
      thresh->getValType().value() == ValType::Scalar &&
          value->getValType().value() == ValType::Scalar,
      "Thresh and Value values should be Scalars");

  Val* out = newOutputVal({in});

  new TernaryOp(TernaryOpType::Threshold, out, in, thresh, value);
  return out;
}

TensorView* threshold(TensorView* in, Val* thresh, Val* value) {
  return threshold(in->as<Val>(), thresh, value)->as<TensorView>();
}

Val* clamp(Val* in, Val* min_val, Val* max_val) {
  TORCH_CHECK(
      in->getDataType().value() == min_val->getDataType().value() &&
          in->getDataType().value() == max_val->getDataType().value(),
      "All input DataType values should match the input ",
      in->getDataType().value());
  TORCH_CHECK(
      min_val->getValType().value() == ValType::Scalar &&
          max_val->getValType().value() == ValType::Scalar,
      "Min and Max values should be Scalars");

  Val* out = newOutputVal({in});

  new TernaryOp(TernaryOpType::Clamp, out, in, min_val, max_val);
  return out;
}

TensorView* clamp(TensorView* in, Val* min_val, Val* max_val) {
  return clamp(in->as<Val>(), min_val, max_val)->as<TensorView>();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
