#pragma once

#include <ir_all_nodes.h>
#include <type.h>

#include <vector>

namespace nvfuser {
namespace ops {

TensorView* maybe_broadcast_inner_to_rank(TensorView* t, size_t rank);

TensorView* maybe_broadcast_index_tv(TensorView* t, size_t dim, size_t rank);

Val* simplifiedInt(Val* val);

// If one size is nullptr, return the other. If both symbolic just return v1. If
// one's concrete, prefer that one (simplified). If both concrete make sure
// they're the same size.
Val* promoteSize(Val* v1, Val* v2);

// Will return a new value of type val with the DataType dtype.
Val* newScalar(ValType vtype, DataType dtype);

IterType promoteIterType(IterType type1, IterType type2);

TensorView* newOutputTV(const std::vector<Val*>& vals, DataType dtype);

std::vector<Val*> maybeBroadcast(const std::vector<Val*>& vals);

Val* newValLike(Val* val, DataType dtype);

// returns the minimum init value for reduction:
//   -inf for floating type;
//   lowest value for integer type;
//   false for bool.
Val* getMinimumValue(DataType v);

// returns the maximum init value for reduction:
//   inf for floating type;
//   highest value for integer type;
//   true for bool.
Val* getMaximumValue(DataType v);

} // namespace ops
} // namespace nvfuser
