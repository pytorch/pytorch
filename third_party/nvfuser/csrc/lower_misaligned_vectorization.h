#pragma once
#include <c10/macros/Export.h>

#include <ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Transform for-loop structure to handle misaligned addresses
//!
//! Sections of misaligned addresses are handled sequentially
//! while aligned addresses use vectorized memory accesses.
//!
//! ---------------------------------------------------------------------------
//! Before Misaligned Vectorization:
//!
//! Inputs: T0
//! Outputs: T3
//!
//! for(...) {
//!   T1[vector_size];
//!   for( i : vector_size ) {
//!     T1[i] = T0[...]
//!   }
//!
//!   T2[vector_size];
//!   for( i : vector_size ) {
//!     T2[i] = unaryOp(T1[i])
//!   }
//!
//!   for( i : vector_size ) {
//!     T3[...] = T2[i]
//!   }
//! }
//!
//! ---------------------------------------------------------------------------
//! After Misaligned Vectorization:
//!
//! Inputs: T0
//! Outputs: T3
//!
//! for(...) {
//!   T1[vector_size];
//!   T2[vector_size];
//!
//!   if (inline_predicate_except_last_root_domain) {
//!     index_except_last_root_domain = ...
//!     address = (int64_t) &T1[index_except_last_root_domain]
//!
//!     offset_size = (address % vector_size_bytes) / data_type_size_bytes
//!     shift_init = vector_size - offset_size
//!     shift = (shift_init == vector_size) ? 0 : shift_init
//!
//!     // size of the last root domain
//!     extent = ...
//!     remainder = (extent - shift) % vector_size
//!
//!     last_root_domain_index = ...
//!
//!     // Vectorize Section
//!     if ( (last_root_domain_index + shift) < (extent - remainder) ) {
//!       T1[0] = vectorize_load( T0[index + shift] );
//!
//!       for( i : vector_size ) {
//!         T2[i] = unaryOp(T1[i])
//!       }
//!
//!       T3[index + shift] = vectorize_store( T2[0] );
//!     }
//!
//!     // Initial Section
//!     if ( last_root_domain_index == 0 ) {
//!       for( i : shift ) {
//!         T1[i] = T0[...]
//!       }
//!
//!       for( i : shift ) {
//!         T2[i] = unaryOp(T1[i])
//!       }
//!
//!       for( i : shift ) {
//!         T3[...] = T2[i]
//!       }
//!     }
//!
//!     // Remainder Section
//!     if ( (last_root_domain_index + shift) >= (extent - remainder) &&
//!          (last_root_domain_index + shift) < extent) {
//!
//!       for( i : remainder ) {
//!         T1[i] = T0[index + shift]
//!       }
//!
//!       for( i : remainder ) {
//!         T2[i] = unaryOp(T1[i])
//!       }
//!
//!       for( i : remainder ) {
//!         T3[index + shift] = T2[i]
//!       }
//!     }
//!   }
//! }
//!
std::vector<Expr*> processMisalignedVectorization(
    const std::vector<Expr*>& exprs);

bool containsAnyDirectChildMisalignedVectorize(const kir::ForLoop* fl);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
