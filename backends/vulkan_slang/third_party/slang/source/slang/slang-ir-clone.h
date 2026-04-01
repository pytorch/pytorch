// slang-ir-clone.h
#pragma once

#include "../core/slang-dictionary.h"
#include "slang-ir.h"

namespace Slang
{
struct IRBuilder;
struct IRInst;

// This file provides an interface to simplify the task of
// correcting "cloning" IR code, whether individual
// instructions, or whole functions.

/// An environment for mapping existing values to their cloned replacements.
///
/// This type serves two main roles in the process of IR cloning:
///
/// * Before cloning begins, a client will usually
///   register the mapping from things that are to be
///   replaced entirely (like function parameters to
///   be specialized away) to their replacements (e.g.,
///   a constant value).
///
/// * During the process of cloning, env environment
///   will be maintained and updated so that when, e.g.,
///   an instruction later in a function refers to
///   something from earlier, we can look up the
///   replacement.
///
struct IRCloneEnv
{
    /// A mapping from old values to their replacements.
    Dictionary<IRInst*, IRInst*> mapOldValToNew;

    /// A parent environment to fall back to if `mapOldValToNew` doesn't contain a key.
    IRCloneEnv* parent = nullptr;

    /// Should `mapOldValToNew` keep a copy of children's oldToNew mapping?
    bool squashChildrenMapping = false;
};

/// Look up the replacement for `oldVal`, if any, registered in `env`.
///
/// Returns `nullptr` if `oldVal` has no registered replacement.
///
IRInst* lookUp(IRCloneEnv* env, IRInst* oldVal);

// The SSA property and the way we have structured
// our "phi nodes" (block parameters) means that
// just going through the children of a function,
// and then the children of a block will generally
// do the Right Thing and always visit an instruction
// before its uses.
//
// The big exception to this is that branch instructions
// can refer to blocks later in the same function.
//
// We work around this sort of problem in a fairly
// general fashion, by splitting the cloning of
// an instruction into two steps.
//
// The first step is just to clone the instruction
// and its direct operands, but not any decorations
// or children.

/// Clone `oldInst` and its direct operands.
///
/// The "direct operands" include the type of the instruction.
/// The type and operands of `oldInst` will be mapped to now
/// values using `findOrCloneOperand` with the given `env`.
///
/// Any new instruction that gets emitted will be output to
/// the provided `builder`, which must be non-null.
///
/// This operation does *not* clone any children or decorations on `oldInst`.
/// This operation does *not* register its result as a replacement
/// for `oldInst` in the given `env`.
///
IRInst* cloneInstAndOperands(IRCloneEnv* env, IRBuilder* builder, IRInst* oldInst);

// The second phase of cloning an instruction is to clone
// its decorations and children. This step only needs to
// be performed on those instructions that *have* decorations
// and/or children.

/// Clone any decorations and/or children of `oldInst` onto `newInst`
///
/// Any new instructions that get emitted will use the
/// provided `sharedBuilder`, which must be non-null.
///
/// During the process of cloning decorations/children, operand values
/// will be looked up in the provided `env`, which should provide
/// replacement values for instructions that should have a different
/// identity in the clone.
/// The provided `env` will *not* be updated/modified during the
/// process of cloding decorations/children.
///
/// If any child or decoration on `oldInst` already has a replacement
/// registered in `env`, it will *not* be cloned into `newInst`.
///
void cloneInstDecorationsAndChildren(
    IRCloneEnv* env,
    IRModule* module,
    IRInst* oldInst,
    IRInst* newInst);

// For the case where the user knows the sequencing constraints
// on cloning operands before uses can be satisfied, we provide
// a convenience wrapper around the two phases of cloning:

/// Clone `oldInst` and return the cloned value.
///
/// This function is a convenience wrapper around
/// `cloneInstAndOperands` and `cloneInstDecorationsAndChildren`.
/// It also registers the resultint instruction as
/// the replacement value for `oldInst` in the given `env`
/// which must therefore be non-null.
///
IRInst* cloneInst(IRCloneEnv* env, IRBuilder* builder, IRInst* oldInst);

/// Clone `oldDecoration` and attach the clone to `newParent`.
///
/// Uses `module` to allocate any new instructions.
///
void cloneDecoration(
    IRCloneEnv* parentEnv,
    IRDecoration* oldDecoration,
    IRInst* newParent,
    IRModule* module);

/// Clone `oldDecoration` and attach the clone to `newParent`.
///
/// Uses the module of `newParent` to allocate any new instructions,
/// so that `newParent` must already be installed somewhere
/// in the ownership hierarchy of an existing module.
///
void cloneDecoration(IRDecoration* oldDecoration, IRInst* newParent);


/// Find the "cloned" value to use for an operand.
///
/// This either returns the value registered for `oldOperand`
/// in `env`, or else `oldOperand` itself.
IRInst* findCloneForOperand(IRCloneEnv* env, IRInst* oldOperand);

// It isn't technically part of the cloning infrastructure,
// but when make specialized copies of IR instructions via
// cloning we often need a simple kind of key suitable
// for caching existing specializations, so we'll define
// it here so that is is easily accessible to code that
// needs it.

struct IRSimpleSpecializationKey
{
    // The structure of a specialization key will be a list
    // of instructions, typically starting with the function,
    // generic, or other object to be specialized, and then
    // having one or more entries to represent the specialization
    // arguments.
    //
    List<IRInst*> vals;

    // In order to use this type as a `Dictionary` key we
    // need it to support equality and hashing.
    //
    // TODO: honestly we might consider having `getHashCode`
    // and `operator==` defined for `List<T>`.

    bool operator==(IRSimpleSpecializationKey const& other) const;
    HashCode getHashCode() const;
};

} // namespace Slang
