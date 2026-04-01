#include "slang-ir-legalize-vector-types.h"

#include "slang-ir-insts.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{
struct VectorTypeLoweringContext
{
    IRModule* module;
    DiagnosticSink* sink;

    InstWorkList workList;
    InstHashSet workListSet;

    Dictionary<IRInst*, IRInst*> replacements;

    VectorTypeLoweringContext(IRModule* module)
        : module(module), workList(module), workListSet(module)
    {
    }

    void addToWorkList(IRInst* inst)
    {
        for (auto ii = inst->getParent(); ii; ii = ii->getParent())
        {
            if (as<IRGeneric>(ii))
                return;
        }

        if (workListSet.contains(inst))
            return;

        workList.add(inst);
        workListSet.add(inst);
    }

    bool is1Vector(IRType* t)
    {
        const auto lenLit = composeGetters<IRIntLit>(t, &IRVectorType::getElementCount);
        return lenLit ? getIntVal(lenLit) == 1 : false;
    };

    bool has1VectorType(IRInst* i) { return is1Vector(i->getDataType()); }

    bool has1VectorPtrType(IRInst* i)
    {
        const auto ptr = as<IRPtrTypeBase>(i->getDataType());
        return ptr && is1Vector(ptr->getValueType());
    }

    // If necessary, this returns a new instruction which operates on the
    // single component of a 1-vector.
    // If no new instruction was created, then the old one is returned
    // unmodified, when we replace the 1-vector type globally, only then
    // will the return type of that instruction be updated; thus you
    // shouldn't rely on this function returning an instruction with a non
    // 1-vector return type (even if we didn't have the deferred
    // replacement this is not true, as it'll only eliminate at most one
    // level of 1-vectornes, and nested vectors exist)
    IRInst* getReplacement(IRInst* inst)
    {
        IRInst* replacement = nullptr;
        if (replacements.tryGetValue(inst, replacement))
            return replacement;

        IRBuilder builder(module);
        builder.setInsertBefore(inst);
        replacement = instMatch<IRInst*>(
            inst,
            nullptr,
            // The following match instructions which take a 1-vector as an
            // operand and are sensitive to the fact that it's a vector.
            // Likewise for pointers.
            [&](IRGetElement* getElement)
            {
                const auto base = getElement->getBase();
                return has1VectorType(base) ? getReplacement(base) : nullptr;
            },
            [&](IRSwizzle* swizzle) -> IRInst*
            {
                const auto swizzled = swizzle->getBase();

                // Is this a swizzle of a 1-vector
                if (has1VectorType(swizzled))
                {
                    // If this is a unary swizzle, just return the element
                    // inside
                    const auto scalar = getReplacement(swizzled);
                    if (swizzle->getElementCount() == 1)
                        return scalar;
                    // Otherwise, create a broadcast of this scalar
                    else
                        return builder.emitMakeVectorFromScalar(swizzle->getFullType(), scalar);
                }
                return nullptr;
            },
            [&](IRGetElementPtr* gep)
            {
                const auto base = gep->getBase();
                return has1VectorPtrType(base) ? getReplacement(base) : nullptr;
            },
            [&](IRSwizzledStore* swizzledStore)
            {
                const auto base = swizzledStore->getDest();
                return has1VectorPtrType(base)
                           ? builder.emitStore(getReplacement(base), swizzledStore->getSource())
                           : nullptr;
            },
            // The following should match any instruction which can construct,
            // specifically, a 1-vector. For example 'MakeVector'
            //
            // Instruction like, for example, arithmetic instructions don't
            // need to be handled here, and they'll be fixed by the global
            // 1-vector to scalar type replacement.
            [&](IRMakeVectorFromScalar* makeVec)
            { return has1VectorType(makeVec) ? getReplacement(makeVec->getOperand(0)) : nullptr; },
            [&](IRMakeVector* makeVec)
            { return has1VectorType(makeVec) ? getReplacement(makeVec->getOperand(0)) : nullptr; },
            // Otherwise if this is a 1-vector type itself, replace it with
            // the scalar version.
            [&](IRVectorType* vecTy)
            { return is1Vector(vecTy) ? getReplacement(vecTy->getElementType()) : nullptr; });

        // Sadly it's not really possible to catch missing cases here, as
        // there are heaps of instructions which don't do anything special
        // with vectors, but can take or return vector types, for example
        // arithmetic, IRGetElement, IRGetField etc...

        // If we did get a replacement, add that to our mapping and return
        // it, otherwise return the original (to maybe be updated later)
        if (replacement)
        {
            replacements.set(inst, replacement);
            addToWorkList(replacement);
        }

        return replacement ? replacement : inst;
    }

    void processModule()
    {
        addToWorkList(module->getModuleInst());

        while (workList.getCount() != 0)
        {
            IRInst* inst = workList.getLast();

            workList.removeLast();
            workListSet.remove(inst);

            // Run this inst through the replacer
            getReplacement(inst);

            for (auto child = inst->getLastChild(); child; child = child->getPrevInst())
            {
                addToWorkList(child);
            }
        }

        // Apply all replacements
        //
        // It's important to defer this as if we were updating things
        // on-the-fly we would be losing information about what was
        // actually a 1-vector or not. The alternative would be cloning
        // every function with a 1-vector type as we process it, and
        // cleaning up at the end. This involves less copying, but is
        // necessarily a little less type-safe.
        for (const auto& [old, replacement] : replacements)
        {
            if (old != replacement)
            {
                old->replaceUsesWith(replacement);
                old->removeAndDeallocate();
            }
        }
    }
};

void legalizeVectorTypes(IRModule* module, DiagnosticSink* sink)
{
    VectorTypeLoweringContext context(module);
    context.sink = sink;
    context.processModule();
}
} // namespace Slang
