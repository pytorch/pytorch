#include "slang-ir-fuse-satcoop.h"

#include "slang-ir-inline.h"
#include "slang-ir-insts.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir-ssa-simplification.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

namespace Slang
{

//
// Some helpers
//

static bool uses(IRInst* used, IRInst* user)
{
    for (auto use = used->firstUse; use; use = use->nextUse)
    {
        if (use->getUser() == user)
            return true;
    }
    return false;
};

// given: `f; x; g`
// reorder instructions such that f and g are adjacent, in the form:
// `p; f; g; q`,
//
// p is the set of instructions upon which g depends and q is the
// set of instructions which depend on f. If these sets are not disjoint then
// we can't float f and g together. Instructions not used by g and which don't
// use f can go in either p or q.
//
// Returns g on success
static IRInst* floatTogether(IRInst* f, IRInst* g)
{
    List<IRInst*> ps, qs;

    auto usesF = [&](IRInst* i)
    {
        if (uses(f, i))
            return true;
        for (auto q : qs)
            if (uses(q, i))
                return true;
        return false;
    };
    auto usedByG = [&](IRInst* i)
    {
        if (uses(i, g))
            return true;
        for (auto p : ps)
            if (uses(i, p))
                return true;
        return false;
    };

    // Scan backwards to find which instructions g depends on, known as p
    auto i = g->prev;
    while (i != f)
    {
        SLANG_ASSERT(i);

        // If the instruction is not movable, then obviously we can't move it.
        //
        // For a slight optimization: This is actually stricter than we need:
        // if `x = p;q` and f and g are movable, then we can safely move f and
        // g in and maintain the ordering of p and q
        if (!isMovableInst(i))
            return nullptr;

        if (usedByG(i))
            ps.add(i);
        i = i->prev;
    }

    // Scan forwards to compute instructions which depend on f, the instructions in q
    i = f->next;
    while (i != g)
    {
        if (usesF(i))
        {
            // If this happens then ps and qs are not disjoint, and we will not
            // be able to float f and g together
            if (ps.contains(i))
                return nullptr;
            qs.add(i);
        }

        i = i->next;
    }

    // Now we can safely reorder things by moving p;f;g before everything else
    // Remember, we constructed ps in reverse, so we must insert these
    // backwards too
    for (Index j = ps.getCount() - 1; j >= 0; --j)
    {
        auto p = ps[j];
        p->removeFromParent();
        p->insertBefore(f);
    }
    g->removeFromParent();
    g->insertAfter(f);
    return g;
}

// bifanout(f, g)((x, y), (a, b)) = (f(x, a), g(y, b))
//
// Make a function `bifanout` which applies two functions to their respective
// elements in two pairs. Optionally the first and second inputs can be shared
// instead of split in a tuple.
//
// The outputs are returned in a 2-tuple
static IRFunc* makeBiFanout(
    IRBuilder& builder,
    IRFunc* f,
    IRFunc* g,
    bool shareFirst,
    bool shareSecond)
{
    SLANG_ASSERT(f->getParamCount() == 2);
    SLANG_ASSERT(g->getParamCount() == f->getParamCount());
    SLANG_ASSERT(!shareFirst || f->getParamType(0) == g->getParamType(0));
    SLANG_ASSERT(!shareSecond || f->getParamType(1) == g->getParamType(1));
    IRBuilderInsertLocScope insertLocScope(&builder);

    // Create (using shareFirst = false, shareSecond = true as an example)
    // func myFunc(s : S, u : (U1,U2)) -> (R1, R2)
    // {
    //     let fRes = f(s, u.fst);
    //     let gRes = g(s, u.snd);
    //     return (fRes, gRes);
    // }

    // The return type is the tuple of f and g's return types
    auto resType = builder.getTupleType(f->getResultType(), g->getResultType());
    auto firstInputType = shareFirst ? f->getParamType(0)
                                     : builder.getTupleType(f->getParamType(0), g->getParamType(0));
    auto secondInputType = shareSecond
                               ? f->getParamType(1)
                               : builder.getTupleType(f->getParamType(1), g->getParamType(1));

    // Set up our function
    // func myFunc(s : S, u : (U1,U2)) -> (R1, R2)
    auto func = builder.createFunc();
    builder.addDecoration(func, kIROp_ForceInlineDecoration);
    builder.setDataType(func, builder.getFuncType({firstInputType, secondInputType}, resType));
    builder.setInsertInto(func);
    auto b = builder.emitBlock();
    builder.setInsertInto(b);

    auto s = builder.emitParam(firstInputType);
    auto s1 = shareFirst ? s : builder.emitGetTupleElement(f->getParamType(0), s, 0);
    auto s2 = shareFirst ? s : builder.emitGetTupleElement(g->getParamType(0), s, 1);

    auto u = builder.emitParam(secondInputType);
    auto u1 = shareSecond ? u : builder.emitGetTupleElement(f->getParamType(1), u, 0);
    auto u2 = shareSecond ? u : builder.emitGetTupleElement(g->getParamType(1), u, 1);

    //     let fRes = f(s, u.fst);
    auto fRes = builder.emitCallInst(f->getResultType(), f, {s1, u1});
    //     let gRes = g(s, u.snd);
    auto gRes = builder.emitCallInst(g->getResultType(), g, {s2, u2});
    //     return (fRes, gRes);
    builder.emitReturn(builder.emitMakeTuple(fRes, gRes));
    return func;
}

// Given f : a -> uint4, g : b -> uint4, return z : (a, b) -> uint4 using
// bitwise and to combine the outputs
static IRFunc* makeWaveMatchBoth(
    IRBuilder& builder,
    IRType* inputTypeF,
    IRType* inputTypeG,
    IRInst* f,
    IRInst* g)
{
    // SLANG_ASSERT(f->getParamCount() == 1);
    // SLANG_ASSERT(g->getParamCount() == f->getParamCount());
    auto uint4Type = builder.getVectorType(builder.getUIntType(), 4);
    // SLANG_ASSERT(f->getResultType() == uint4Type);
    // SLANG_ASSERT(g->getResultType() == f->getResultType());
    IRBuilderInsertLocScope insertLocScope(&builder);

    // Create (using shareFirst = false, shareSecond = true as an example)
    // func myFunc(x : (A,B)) -> uint4
    // {
    //     let fRes = f(x.fst);
    //     let gRes = g(x.snd);
    //     return fRes & gRes;
    // }

    auto inputTypeFG = builder.getTupleType(inputTypeF, inputTypeG);
    auto resType = uint4Type;

    auto func = builder.createFunc();
    builder.addDecoration(func, kIROp_ForceInlineDecoration);
    builder.setDataType(func, builder.getFuncType({inputTypeFG}, resType));
    builder.setInsertInto(func);
    auto b = builder.emitBlock();
    builder.setInsertInto(b);

    auto x = builder.emitParam(inputTypeFG);
    auto x1 = builder.emitGetTupleElement(inputTypeF, x, 0);
    auto x2 = builder.emitGetTupleElement(inputTypeG, x, 1);

    auto b1 = builder.emitCallInst(uint4Type, f, {x1});
    auto b2 = builder.emitCallInst(uint4Type, g, {x2});
    auto r = builder.emitBitAnd(uint4Type, b1, b2);

    builder.emitReturn(r);
    return func;
}

// Similar to above
static IRFunc* makeBroadcastBoth(
    IRBuilder& builder,
    IRType* inputTypeF,
    IRType* inputTypeG,
    IRInst* f,
    IRInst* g)
{
    // SLANG_ASSERT(f->getParamCount() == 2);
    // SLANG_ASSERT(g->getParamCount() == f->getParamCount());
    auto intType = builder.getIntType();
    // SLANG_ASSERT(f->getParamType(1) == intType);
    // SLANG_ASSERT(g->getParamType(1) == f->getParamType(1));
    IRBuilderInsertLocScope insertLocScope(&builder);

    // Create (using shareFirst = false, shareSecond = true as an example)
    // func myFunc(x : (A,B), i : int) -> (A, B)
    // {
    //     let fRes = f(x.fst, i);
    //     let gRes = g(x.snd, i);
    //     return (fRes, gRes);
    // }

    auto inputTypeFG = builder.getTupleType(inputTypeF, inputTypeG);
    auto resType = inputTypeFG;

    auto func = builder.createFunc();
    builder.addDecoration(func, kIROp_ForceInlineDecoration);
    builder.setDataType(func, builder.getFuncType({inputTypeFG, intType}, resType));
    builder.setInsertInto(func);
    auto b = builder.emitBlock();
    builder.setInsertInto(b);

    auto x = builder.emitParam(inputTypeFG);
    auto i = builder.emitParam(intType);
    auto x1 = builder.emitGetTupleElement(inputTypeF, x, 0);
    auto x2 = builder.emitGetTupleElement(inputTypeG, x, 1);

    auto b1 = builder.emitCallInst(inputTypeF, f, {x1, i});
    auto b2 = builder.emitCallInst(inputTypeG, g, {x2, i});
    auto r = builder.emitMakeTuple(b1, b2);

    builder.emitReturn(r);
    return func;
}

// All the information on a call to saturated_cooperation_using
struct SatCoopCall
{
    // The definition in hlsl.slang
    IRGeneric* generic;

    // The specialization of that call
    IRSpecialize* specialize;

    // Called 'A' in the definition
    IRType* sharedInputType;
    // Called 'B' in the definition
    IRType* distinctInputType;
    // Called 'C' in the definition
    IRType* retType;

    // The function arguments to the call
    IRFunc* cooperate;
    IRFunc* fallback;

    // The inter-lane communication functions
    // TODO: call specializeGeneric on these and extract the IRFunc
    IRInst* waveMatch;
    IRInst* broadcast;

    // The values to pass to these functions
    IRInst* sharedInput;
    IRInst* distinctInput;
};

static SatCoopCall getSatCoopCall(IRCall* f)
{
    SatCoopCall ret;
    ret.specialize = as<IRSpecialize>(f->getCallee());

    // Since this is a call to saturated_cooperation, it must have at least
    // three specialization arguments for the type parameters A, B, C. We allow
    // more here for any dictionaries or witnesses.
    SLANG_ASSERT(ret.specialize && ret.specialize->getArgCount() >= 3);
    ret.generic = as<IRGeneric>(ret.specialize->getBase());
    SLANG_ASSERT(ret.generic);
    ret.sharedInputType = as<IRType>(ret.specialize->getArg(0));
    ret.distinctInputType = as<IRType>(ret.specialize->getArg(1));
    ret.retType = as<IRType>(ret.specialize->getArg(2));
    SLANG_ASSERT(ret.sharedInputType);
    SLANG_ASSERT(ret.distinctInputType);
    SLANG_ASSERT(ret.retType);

    SLANG_ASSERT(f->getArgCount() == 6);
    ret.cooperate = as<IRFunc>(f->getArg(0));
    ret.fallback = as<IRFunc>(f->getArg(1));
    SLANG_ASSERT(ret.cooperate);
    SLANG_ASSERT(ret.fallback);

    ret.waveMatch = f->getArg(2);
    ret.broadcast = f->getArg(3);
    SLANG_ASSERT(ret.waveMatch);
    SLANG_ASSERT(ret.broadcast);

    ret.sharedInput = f->getArg(4);
    ret.distinctInput = f->getArg(5);
    SLANG_ASSERT(ret.sharedInput->getDataType() == ret.sharedInputType);
    SLANG_ASSERT(ret.distinctInput->getDataType() == ret.distinctInputType);
    return ret;
}

// transform:
//     a = sat_coop(c1, f1, s1, u1); // f
//     p;
//     q;
//     b = sat_coop(c2, f2, s2, u2); // g
// to:
//     p;
//     (a,b) = sat_coop(c1 &&& c2, f1 &&& f2, (s1, s2), (u1, u2));
//     q;
//
// Removes the first two calls, and returns the second one if creation was
// successful.
//
// This can fail if:
//
// p has side effects which c1 or f1 may depend on
// q has side effects which c2 or f2 may depend on
// p depends on a
// the second call to sat_coop depends on a
// the second call to sat_coop depends on q
static IRCall* tryFuseCalls(IRBuilder& builder, IRCall* f, IRCall* g)
{
    // TODO: Make sure that the types in here are concrete, use
    // `isGenericParam`

    IRBuilderInsertLocScope insertLocScope(&builder);

    SatCoopCall callF = getSatCoopCall(f);
    SatCoopCall callG = getSatCoopCall(g);
    // If these aren't referencing the same generic, then something has gone
    // wrong in our assumptions.
    SLANG_ASSERT(callF.generic == callG.generic);

    // If g uses the result of f, we can't fuse them with this logic (we could
    // however with a replacement for 'fanout')
    if (uses(f, g))
        return nullptr;

    // If there is no safe way to float these together, then fail
    const auto q = floatTogether(f, g);
    if (!q)
        return nullptr;
    builder.setInsertBefore(q);

    // As a slight neatening, we'll avoid wrapping and upwrapping a tuple (u,u)
    // if both f and g use the same distinct input..
    bool usesSameDistinctInput = callF.distinctInput == callG.distinctInput;
    SLANG_ASSERT(!usesSameDistinctInput || callF.distinctInputType == callG.distinctInputType);

    // Similarly for the shared input: if these use the same shared input then
    // the fusing is simpler (no need to make a product of s1 and s2)
    // TODO: if there is an injection from s1 to s2, then we can avoid the WaveMatch on s2
    const bool usesSameSharedInput = callF.sharedInput == callG.sharedInput &&
                                     callF.waveMatch == callG.waveMatch &&
                                     callF.broadcast == callG.broadcast;
    SLANG_ASSERT(!usesSameSharedInput || callF.sharedInputType == callG.sharedInputType);

    // Generate a new specialization of our saturated_cooperation_using function,
    // reflecting the new input and output types.
    const auto newRetType = builder.getTupleType(callF.retType, callG.retType);
    const auto sharedInputType =
        usesSameSharedInput ? callF.sharedInputType
                            : builder.getTupleType(callF.sharedInputType, callG.sharedInputType);
    const auto distinctInputType =
        usesSameDistinctInput
            ? callF.distinctInputType
            : builder.getTupleType(callF.distinctInputType, callG.distinctInputType);

    // Make sure there are no other generic parameters which are are failing to
    // take care of here.
    SLANG_ASSERT(callF.specialize->getArgCount() == 3);
    SLANG_ASSERT(callG.specialize->getArgCount() == 3);

    // Specialize our new call
    const auto newSpec = builder.emitSpecializeInst(
        builder.getTypeKind(),
        callF.generic,
        {sharedInputType, distinctInputType, newRetType});

    // Make our new functions, and joined inputs
    const auto newCooperate = makeBiFanout(
        builder,
        callF.cooperate,
        callG.cooperate,
        usesSameSharedInput,
        usesSameDistinctInput);
    const auto newFallback = makeBiFanout(
        builder,
        callF.fallback,
        callG.fallback,
        usesSameSharedInput,
        usesSameDistinctInput);
    const auto newWaveMatch = usesSameSharedInput ? callF.waveMatch
                                                  : makeWaveMatchBoth(
                                                        builder,
                                                        callF.sharedInputType,
                                                        callG.sharedInputType,
                                                        callF.waveMatch,
                                                        callG.waveMatch);
    const auto newBroadcast = usesSameSharedInput ? callF.broadcast
                                                  : makeBroadcastBoth(
                                                        builder,
                                                        callF.sharedInputType,
                                                        callG.sharedInputType,
                                                        callF.broadcast,
                                                        callG.broadcast);
    const auto newSharedInput = usesSameSharedInput
                                    ? callF.sharedInput
                                    : builder.emitMakeTuple(callF.sharedInput, callG.sharedInput);
    const auto newDistinctInput =
        usesSameDistinctInput ? callF.distinctInput
                              : builder.emitMakeTuple(callF.distinctInput, callG.distinctInput);

    // Call it and extract the results from f and g
    const auto res = builder.emitCallInst(
        newRetType,
        newSpec,
        {newCooperate, newFallback, newWaveMatch, newBroadcast, newSharedInput, newDistinctInput});
    const auto resF = builder.emitGetTupleElement(callF.retType, res, 0);
    const auto resG = builder.emitGetTupleElement(callG.retType, res, 1);
    f->replaceUsesWith(resF);
    g->replaceUsesWith(resG);
    f->removeAndDeallocate();
    g->removeAndDeallocate();

    return res;
}

//
// Identify calls which we can fuse
//
IRCall* isKnownFunction(const char* n, IRInst* i)
{
    auto call = as<IRCall>(i);
    if (!call)
        return nullptr;
    // saturated_cooperation is a generic function, so look for specializations thereof
    auto spec = as<IRSpecialize>(call->getCallee());
    if (!spec)
        return nullptr;
    auto generic = findSpecializedGeneric(spec);
    if (!generic)
        return nullptr;

    auto inner = findGenericReturnVal(generic);
    if (!inner)
        return nullptr;

    auto h = inner->findDecoration<IRKnownBuiltinDecoration>();
    if (!h || h->getName() != n)
        return nullptr;
    return call;
}

//
// We perform a left fold over calls to saturated_cooperation
//
// sc(ca, fa)
// sc(cb, fb)
// sc(cc, fc)
//
// to
//
// sc(cacbcc, fafbfc)
//
// where cacbcc (and fafbfc) look like
//
// cacbcc(){
//   cacb();
//   cc();
// }
//
// cacb(){
//   ca();
//   cb();
// }
//
// These helper functions are inlined shortly after and the generated code is
// exactly what you'd expect: it's the body of sat_coop except that the
// original call to cooperate is replaced by three calls to ca, cb, cc.
//
// We use a fold here rather than accumulating everything at once as it's
// easier to implement fusing for 2 functions than n
static void fuseCallsInBlock(IRBuilder& builder, IRBlock* block)
{
    // first, inline calls to saturated_cooperation to expose
    // saturated_cooperation_using which is simpler to fuse.
    // It is simpler to fuse because it makes explicit the inter-lane
    // communication functions, which we can use as buiding blocks in our
    // composition.

    List<IRCall*> toInline;
    for (auto inst : block->getChildren())
    {
        if (auto sat_coop = isKnownFunction("saturated_cooperation", inst))
            toInline.add(sat_coop);
    }
    for (auto c : toInline)
        inlineCall(c);

    // Walk over the instructions in this block
    // If we see a call to sat_coop then remember where it is and keep
    // walking, if we reach another call without first encountering any
    // instructions with which our first call can't be safely reordered
    // then we remove the first call and replace the second with a fused
    // call.
    IRCall* lastCall = nullptr;
    for (auto inst = block->getFirstInst(); inst != block->getTerminator();
         inst = inst->getNextInst())
    {
        if (auto call = isKnownFunction("saturated_cooperation_using", inst))
        {
            if (lastCall)
            {
                auto fused = tryFuseCalls(builder, lastCall, call);
                if (fused)
                {
                    inst = fused;
                    lastCall = fused;
                }
                else
                {
                    lastCall = call;
                }
            }
            else
            {
                lastCall = call;
            }
        }
    }
}

void fuseCallsToSaturatedCooperation(IRModule* module)
{
    IRBuilder builder(module);
    overAllBlocks(module, [&](auto b) { fuseCallsInBlock(builder, b); });
}

} // namespace Slang
