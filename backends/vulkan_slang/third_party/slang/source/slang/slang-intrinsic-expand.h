// slang-intrinsic-expand.h
#ifndef SLANG_INTRINSIC_EXPAND_H
#define SLANG_INTRINSIC_EXPAND_H

#include "slang-emit-c-like.h"

namespace Slang
{

/* Handles all the special case handling of expansions of intrinsics. In particular handles the
expansion of the 'special cases' prefixed with '$' */
struct IntrinsicExpandContext
{
    IntrinsicExpandContext(CLikeSourceEmitter* emitter)
        : m_emitter(emitter), m_writer(emitter->getSourceWriter())
    {
    }

    void emit(
        IRCall* inst,
        IRUse* args,
        Int argCount,
        const UnownedStringSlice& intrinsicText,
        IRInst* intirnsicInst);

protected:
    const char* _emitSpecial(const char* cursor);

    SourceWriter* m_writer;
    UnownedStringSlice m_text;
    IRCall* m_callInst;
    IRInst* m_intrinsicInst = nullptr;
    IRUse* m_args = nullptr;
    Int m_argCount = 0;
    Index m_openParenCount = 0;
    CLikeSourceEmitter* m_emitter;

    // An arbitrary offset to apply to argument indices.
    //
    // Note: This is a bit of a gross hack to allow the definitions
    // of the texture-sampling operations to be easier to share
    // between combined and non-combined cases.
    //
    // TODO: It would be great to slowly migrate away from needing
    // so much complicated logic here, but if we decide to keep this
    // general approach it would be great to move some of the processing
    // to the front-end and allow things like:
    //
    //      __target_intrinsic(hlsl, "specialOp($a - $b)")
    //      int SomeCoolFunction(int a, int b);
    //
    // That is, we could try to allow direct by-name references to parameters
    // in the intrinsic strings as they appear in the front-end, and then remap
    // those to be index-based as part of translation to the IR.
    //
    Index m_argIndexOffset = 0;
};

} // namespace Slang
#endif
