// slang-ir-byte-address-legalize.cpp
#include "slang-ir-byte-address-legalize.h"

// This file implements an IR pass that translates load/store operations
// on byte-address buffers to be legal for a chosen target.
//
// Currently this pass only applies to the operations generated for
// the generic `*ByteAddressBuffer.Load<T>` and `.Store<T>` operations,
// and not the non-generic versions that traffic in `uint` (e.g.,
// `Load2` or `Store3`).

#include "slang-ir-insts.h"
#include "slang-ir-layout.h"

namespace Slang
{

bool isCPUTarget(TargetRequest* targetReq);

// As is typical for IR passes in Slang, we will encapsulate the state
// while we process the code in a context type.
//
struct ByteAddressBufferLegalizationContext
{
    // We need access to the original session, as well as the options
    // that control what constructs we legalize, and how.
    //
    Session* m_session = nullptr;
    TargetProgram* m_targetProgram = nullptr;
    TargetRequest* m_target = nullptr;
    ByteAddressBufferLegalizationOptions m_options;

    DiagnosticSink* m_sink = nullptr;
    // We will also use a central IR builder when generating new
    // code as part of legalization (rather than create/destroy
    // IR builders on the fly).
    //
    IRModule* m_module;
    IRBuilder m_builder;

    Dictionary<IRInst*, IRType*> byteAddrBufferToReplace;

    // Everything starts with a request to process a module,
    // which delegates to the central recrusive walk of the IR.
    //
    void processModule(IRModule* module)
    {
        m_module = module;
        m_builder = IRBuilder(m_module);

        processInstRec(module->getModuleInst());
    }

    // We recursively walk the entire IR structure (except
    // for decorations), and process any byte-address buffer
    // load or store operations.
    //
    void processInstRec(IRInst* inst)
    {
        switch (inst->getOp())
        {
        case kIROp_ByteAddressBufferLoad:
            processLoad(inst);
            break;

        case kIROp_ByteAddressBufferStore:
            processStore(inst);
            break;

        case kIROp_GetEquivalentStructuredBuffer:
            processGetEquivalentStructuredBuffer(inst);
            break;
        }

        for (IRInst* child : inst->getModifiableChildren())
        {
            processInstRec(child);
        }
    }

    void processGetEquivalentStructuredBuffer(IRInst* inst)
    {
        m_builder.setInsertBefore(inst);

        // We need to see what type it is to be interpreted as.
        auto type = inst->getDataType();

        // We want to determine the element type
        auto structuredBufferType = as<IRHLSLStructuredBufferTypeBase>(type);
        auto elementType = structuredBufferType->getElementType();

        // The buffer is operand 0
        auto buffer = inst->getOperand(0);
        // Get the equivalent structured buffer for the buffer.
        if (auto structuredBuffer = getEquivalentStructuredBuffer(elementType, buffer))
        {
            // We want to replace the the inst, with the equivalent structured buffer reference
            inst->replaceUsesWith(structuredBuffer);
            // Once replaced we don't need anymore
            inst->removeAndDeallocate();
        }
    }

    // The logic for both the load and store cases is similar,
    // so we will present the entire load case first and then
    // move on to stores.
    //
    void processLoad(IRInst* load)
    {
        // What we want to do with a load depends on the type
        // being loaded.
        //
        auto type = load->getDataType();

        // We start by looking at the type being loaded so
        // that we can opt out if it is legal.
        //
        if (isTypeLegalForByteAddressLoadStore(type))
            return;

        // If the type is one that requires legalization,
        // then we will set up to insert new IR instructions
        // to replace it.
        //
        m_builder.setInsertBefore(load);

        // We then emit a "legal load" with the same buffer
        // and byte offset from the original.
        //
        auto buffer = load->getOperand(0);
        auto offset = load->getOperand(1);
        auto alignment = load->getOperand(2);
        auto legalLoad = emitLegalLoad(type, buffer, offset, 0, alignment);

        // If it currently possible for the legalization
        // to fail (perhaps because of something else that
        // is invalid in the IR), so we will defensively
        // leave the code along in that case.
        //
        if (!legalLoad)
            return;

        // If we were able to generate a legal load operation,
        // then the value it yields can be used to fully
        // replace the previous illegal load.
        //
        load->replaceUsesWith(legalLoad);
        load->removeAndDeallocate();
    }

    bool isTypeLegalForByteAddressLoadStore(IRType* type)
    {
        // Whether or not a type is legal to use for
        // byte-address buffer load/store depends on
        // properties of the target, which will have
        // been passed into this pass via its options.
        //
        // If we are expected to translate all byte-address
        // operations to equivalent structured-buffer
        // operations, then that means *no* type is
        // legal for byte-address load/store.
        //
        if (m_options.translateToStructuredBufferOps)
            return false;

        // Basic types are usually legal to load/store
        // on all targets.
        //
        if (auto basicType = as<IRBasicType>(type))
        {
            // On targets that require translation to
            // make all load/store use `uint` values,
            // any scalar type that isn't `uint` is
            // illegal.
            //
            if (m_options.useBitCastFromUInt && basicType->getBaseType() != BaseType::UInt)
            {
                return false;
            }

            // Otherwise, scalar types are assumed
            // legal for load/store.
            //
            return true;
        }

        // Vector types also depend on the options.
        //
        if (as<IRVectorType>(type))
        {
            // If we've been asked to scalarize all
            // vector load/store, then we need to
            // tread them as illegal.
            //
            if (m_options.scalarizeVectorLoadStore)
                return false;
        }

        // All other types are treated as always illegal,
        // so that we will legalize the load/store ops
        // in all cases.
        //
        // Note: recent builds of dxc (perhaps coupled with
        // recent shader models) support byte-address load/store
        // of more complex types, but it is simpler for Slang
        // to just legalize all the composite cases rather
        // than rely on a downstream compiler.
        //
        return false;
    }

    // Helper function to check if the alignment value passed is
    // divisible by the offset at which the resource is indexed into
    // in order to ensure if the load or store can be vectorized.
    bool isAligned(IRInst* offset, IRInst* unknownOffsetAlignment, IRIntegerValue alignmentVal)
    {
        if (auto baseOffsetVal = as<IRIntLit>(offset))
        {
            // If the offset is a constant known at compile time, simply check if it aligned to
            // the elementsize of the underlying resource.
            return (baseOffsetVal->getValue() % alignmentVal) == 0;
        }
        else if (auto alignInst = as<IRIntLit>(unknownOffsetAlignment))
        {
            // If the offset is not known during compile time, use the explicit align
            // field of the overloaded `Load` or `Store` operation or vi `LoadAligned`
            // or `StoreAligned` function.
            //
            // Unaligned `Load`s or `Store`s are identified with 0 alignment, to prevent
            // accidentally issuing a wide vectorized operations.
            if (!alignInst->getValue())
                return false;

            if ((alignInst->getValue() % alignmentVal) == 0)
            {
                return true;
            }
            m_sink->diagnose(
                offset->sourceLoc,
                Slang::Diagnostics::byteAddressBufferUnaligned,
                alignInst->getValue(),
                alignmentVal);
        }
        return false;
    }

    SlangResult getOffset(TargetProgram* target, IRStructField* field, IRIntegerValue* outOffset)
    {
        if (target->getHLSLToVulkanLayoutOptions() &&
            target->getHLSLToVulkanLayoutOptions()->shouldUseGLLayout())
        {
            return getStd430Offset(target->getOptionSet(), field, outOffset);
        }
        return getNaturalOffset(target->getOptionSet(), field, outOffset);
    }

    SlangResult getSizeAndAlignment(
        TargetProgram* target,
        IRType* type,
        IRSizeAndAlignment* outSizeAlignment)
    {
        if (target->getHLSLToVulkanLayoutOptions() &&
            target->getHLSLToVulkanLayoutOptions()->shouldUseGLLayout())
        {
            return getStd430SizeAndAlignment(target->getOptionSet(), type, outSizeAlignment);
        }
        return getNaturalSizeAndAlignment(target->getOptionSet(), type, outSizeAlignment);
    }

    // The core workhorse routine for the load case is `emitLegalLoad`,
    // which tries to emit load operations that read a value of the
    // given `type` from the given `buffer` at the required `baseOffset`
    // plus the `immediateOffset` if any.
    //
    IRInst* emitLegalLoad(
        IRType* type,
        IRInst* buffer,
        IRInst* baseOffset,
        IRIntegerValue immediateOffset,
        IRInst* alignment)
    {
        // The right way to load a value depends primarily
        // on the type, and secondarily on the options
        // that have been specified for this pass.
        //
        if (auto structType = as<IRStructType>(type))
        {
            // When loading a value of `struct` type, we will
            // load each field with its own operation.
            //
            // Note: A more "clever" implementation might try
            // to emit a minimal number of loads of whatever
            // is the largest supported type matching the
            // alignment of `structType`, and then break those
            // loaded values into fields with bit-level ops
            // once they are in registers.
            //
            // Such an approach could conceivably allow more
            // types to be loadable even on targets that
            // don't directly support them (e.g., a structure
            // with an `int` and two `int16_t` could be loadable
            // even when targetting DXBC).
            //
            // The flip side to such an approach would be that
            // it would complicate the generated code, and also
            // make the rules about when a type is supported
            // for byte-address load/store much more complicated.

            // We collect the loaded per-field values into an
            // array, which we will then use to construct the
            // full value of the `struct` type.
            //
            List<IRInst*> fieldVals;
            for (auto field : structType->getFields())
            {
                auto fieldType = field->getFieldType();

                // The relative offset of each field is calculated using
                // the IR-based layout subsystem, which works with the
                // "natural" in-memory layout of types.
                //
                // It is possible for layout computation to fail (e.g.,
                // if the field type somehow wasn't one that can be
                // laid out "naturally"). If the layout process fails,
                // then we fail to legalize this load.
                //
                IRIntegerValue fieldOffset = 0;
                SLANG_RETURN_NULL_ON_FAIL(getOffset(m_targetProgram, field, &fieldOffset));

                // Otherwise, we load the field by recursively calling this function
                // on the field type, with an adjusted immediate offset.
                //
                // If legalizing the field load fails, then we fail the load
                // of the structure as well. Any loads that were generated
                // for earlier fields will be left behind but can be eliminated
                // as dead code.
                //
                auto fieldVal = emitLegalLoad(
                    fieldType,
                    buffer,
                    baseOffset,
                    immediateOffset + fieldOffset,
                    alignment);
                if (!fieldVal)
                    return nullptr;

                fieldVals.add(fieldVal);
            }

            // Once all the field values have been loaded, we can bind
            // then together to make a singel value of the `struct` type,
            // representing the reuslt of the legalized load.
            //
            return m_builder.emitMakeStruct(type, fieldVals);
        }
        else if (auto arrayType = as<IRArrayTypeBase>(type))
        {
            // Loading a value of array type amounts to loading each
            // of its elements. There is shared logic between the
            // array, matrix, and vector cases, so we factor it into
            // a subroutien that we will explain later.
            //
            // We need a known constant number of elements in an array
            // to be able to emit per-element loads, so we skip
            // legalization if the array type isn't in the right form
            // for us to proceed.
            //

            if (auto elementCountInst = as<IRIntLit>(arrayType->getElementCount()))
            {
                // Emit an aligned load operation on an array when using a LoadAligned inst.
                // Else, fallback to scalarizing the loads.
                IRSizeAndAlignment elementLayout;
                SLANG_RELEASE_ASSERT(!getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    arrayType->getElementType(),
                    &elementLayout));
                IRIntegerValue elementStride = elementLayout.getStride();
                auto alignmentVal = elementStride * elementCountInst->getValue();
                if (!isAligned(
                        emitOffsetAddIfNeeded(baseOffset, immediateOffset),
                        alignment,
                        alignmentVal))
                {
                    return emitLegalSequenceLoad(
                        type,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        kIROp_MakeArray,
                        arrayType->getElementType(),
                        elementCountInst->getValue(),
                        alignment);
                }
                else
                {
                    return emitSimpleLoad(type, buffer, baseOffset, immediateOffset);
                }
            }
        }
        else if (auto matType = as<IRMatrixType>(type))
        {
            // Handling a matrix is largely like an array, with the
            // small detail that we need to construct the row type
            // that we expect to load for each "element."
            //
            if (getIntVal(matType->getLayout()) != SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
            {
                auto rowCountInst = as<IRIntLit>(matType->getRowCount());
                if (rowCountInst)
                {
                    auto rowType = m_builder.getVectorType(
                        matType->getElementType(),
                        matType->getColumnCount());
                    return emitLegalSequenceLoad(
                        type,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        kIROp_MakeMatrix,
                        rowType,
                        rowCountInst->getValue(),
                        alignment);
                }
            }
            else
            {
                List<IRInst*> elements;
                auto colCount = (Index)getIntVal(matType->getColumnCount());
                auto rowCount = (Index)getIntVal(matType->getRowCount());
                auto colVectorType = m_builder.getVectorType(matType->getElementType(), rowCount);
                IRSizeAndAlignment colVectorSizeAlignment;
                getSizeAndAlignment(m_targetProgram, colVectorType, &colVectorSizeAlignment);
                for (Index c = 0; c < colCount; c++)
                {
                    auto colVector = emitLegalLoad(
                        colVectorType,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        alignment);
                    for (Index r = 0; r < rowCount; r++)
                    {
                        elements.add(m_builder.emitElementExtract(colVector, (IRIntegerValue)r));
                    }
                    immediateOffset += colVectorSizeAlignment.getStride();
                }
                List<IRInst*> args;
                for (Index r = 0; r < rowCount; r++)
                {
                    for (Index c = 0; c < colCount; c++)
                    {
                        auto index = c * rowCount + r;
                        args.add(elements[index]);
                    }
                }
                return m_builder.emitMakeMatrix(matType, (UInt)args.getCount(), args.getBuffer());
            }
        }
        else if (auto vecType = as<IRVectorType>(type))
        {
            // One of the options that can vary per-target is whether to
            // scalarize vetor load/store operations. When that option
            // is turned on, we can treat a vector load just like an
            // array load.
            //
            if (auto elementCountInst = as<IRIntLit>(vecType->getElementCount()))
            {
                // Emit an aligned vector load operation when using a LoadAligned inst.
                // Else, fallback to scalarizing the loads.
                IRSizeAndAlignment elementLayout;
                SLANG_RELEASE_ASSERT(!getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    vecType->getElementType(),
                    &elementLayout));
                IRIntegerValue elementStride = elementLayout.getStride();
                auto alignmentVal = elementStride * elementCountInst->getValue();
                if (m_options.scalarizeVectorLoadStore ||
                    !isAligned(
                        emitOffsetAddIfNeeded(baseOffset, immediateOffset),
                        alignment,
                        alignmentVal))
                {
                    return emitLegalSequenceLoad(
                        type,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        kIROp_MakeVector,
                        vecType->getElementType(),
                        elementCountInst->getValue(),
                        alignment);
                }
                else
                {
                    return emitSimpleLoad(type, buffer, baseOffset, immediateOffset);
                }
            }

            // If we aren't scalarizing a vetor load then we next need
            // to consider the case where the target might only support
            // byte-address load/store of unsigned integer data (e.g.,
            // this is the case for D3D11/DXBC).
            //
            // We can still support loads of vectors with other element
            // types by first loading the data as unsigned integers, and
            // then bit-casting it to the correct type (e.g., load a
            // `uint4` with `Load4()` and then bit-cast to `float4` using
            // `asfloat()`).
            //
            if (m_options.useBitCastFromUInt)
            {
                // We will look at the element type of the vector (which must
                // be a basic type for this to work).
                //
                if (auto elementType = as<IRBasicType>(vecType->getElementType()))
                {
                    // If there is a distinct unsigned integer type of the
                    // same size as the element type, then we can use that
                    // for our load.
                    //
                    if (auto unsignedElementType = getSameSizeUIntType(elementType))
                    {
                        // We form the appropriate unsigned-integer vector type,
                        // and then emit a load for it.
                        //
                        auto unsignedVecType = m_builder.getVectorType(
                            unsignedElementType,
                            vecType->getElementCount());
                        auto unsignedVecVal =
                            emitSimpleLoad(unsignedVecType, buffer, baseOffset, immediateOffset);

                        // Once we have loaded the bits into a temporary,
                        // we can bit-cast it to the correct tyep and
                        // we have our result.
                        //
                        return m_builder.emitBitCast(vecType, unsignedVecVal);
                    }
                }
            }

            // Any cases of vectors not handled above are allowed to fall through
            // and be handled in the catch-all logic below.
        }
        else if (auto basicType = as<IRBasicType>(type))
        {
            // Most basic scalar types can be handled directly on targets,
            // but as described above for vectors, the D3D11/DXBC target
            // only support loading `uint` values, so we need to emulate
            // loads of other types (like `float`) by first loading a
            // `uint` and then bit-casting.
            //
            if (m_options.useBitCastFromUInt)
            {
                if (auto unsignedType = getSameSizeUIntType(basicType))
                {
                    auto unsignedVal =
                        emitSimpleLoad(unsignedType, buffer, baseOffset, immediateOffset);
                    return m_builder.emitBitCast(basicType, unsignedVal);
                }
            }
        }

        // If none of the many special cases above was triggered, then we
        // are in the base case and assume we want to emit a single load
        // for the type we were given.
        //
        return emitSimpleLoad(type, buffer, baseOffset, immediateOffset);
    }

    // Loading of sequences for arrays, matrices, and vectors is
    // bottlenecked through a single function.
    //
    IRInst* emitLegalSequenceLoad(
        IRType* type,
        IRInst* buffer,
        IRInst* baseOffset,
        IRIntegerValue immediateOffset,
        IROp op,
        IRType* elementType,
        IRIntegerValue elementCount,
        IRInst* alignment)
    {
        // Or goal here is to produce a value of the given `type`, loaded from `buffer`
        // at `baseOffset` plus `immediateOffset`.
        //
        // We will do this by emitting `elementCount` loads for the elements of
        // the given `elementType`, and then grouping them into the final sequence
        // using the given `op` (which will be something like `kIROp_MakeArray`).

        // To know how many bytes to step between loads, we must compute
        // the "stride" of the element type.
        //
        IRSizeAndAlignment elementLayout;
        SLANG_RETURN_NULL_ON_FAIL(getNaturalSizeAndAlignment(
            m_targetProgram->getOptionSet(),
            elementType,
            &elementLayout));
        IRIntegerValue elementStride = elementLayout.getStride();

        // We will collect all the element values into an array so
        // that we can construct the sequence when we are done.
        //
        List<IRInst*> elementVals;
        for (IRIntegerValue ii = 0; ii < elementCount; ++ii)
        {
            auto elementVal = emitLegalLoad(
                elementType,
                buffer,
                baseOffset,
                immediateOffset + ii * elementStride,
                alignment);
            if (!elementVal)
                return nullptr;

            elementVals.add(elementVal);
        }

        // Once we are done loading the elements we construct the sequence value.
        //
        return m_builder
            .emitIntrinsicInst(type, op, elementVals.getCount(), elementVals.getBuffer());
    }

    // All of the loading operations above eventually bottom out at `emitSimpleLoad`,
    // which is meant to handle the base case where we do *not* want to
    // recurse on the structure of `type`.
    //
    IRInst* emitSimpleLoad(
        IRType* type,
        IRInst* buffer,
        IRInst* baseOffset,
        IRIntegerValue immediateOffset)
    {
        // For all of the operations above this in the call chain we have been
        // tracking a pair of a `baseOffset` as an IR instruction, and an
        // `immediateOffset` value. Keeping things split avoided introducing
        // a bunch of `add` instructions that could be constant-folded away.
        //
        // Instead, now that we are about to emit a load "for real"
        // we want to turn those two offset values into one.
        //
        IRInst* offset = emitOffsetAddIfNeeded(baseOffset, immediateOffset);

        // At this point there is one last (major) detail we need to
        // get into, which is that some targets (currently just GLSL)
        // do not actually have anything like byte-address buffers
        // as a built-in feature.
        //
        // Instead, GLSL has "shader storage buffers" which are
        // tied to a particular element type when declared. E.g.,:
        //
        //      buffer MyBuffer { uint _data[]; } myBuffer;
        //
        // The `myBuffer` declaration above can be used to load
        // `uint` values, but isn't much use if you want to load/store
        // a `half` or a `double` efficiently (and atomically,
        // where possible/guaranteed).
        //
        // Shader storage buffers like this are closer in spirit to
        // HLSL/Slang "structured buffers," so we think of this code
        // path as converting byte-address buffer operations into
        // structured-buffer operations.
        //
        // To make things work for GLSL output, we need to generate
        // multiple `buffer` declarations that all alias one another
        // (accomplished by giving them the same `binding`), but that
        // declare buffers with different element types.
        //
        if (m_options.translateToStructuredBufferOps)
        {
            // In order to emit a suitable structured-buffer load,
            // we need to find or create a global declaration for
            // a structured buffer that is "equivalent" to `buffer`,
            // but has `type` as its element type.
            //
            // That operation could conceivably fail, and when it
            // does we will fall back to the default handling of
            // emitting a byte-address buffer load (which will
            // then fail to generate valid GLSL code).
            //
            if (auto structuredBuffer = getEquivalentStructuredBuffer(type, buffer))
            {
                // The `offset` instruction represents the byte offset of
                // the thing we are trying to load, and we need to translate
                // that into an *index* for use with a structured buffer.
                //
                // We convert the offset to an index by dividing by the
                // stride of `type` as computed with our "natural layout" rules.
                //
                // This logic will be invalid if `offset` isn't a multiple of
                // the stride of `type`, but that case would have been
                // undefined behavior anyway.
                //
                auto offsetType = offset->getDataType();

                IRSizeAndAlignment typeLayout;
                SLANG_RETURN_NULL_ON_FAIL(
                    getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), type, &typeLayout));
                auto typeStrideVal = typeLayout.getStride();

                auto typeStrideInst = m_builder.getIntValue(offsetType, typeStrideVal);
                IRInst* divArgs[] = {offset, typeStrideInst};
                auto index = m_builder.emitIntrinsicInst(offsetType, kIROp_Div, 2, divArgs);

                IRInst* args[] = {structuredBuffer, index};
                return m_builder.emitIntrinsicInst(type, kIROp_StructuredBufferLoad, 2, args);
            }
        }

        if (m_options.lowerBasicTypeOps)
        {
            // Some platforms e.g. Metal does not allow loading basic types that are not 4-byte
            // sized. We need to lower such loads.
            IRSizeAndAlignment sizeAlignment;
            SLANG_RETURN_NULL_ON_FAIL(
                getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), type, &sizeAlignment));
            if (sizeAlignment.size == 8)
            {
                // We need to load the value as two 4-byte values and then combine them.
                auto loOffset = offset;
                auto hiOffset = emitOffsetAddIfNeeded(offset, 4);
                IRInst* loadLoArgs[] = {buffer, loOffset};
                IRInst* loadHiArgs[] = {buffer, hiOffset};
                auto loLoad = m_builder.emitIntrinsicInst(
                    m_builder.getUIntType(),
                    kIROp_ByteAddressBufferLoad,
                    2,
                    loadLoArgs);
                auto hiLoad = m_builder.emitIntrinsicInst(
                    m_builder.getUIntType(),
                    kIROp_ByteAddressBufferLoad,
                    2,
                    loadHiArgs);
                auto lo64 = m_builder.emitCast(m_builder.getUInt64Type(), loLoad);
                auto hi64 = m_builder.emitCast(m_builder.getUInt64Type(), hiLoad);
                auto shift = m_builder.emitShl(
                    m_builder.getUInt64Type(),
                    hi64,
                    m_builder.getIntValue(m_builder.getUInt64Type(), 32));
                auto fullValue = m_builder.emitBitOr(m_builder.getUInt64Type(), lo64, shift);
                return m_builder.emitBitCast(type, fullValue);
            }
            else if (sizeAlignment.size < 4)
            {
                auto alignedOffset = m_builder.emitDiv(
                    offset->getDataType(),
                    offset,
                    m_builder.getIntValue(offset->getDataType(), 4));
                alignedOffset = m_builder.emitMul(
                    offset->getDataType(),
                    alignedOffset,
                    m_builder.getIntValue(offset->getDataType(), 4));
                IRInst* loadArgs[] = {buffer, alignedOffset};
                auto val = m_builder.emitIntrinsicInst(
                    m_builder.getUIntType(),
                    kIROp_ByteAddressBufferLoad,
                    2,
                    loadArgs);
                auto shiftAmount = m_builder.emitSub(offset->getDataType(), offset, alignedOffset);
                shiftAmount = m_builder.emitMul(
                    offset->getDataType(),
                    shiftAmount,
                    m_builder.getIntValue(offset->getDataType(), 8));
                IRInst* mask = nullptr;
                switch (sizeAlignment.size)
                {
                case 1:
                    mask = m_builder.getIntValue(m_builder.getUIntType(), 0xFF);
                    break;
                case 2:
                    mask = m_builder.getIntValue(m_builder.getUIntType(), 0xFFFF);
                    break;
                default:
                    SLANG_ASSERT(!"Unexpected size");
                    break;
                }
                auto shift = m_builder.emitShr(m_builder.getUIntType(), val, shiftAmount);
                auto masked = m_builder.emitBitAnd(m_builder.getUIntType(), shift, mask);
                IRInst* casted = nullptr;
                switch (sizeAlignment.size)
                {
                case 1:
                    casted = m_builder.emitCast(m_builder.getUInt8Type(), masked);
                    break;
                case 2:
                    casted = m_builder.emitCast(m_builder.getUInt16Type(), masked);
                    break;
                default:
                    SLANG_ASSERT(!"Unexpected size");
                    break;
                }
                return m_builder.emitBitCast(type, casted);
            }
        }
        // When we finally run out of special cases to handle, we just emit
        // a byte-address buffer load operation directly, assuming it will
        // work for the chosen target.
        //
        {
            IRInst* loadArgs[] = {buffer, offset};
            return m_builder.emitIntrinsicInst(type, kIROp_ByteAddressBufferLoad, 2, loadArgs);
        }
    }

    IRInst* emitOffsetAddIfNeeded(IRInst* baseOffset, IRIntegerValue immediateOffset)
    {
        // We need to create an instruction to represent
        // `baseOffset` plus `immediateOffset`.
        //
        // An important special case is when `immediateOffset` is zero:
        //
        if (immediateOffset == 0)
            return baseOffset;

        // Otherwise, we emit an `add` instruction of the appropriate type
        //
        auto type = baseOffset->getDataType();
        IRInst* args[] = {baseOffset, m_builder.getIntValue(type, immediateOffset)};
        return m_builder.emitIntrinsicInst(type, kIROp_Add, 2, args);
    }

    // At this point we have gone through the main logic of the load path,
    // and before we turn our attention to the store path we can go
    // ahead and define some of the utility functions that the code above
    // requires.

    // In order to handle interesting types on D3D11/DXBC, we need to
    // be able to map a base type to another type of the same size.
    //
    BaseType getSameSizeUIntBaseType(IROp op)
    {
        // For now we are only handling the 32-bit types here, because
        // the D3D11/DXBC target will not be able to handle 16- or
        // 64-bit types anyway. This could be improved over time
        // if needed.
        //
        switch (op)
        {
        case kIROp_IntType:
        case kIROp_FloatType:
        case kIROp_BoolType:
            // The basic 32-bit types (and `bool`) can be handled by
            // loading `uint` values and then bit-casting.
            //
            // Note: We aren't listing `kIROp_UIntType` here because
            // we don't want to introduce a bit-cast in the case where
            // the load was already for a `uint`.
            //
            return BaseType::UInt;
        case kIROp_Int8Type:
        case kIROp_UInt8Type:
            return BaseType::UInt8;
        case kIROp_Int16Type:
        case kIROp_UInt16Type:
        case kIROp_HalfType:
            return BaseType::UInt16;
        case kIROp_Int64Type:
        case kIROp_UInt64Type:
        case kIROp_DoubleType:
            return BaseType::UInt64;
        case kIROp_IntPtrType:
        case kIROp_UIntPtrType:
        case kIROp_RawPointerType:
        case kIROp_PtrType:
            if (isCPUTarget(m_target) && sizeof(void*) == 4)
                return BaseType::UInt;
            else
                return BaseType::UInt64;
        default:
            // All other types map to a sentinel value of `Void` to
            // indicate that a bit-cast solution shouldn't be attempted:
            // either load the original type, or fail.
            //
            return BaseType::Void;
        }
    }
    IRBasicType* getSameSizeUIntType(IRType* type)
    {
        auto unsignedBaseType = getSameSizeUIntBaseType(type->getOp());
        if (unsignedBaseType == BaseType::Void)
            return nullptr;

        return m_builder.getBasicType(unsignedBaseType);
    }

    // When replacing byte-address buffer load/store operations with
    // structured buffer ones, we need to be able to map an IR instruction
    // that represents a byte-address buffer to one that represents an
    // "equivalent" structured buffer.
    //
    // An important/tricky detail here is that the byte-address buffer
    // might have been passed in as a function parameter, or be indexed
    // from an array, etc.
    //
    // The logic here assumes this pass has run after a full legalization
    // pass on resource parameter usage, so that any references to
    // buffers in an instruction are "grounded" in a known global shader
    // parameter.

    IRInst* getEquivalentStructuredBuffer(IRType* elementType, IRInst* byteAddressBuffer)
    {
        if (this->m_options.treatGetEquivalentStructuredBufferAsGetThis)
            return byteAddressBuffer;

        if (!elementType)
        {
            return nullptr;
        }
        if (as<IRHLSLStructuredBufferTypeBase>(byteAddressBuffer->getDataType()))
            return byteAddressBuffer;
        // The simple case for replacement is when the byte-address buffer to
        // be replaced is a global shader parameter. That path will get its
        // own routine.
        if (auto byteAddressBufferParam = as<IRGlobalParam>(byteAddressBuffer))
        {
            return getEquivalentStructuredBufferParam(elementType, byteAddressBufferParam);
        }

        if (byteAddressBuffer->getOp() == kIROp_GetElement)
        {
            // If the code is fetching the byte-address buffer from an
            // array, then we need to create an "equivalent" structured
            // buffer array, and then index into that.
            //
            auto byteAddressBufferArray = byteAddressBuffer->getOperand(0);
            auto index = byteAddressBuffer->getOperand(1);

            auto structuredBufferArray =
                getEquivalentStructuredBuffer(elementType, byteAddressBufferArray);
            if (!structuredBufferArray)
                return nullptr;

            auto structuredBufferArrayType =
                as<IRArrayTypeBase>(structuredBufferArray->getDataType());
            if (!structuredBufferArrayType)
                return nullptr;

            // If we succeeded in creating a declaration for an array of
            // structured buffers to index into, we can now emit a new
            // operation to index into that array instead, and the result
            // will work as our "equivalent" structured buffer.
            //
            return m_builder.emitElementExtract(
                structuredBufferArrayType->getElementType(),
                structuredBufferArray,
                index);
        }

        // If we failed to pattern-match the byte-address buffer operand
        // against something we can handle, then we need to bail out
        // of our attempt to legalize things here.
        //
        // TODO: Should we make this case an error?
        //
        return nullptr;
    }

    // Our seach for an "equivalent" structured buffer should bottom out when
    // we find a global shader parameter of byte-address buffer type, or an
    // array (of array of array of ...) byte-address buffer type.
    //
    // We then want to create an equivalent shader parameter of a matching
    // structured buffer (or array...) type.
    //
    // To avoid creating too many buffers (e.g., one per load), we will cache and
    // re-use the buffers we declare in this way. Note that we do *not* need
    // to worry if the deduplication is perfect, because we are already assuming
    // that the target will handle multiple buffers with the same `binding`
    // correctly.
    //
    Dictionary<KeyValuePair<IRInst*, IRInst*>, IRGlobalParam*> m_cachedStructuredBuffers;
    IRGlobalParam* getEquivalentStructuredBufferParam(
        IRType* elementType,
        IRGlobalParam* byteAddressBufferParam)
    {
        KeyValuePair<IRInst*, IRInst*> key(elementType, byteAddressBufferParam);

        IRGlobalParam* structuredBufferParam;
        if (!m_cachedStructuredBuffers.tryGetValue(key, structuredBufferParam))
        {
            structuredBufferParam =
                createEquivalentStructuredBufferParam(elementType, byteAddressBufferParam);
            m_cachedStructuredBuffers.add(key, structuredBufferParam);
        }
        return structuredBufferParam;
    }

    void cloneBufferDecorations(IRBuilder& builder, IRInst* dest, IRInst* src)
    {
        for (auto decoration : src->getDecorations())
        {
            switch (decoration->getOp())
            {
            case kIROp_MemoryQualifierSetDecoration:
                builder.addMemoryQualifierSetDecoration(
                    dest,
                    as<IRMemoryQualifierSetDecoration>(decoration)->getMemoryQualifierBit());
                break;
            default:
                break;
            }
        }
    }

    IRGlobalParam* createEquivalentStructuredBufferParam(
        IRType* elementType,
        IRGlobalParam* byteAddressBufferParam)
    {
        // When we need to create a new structured buffer to stand in for
        // some byte-address buffer (with a new `elementType` being used
        // for load/store), we need to figure out the "equivalent" type
        // to use for the new buffer.
        //
        auto byteAddressBufferParamType = byteAddressBufferParam->getDataType();
        auto structuredBufferParamType =
            getEquivalentStructuredBufferParamType(elementType, byteAddressBufferParamType);
        if (!structuredBufferParamType)
            return nullptr;

        // Next we will create a global shader parameter using the new
        // type.
        //
        // Note: we are creating a new `IRBuilder` here rather than using
        // `m_builder` because this logic could get called during the middle
        // of legalizing a load or store, and we don't want to mess with
        // the insertion location of `m_builder`.
        //
        IRBuilder paramBuilder(m_module);
        paramBuilder.setInsertBefore(byteAddressBufferParam);

        auto structuredBufferParam = paramBuilder.createGlobalParam(structuredBufferParamType);
        if (auto nameHint = byteAddressBufferParam->findDecoration<IRNameHintDecoration>())
            paramBuilder.addNameHintDecoration(structuredBufferParam, nameHint->getName());
        if (auto typeHint = byteAddressBufferParam->findDecoration<IRUserTypeNameDecoration>())
            paramBuilder.addUserTypeNameDecoration(
                structuredBufferParam,
                typeHint->getUserTypeName());

        // The new parameter needs to be given a layout to match the existing
        // parameter, so that it is given the same `binding` in the generated code.
        //
        if (auto layoutDecoration = byteAddressBufferParam->findDecoration<IRLayoutDecoration>())
        {
            paramBuilder.addLayoutDecoration(structuredBufferParam, layoutDecoration->getLayout());
        }
        cloneBufferDecorations(paramBuilder, structuredBufferParam, byteAddressBufferParam);
        return structuredBufferParam;
    }

    IRType* getEquivalentStructuredBufferParamType(
        IRType* elementType,
        IRType* byteAddressBufferType)
    {
        // Our task in this function is to compute the type for
        // a structure buffer that is equivalent to `byteAddressBufferType`,
        // but with the given `elementType`.
        IRInst* typeOperands[2] = {elementType, m_builder.getType(kIROp_DefaultBufferLayoutType)};
        switch (byteAddressBufferType->getOp())
        {
            // The basic `*ByteAddressBuffer` types map directly to the
            // `*StructuredBuffer<elementType>` cases.
        case kIROp_HLSLByteAddressBufferType:
            return m_builder.getType(kIROp_HLSLStructuredBufferType, 2, typeOperands);
        case kIROp_HLSLRWByteAddressBufferType:
            return m_builder.getType(kIROp_HLSLRWStructuredBufferType, 2, typeOperands);
        case kIROp_HLSLRasterizerOrderedByteAddressBufferType:
            return m_builder.getType(
                kIROp_HLSLRasterizerOrderedStructuredBufferType,
                2,
                typeOperands);

        case kIROp_ArrayType:
        case kIROp_UnsizedArrayType:
            {
                // Array types (both sized and unsized) need to translate
                // their element type to an equivalent structured buffer
                // and build a new array type with the same element count.
                //
                auto arrayType = cast<IRArrayTypeBase>(byteAddressBufferType);
                return m_builder.getArrayTypeBase(
                    byteAddressBufferType->getOp(),
                    getEquivalentStructuredBufferParamType(
                        elementType,
                        arrayType->getElementType()),
                    arrayType->getElementCount());
            }

        default:
            return nullptr;
        }
    }

    // At this point we've covered all the logic for the load case down
    // to the last detail.
    //
    // All that remains is to go over the equivalent logic for the case
    // of byte-address buffer stores, which mostly parallels code we've
    // already discussed.

    void processStore(IRInst* store)
    {
        // Just as for loads, the logic for stores is base don the type
        // being used, but unlike in the load case we don't care about
        // the type of the store operation, but instead the operand
        // that represents the value to be stored.
        //
        auto value = store->getOperand(3);
        auto type = value->getDataType();

        // Types that are already legal to use don't require any processing.
        //
        if (isTypeLegalForByteAddressLoadStore(type))
            return;

        // Otherwise we set up to try and emit a replacement.
        //
        m_builder.setInsertBefore(store);

        // It is possible that our attempt to emit a replacement will fail
        // (this should only happen if we run into types that shouldn't
        // actually be allowed on a target), and in those cases we will
        // leave the original store around as well (this is at worst a
        // performance issue, but we should still consider trying to
        // tighten this up and make all uhandled cases be hard errors).
        //
        auto result = emitLegalStore(
            type,
            store->getOperand(0),
            store->getOperand(1),
            0,
            store->getOperand(2),
            value);
        if (SLANG_FAILED(result))
            return;

        store->removeAndDeallocate();
    }

    Result emitLegalStore(
        IRType* type,
        IRInst* buffer,
        IRInst* baseOffset,
        IRIntegerValue immediateOffset,
        IRInst* alignment,
        IRInst* value)
    {
        // The flow for emitting a legal store is very similar to that for
        // legal loads; we will recurse on the structure of `type` and
        // emit stores for fields/elements as needed.

        if (auto structType = as<IRStructType>(type))
        {
            // To store a structure, we store each of its fields at
            // the appropriate relative offset.
            //
            for (auto field : structType->getFields())
            {
                auto fieldType = field->getFieldType();

                IRIntegerValue fieldOffset;
                SLANG_RETURN_ON_FAIL(getOffset(m_targetProgram, field, &fieldOffset));

                auto fieldVal = m_builder.emitFieldExtract(fieldType, value, field->getKey());
                SLANG_RETURN_ON_FAIL(emitLegalStore(
                    fieldType,
                    buffer,
                    baseOffset,
                    immediateOffset + fieldOffset,
                    alignment,
                    fieldVal));
            }
            return SLANG_OK;
        }
        else if (auto arrayType = as<IRArrayTypeBase>(type))
        {
            // Arrays and other sequences bottleneck through a helper
            // function, which we will cover later.
            //
            if (auto elementCountInst = as<IRIntLit>(arrayType->getElementCount()))
            {
                // Emit an aligned store operation on an array when using a StoreAligned inst.
                // Else, fallback to scalarizing the stores.
                IRSizeAndAlignment elementLayout;
                SLANG_RELEASE_ASSERT(!getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    arrayType->getElementType(),
                    &elementLayout));
                IRIntegerValue elementStride = elementLayout.getStride();
                auto alignmentVal = elementStride * elementCountInst->getValue();
                if (!isAligned(
                        emitOffsetAddIfNeeded(baseOffset, immediateOffset),
                        alignment,
                        alignmentVal))
                {
                    return emitLegalSequenceStore(
                        buffer,
                        baseOffset,
                        immediateOffset,
                        value,
                        arrayType->getElementType(),
                        elementCountInst->getValue(),
                        alignment);
                }
                else
                {
                    return emitSimpleStore(
                        value->getDataType(),
                        buffer,
                        baseOffset,
                        immediateOffset,
                        value);
                }
            }
        }
        else if (auto matType = as<IRMatrixType>(type))
        {
            auto layout = getIntVal(matType->getLayout());
            if (layout != SLANG_MATRIX_LAYOUT_COLUMN_MAJOR)
            {
                auto rowCountInst = as<IRIntLit>(matType->getRowCount());
                if (rowCountInst)
                {
                    auto rowType = m_builder.getVectorType(
                        matType->getElementType(),
                        matType->getColumnCount());
                    return emitLegalSequenceStore(
                        buffer,
                        baseOffset,
                        immediateOffset,
                        value,
                        rowType,
                        rowCountInst->getValue(),
                        alignment);
                }
            }
            else
            {
                auto colCount = (Index)getIntVal(matType->getColumnCount());
                auto rowCount = (Index)getIntVal(matType->getRowCount());
                List<IRInst*> srcRows;
                for (Index r = 0; r < rowCount; r++)
                    srcRows.add(m_builder.emitElementExtract(value, (IRIntegerValue)r));
                for (Index c = 0; c < colCount; c++)
                {
                    List<IRInst*> colVectorArgs;
                    for (Index r = 0; r < rowCount; r++)
                    {
                        auto rowVector = srcRows[r];
                        auto element = m_builder.emitElementExtract(rowVector, (IRIntegerValue)c);
                        colVectorArgs.add(element);
                    }
                    auto colVectorType =
                        m_builder.getVectorType(matType->getElementType(), rowCount);
                    auto colVector = m_builder.emitMakeVector(colVectorType, colVectorArgs);
                    IRSizeAndAlignment colVectorSizeAlignment;
                    getSizeAndAlignment(m_targetProgram, colVectorType, &colVectorSizeAlignment);
                    emitLegalStore(
                        colVectorType,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        alignment,
                        colVector);
                    immediateOffset += colVectorSizeAlignment.getStride();
                }
                return SLANG_OK;
            }
        }
        else if (auto vecType = as<IRVectorType>(type))
        {
            if (auto elementCountInst = as<IRIntLit>(vecType->getElementCount()))
            {
                // Emit an aligned vector store operation when using a StoreAligned inst.
                // Else, fallback to scalarizing the stores.

                IRSizeAndAlignment elementLayout;
                SLANG_RELEASE_ASSERT(!getNaturalSizeAndAlignment(
                    m_targetProgram->getOptionSet(),
                    vecType->getElementType(),
                    &elementLayout));
                IRIntegerValue elementStride = elementLayout.getStride();
                auto alignmentVal = elementStride * elementCountInst->getValue();
                if (m_options.scalarizeVectorLoadStore ||
                    !isAligned(
                        emitOffsetAddIfNeeded(baseOffset, immediateOffset),
                        alignment,
                        alignmentVal))
                {
                    return emitLegalSequenceStore(
                        buffer,
                        baseOffset,
                        immediateOffset,
                        value,
                        vecType->getElementType(),
                        elementCountInst->getValue(),
                        alignment);
                }
                else
                {
                    return emitSimpleStore(
                        value->getDataType(),
                        buffer,
                        baseOffset,
                        immediateOffset,
                        value);
                }
            }

            if (m_options.useBitCastFromUInt)
            {
                auto elementType = as<IRBasicType>(vecType->getElementType());
                if (auto unsignedElementType = getSameSizeUIntType(elementType))
                {
                    // The bit-cast case for stores is similar to the case
                    // for loads, except that we cast the value before
                    // storing it (instead of casting a value after loading).
                    //
                    auto unsignedVecType =
                        m_builder.getVectorType(unsignedElementType, vecType->getElementCount());
                    auto unsignedVecVal = m_builder.emitBitCast(unsignedVecType, value);
                    return emitSimpleStore(
                        unsignedVecType,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        unsignedVecVal);
                }
            }
        }
        else if (auto basicType = as<IRBasicType>(type))
        {
            if (m_options.useBitCastFromUInt)
            {
                if (auto unsignedType = getSameSizeUIntType(basicType))
                {
                    auto unsignedVal = m_builder.emitBitCast(unsignedType, value);
                    return emitSimpleStore(
                        unsignedType,
                        buffer,
                        baseOffset,
                        immediateOffset,
                        unsignedVal);
                }
            }
        }

        return emitSimpleStore(type, buffer, baseOffset, immediateOffset, value);
    }

    Result emitSimpleStore(
        IRType* type,
        IRInst* buffer,
        IRInst* baseOffset,
        IRIntegerValue immediateOffset,
        IRInst* value)
    {
        IRInst* offset = emitOffsetAddIfNeeded(baseOffset, immediateOffset);
        if (m_options.translateToStructuredBufferOps)
        {
            if (auto structuredBuffer = getEquivalentStructuredBuffer(type, buffer))
            {
                // Similar to the load case, if we are replacing byte-address
                // buffers with structured buffers, then once we find the
                // "equivalent" buffer to use, we emit a structured-buffer store,
                // with an index computed by dividing the offset by the stride.
                //
                auto indexType = offset->getDataType();

                IRSizeAndAlignment typeLayout;
                SLANG_RETURN_ON_FAIL(
                    getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), type, &typeLayout));

                auto typeStride = m_builder.getIntValue(indexType, typeLayout.getStride());

                IRInst* divArgs[] = {offset, typeStride};
                auto index = m_builder.emitIntrinsicInst(indexType, kIROp_Div, 2, divArgs);

                IRInst* args[] = {structuredBuffer, index, value};
                m_builder.emitIntrinsicInst(type, kIROp_RWStructuredBufferStore, 3, args);
                return SLANG_OK;
            }
        }
        if (m_options.lowerBasicTypeOps)
        {
            // Some platforms e.g. Metal does not allow storing basic types that are not 4-byte
            // sized. We need to lower such loads.
            IRSizeAndAlignment sizeAlignment;
            SLANG_RETURN_ON_FAIL(
                getNaturalSizeAndAlignment(m_targetProgram->getOptionSet(), type, &sizeAlignment));
            if (sizeAlignment.size == 8)
            {
                // We need to store the value as two 4-byte values.
                auto uint64Val = m_builder.emitBitCast(m_builder.getUInt64Type(), value);
                auto loVal = m_builder.emitCast(m_builder.getUIntType(), uint64Val);
                auto hiVal = m_builder.emitCast(
                    m_builder.getUIntType(),
                    m_builder.emitShr(
                        m_builder.getUInt64Type(),
                        uint64Val,
                        m_builder.getIntValue(m_builder.getUInt64Type(), 32)));
                auto loOffset = offset;
                auto hiOffset = emitOffsetAddIfNeeded(offset, 4);
                IRInst* alignment = m_builder.getIntValue(m_builder.getUIntType(), 0);
                m_builder.emitByteAddressBufferStore(buffer, loOffset, alignment, loVal);
                m_builder.emitByteAddressBufferStore(buffer, hiOffset, alignment, hiVal);
                return SLANG_OK;
            }
            else if (sizeAlignment.size < 4)
            {
                IRInst* loadArgs[] = {buffer, offset};
                auto existingVal = m_builder.emitIntrinsicInst(
                    m_builder.getUIntType(),
                    kIROp_ByteAddressBufferLoad,
                    2,
                    loadArgs);
                auto alignedOffset = m_builder.emitDiv(
                    offset->getDataType(),
                    offset,
                    m_builder.getIntValue(offset->getDataType(), 4));
                alignedOffset = m_builder.emitMul(
                    offset->getDataType(),
                    alignedOffset,
                    m_builder.getIntValue(offset->getDataType(), 4));
                auto shiftAmount = m_builder.emitSub(offset->getDataType(), offset, alignedOffset);
                shiftAmount = m_builder.emitMul(
                    offset->getDataType(),
                    shiftAmount,
                    m_builder.getIntValue(offset->getDataType(), 8));
                auto uintVal = m_builder.emitCast(
                    m_builder.getUIntType(),
                    m_builder.emitBitCast(getSameSizeUIntType(value->getDataType()), value));
                auto shiftedData = m_builder.emitShl(m_builder.getUIntType(), uintVal, shiftAmount);
                IRInst* mask = nullptr;
                switch (sizeAlignment.size)
                {
                case 1:
                    mask = m_builder.getIntValue(m_builder.getUIntType(), 0xFF);
                    break;
                case 2:
                    mask = m_builder.getIntValue(m_builder.getUIntType(), 0xFFFF);
                    break;
                default:
                    SLANG_ASSERT(!"Unexpected size");
                    return SLANG_FAIL;
                }
                mask = m_builder.emitShl(m_builder.getUIntType(), mask, shiftAmount);
                mask = m_builder.emitBitNot(m_builder.getUIntType(), mask);
                auto maskedData = m_builder.emitBitAnd(m_builder.getUIntType(), existingVal, mask);
                auto newData =
                    m_builder.emitBitOr(m_builder.getUIntType(), maskedData, shiftedData);
                m_builder.emitByteAddressBufferStore(buffer, alignedOffset, newData);
                return SLANG_OK;
            }
        }
        {
            m_builder.emitByteAddressBufferStore(buffer, offset, value);
            return SLANG_OK;
        }
    }

    Result emitLegalSequenceStore(
        IRInst* buffer,
        IRInst* baseOffset,
        IRIntegerValue immediateOffset,
        IRInst* value,
        IRType* elementType,
        IRIntegerValue elementCount,
        IRInst* alignment)
    {
        // The store case for sequences is similar to the load case.
        //
        // We iterate over the elements and fetch then store each one.
        //
        IRSizeAndAlignment elementLayout;
        SLANG_RETURN_ON_FAIL(getNaturalSizeAndAlignment(
            m_targetProgram->getOptionSet(),
            elementType,
            &elementLayout));
        IRIntegerValue elementStride = elementLayout.getStride();

        auto indexType = m_builder.getIntType();
        for (IRIntegerValue ii = 0; ii < elementCount; ++ii)
        {
            auto elementIndex = m_builder.getIntValue(indexType, ii);
            auto elementVal = m_builder.emitElementExtract(elementType, value, elementIndex);
            SLANG_RETURN_ON_FAIL(emitLegalStore(
                elementType,
                buffer,
                baseOffset,
                immediateOffset + ii * elementStride,
                alignment,
                elementVal));
        }

        return SLANG_OK;
    }
};


void legalizeByteAddressBufferOps(
    Session* session,
    TargetProgram* program,
    IRModule* module,
    DiagnosticSink* sink,
    ByteAddressBufferLegalizationOptions const& options)
{
    ByteAddressBufferLegalizationContext context;
    context.m_session = session;
    context.m_target = program->getTargetReq();
    context.m_options = options;
    context.m_targetProgram = program;
    context.m_sink = sink;
    context.processModule(module);
}

} // namespace Slang
