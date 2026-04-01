// slang-emit-glsl.h
#ifndef SLANG_EMIT_GLSL_H
#define SLANG_EMIT_GLSL_H

#include "slang-emit-c-like.h"
#include "slang-extension-tracker.h"

namespace Slang
{

class GLSLSourceEmitter : public CLikeSourceEmitter
{
public:
    typedef CLikeSourceEmitter Super;

    virtual SlangResult init() SLANG_OVERRIDE;

    GLSLSourceEmitter(const Desc& desc);

    virtual RefObject* getExtensionTracker() SLANG_OVERRIDE { return m_glslExtensionTracker; }

protected:
    virtual void beforeComputeEmitActions(IRModule* module) SLANG_OVERRIDE;
    virtual void emitParameterGroupImpl(IRGlobalParam* varDecl, IRUniformParameterGroupType* type)
        SLANG_OVERRIDE;
    virtual void emitEntryPointAttributesImpl(
        IRFunc* irFunc,
        IREntryPointDecoration* entryPointDecor) SLANG_OVERRIDE;
    virtual void emitImageFormatModifierImpl(IRInst* varDecl, IRType* varType) SLANG_OVERRIDE;
    virtual void emitLayoutQualifiersImpl(IRVarLayout* layout) SLANG_OVERRIDE;

    virtual void emitSubpassInputTypeImpl(IRSubpassInputType* type) SLANG_OVERRIDE
    {
        _emitGLSLSubpassInputType(type);
    }
    virtual void emitTextureOrTextureSamplerTypeImpl(IRTextureTypeBase* type, char const* baseName)
        SLANG_OVERRIDE
    {
        _emitGLSLTextureOrTextureSamplerType(type, baseName);
    }

    virtual void emitFrontMatterImpl(TargetRequest* targetReq) SLANG_OVERRIDE;

    virtual void emitRateQualifiersAndAddressSpaceImpl(IRRate* rate, AddressSpace addressSpace)
        SLANG_OVERRIDE;
    virtual void emitInterpolationModifiersImpl(
        IRInst* varInst,
        IRType* valueType,
        IRVarLayout* layout) SLANG_OVERRIDE;
    virtual void emitPackOffsetModifier(
        IRInst* varInst,
        IRType* valueType,
        IRPackOffsetDecoration* decoration) SLANG_OVERRIDE;

    virtual void emitMemoryQualifiers(IRInst* varInst) SLANG_OVERRIDE;
    virtual void emitStructFieldAttributes(
        IRStructType* structType,
        IRStructField* field,
        bool allowOffsetLayout) SLANG_OVERRIDE;
    virtual void emitMeshShaderModifiersImpl(IRInst* varInst) SLANG_OVERRIDE;
    virtual void emitSimpleTypeImpl(IRType* type) SLANG_OVERRIDE;
    virtual void emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount)
        SLANG_OVERRIDE;
    virtual void emitVarDecorationsImpl(IRInst* varDecl) SLANG_OVERRIDE;
    virtual void emitMatrixLayoutModifiersImpl(IRType* varType) SLANG_OVERRIDE;
    virtual void emitTypeImpl(IRType* type, const StringSliceLoc* nameAndLoc) SLANG_OVERRIDE;
    virtual void emitParamTypeImpl(IRType* type, String const& name) SLANG_OVERRIDE;
    virtual void emitFuncDecorationImpl(IRDecoration* decoration) SLANG_OVERRIDE;
    virtual void emitGlobalParamDefaultVal(IRGlobalParam* decl) SLANG_OVERRIDE;

    virtual void emitBitfieldExtractImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual void emitBitfieldInsertImpl(IRInst* inst) SLANG_OVERRIDE;

    virtual void handleRequiredCapabilitiesImpl(IRInst* inst) SLANG_OVERRIDE;

    virtual bool tryEmitGlobalParamImpl(IRGlobalParam* varDecl, IRType* varType) SLANG_OVERRIDE;
    virtual bool tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec) SLANG_OVERRIDE;
    virtual bool tryEmitInstStmtImpl(IRInst* inst) SLANG_OVERRIDE;

    virtual void emitGlobalInstImpl(IRInst* inst) override;
    void emitBufferPointerTypeDefinition(IRInst* ptrType);

    virtual void emitSimpleValueImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual void emitLoopControlDecorationImpl(IRLoopControlDecoration* decl) SLANG_OVERRIDE;

    void _emitMemoryQualifierDecorations(IRInst* varDecl);
    void _emitGLSLSubpassInputType(IRSubpassInputType* type);
    void _emitGLSLTextureOrTextureSamplerType(IRTextureTypeBase* type, char const* baseName);
    void _emitGLSLStructuredBuffer(
        IRGlobalParam* varDecl,
        IRHLSLStructuredBufferTypeBase* structuredBufferType);

    void _emitGLSLByteAddressBuffer(
        IRGlobalParam* varDecl,
        IRByteAddressBufferTypeBase* byteAddressBufferType);
    void _emitGLSLParameterGroup(IRGlobalParam* varDecl, IRUniformParameterGroupType* type);
    void _emitGLSLSSBO(IRGlobalParam* varDecl, IRGLSLShaderStorageBufferType* ssboType);
    void emitSSBOHeader(IRGlobalParam* varDecl, IRType* bufferType);

    void _emitGLSLPerVertexVaryingFragmentInput(IRGlobalParam* param, IRType* type);

    void _emitGLSLImageFormatModifier(IRInst* var, IRTextureType* resourceType);

    void _emitGLSLLayoutQualifiers(
        IRVarLayout* layout,
        EmitVarChain* inChain,
        LayoutResourceKind filter = LayoutResourceKind::None);

    /// If bindingKinds is set, it is used for binding index/set lookup. Passing in 0 is equivalent
    /// to using the kind only.
    bool _emitGLSLLayoutQualifierWithBindingKinds(
        LayoutResourceKind kind,
        EmitVarChain* chain,
        LayoutResourceKindFlags bindingKinds);
    bool _emitGLSLLayoutQualifier(LayoutResourceKind kind, EmitVarChain* chain)
    {
        return _emitGLSLLayoutQualifierWithBindingKinds(kind, chain, 0);
    }

    void _emitGLSLTypePrefix(IRType* type, bool promoteHalfToFloat = false);

    void _maybeEmitGLSLBuiltin(IRGlobalParam* var, UnownedStringSlice name);

    bool _maybeEmitInterpolationModifierText(IRInterpolationMode mode, Stage stage, bool isInput);

    void _requireGLSLExtension(const UnownedStringSlice& name);

    void _requireGLSLVersion(ProfileVersion version);
    void _requireGLSLVersion(int version);
    void _requireSPIRVVersion(const SemanticVersion& version);

    // Emit the `flat` qualifier if the underlying type
    // of the variable is an integer type.
    void _maybeEmitGLSLFlatModifier(IRType* valueType);

    void _requireBaseType(BaseType baseType);

    void _maybeEmitGLSLCast(IRType* castType, IRInst* inst);

    /// Emit the legalized form of a bitwise or logical operation on a vector of `bool`.
    ///
    /// This emits GLSL code that converts the operands of `inst` into vectors of
    /// `uint`, then applies `op` bitwise to the result, and finally converts back
    /// into the desired vector-of-bool `type`.
    ///
    void _emitLegalizedBoolVectorBinOp(
        IRInst* inst,
        IRVectorType* type,
        const EmitOpInfo& op,
        const EmitOpInfo& inOuterPrec);

    /// Try to emit specialized code for a logic binary op.
    ///
    /// Returns true if specialized code was emitted, false if the default behavior should be used.
    ///
    /// The `bitOp` parameter should be the bitwise equivalent of the logical op being emitted.
    ///
    bool _tryEmitLogicalBinOp(IRInst* inst, const EmitOpInfo& bitOp, const EmitOpInfo& inOuterPrec);

    /// Try to emit specialized code for a logic binary op.
    ///
    /// Returns true if specialized code was emitted, false if the default behavior should be used.
    ///
    /// The `bitOp` parameter should be the bitwise op being emitted.
    /// The `bitOp` parameter should be the logical equivalent of `bitOp`
    ///
    bool _tryEmitBitBinOp(
        IRInst* inst,
        const EmitOpInfo& bitOp,
        const EmitOpInfo& boolOp,
        const EmitOpInfo& inOuterPrec);

    void _requireRayTracing();

    void _requireRayQuery();

    void _requireFragmentShaderBarycentric();

    void _emitSpecialFloatImpl(IRType* type, const char* valueExpr);

    void emitAtomicImageCoord(IRImageSubscript* operand);

    void _beforeComputeEmitProcessInstruction(IRInst* parentFunc, IRInst* inst, IRBuilder& builder);

    Dictionary<IRInst*, HashSet<IRFunc*>> m_referencingEntryPoints;

    RefPtr<ShaderExtensionTracker> m_glslExtensionTracker;
};

} // namespace Slang
#endif
