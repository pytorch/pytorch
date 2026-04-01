#pragma once

#include "slang-emit-c-like.h"
#include "slang-extension-tracker.h"

namespace Slang
{

class WGSLSourceEmitter : public CLikeSourceEmitter
{
public:
    explicit WGSLSourceEmitter(const Desc& desc);

    virtual bool isResourceTypeBindless(IRType* type) SLANG_OVERRIDE
    {
        SLANG_UNUSED(type);
        return true;
    }
    virtual void emitParameterGroupImpl(IRGlobalParam* varDecl, IRUniformParameterGroupType* type)
        SLANG_OVERRIDE;
    virtual void emitEntryPointAttributesImpl(
        IRFunc* irFunc,
        IREntryPointDecoration* entryPointDecor) SLANG_OVERRIDE;
    virtual void emitSimpleTypeImpl(IRType* type) SLANG_OVERRIDE;
    virtual void emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount)
        SLANG_OVERRIDE;
    virtual void emitFuncHeaderImpl(IRFunc* func) SLANG_OVERRIDE;
    virtual void emitSimpleValueImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual bool tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec) SLANG_OVERRIDE;
    virtual bool tryEmitInstStmtImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual void emitSwitchCaseSelectorsImpl(const SwitchRegion::Case* currentCase, bool isDefault)
        SLANG_OVERRIDE;
    virtual void emitSimpleTypeAndDeclaratorImpl(IRType* type, DeclaratorInfo* declarator)
        SLANG_OVERRIDE;
    virtual void emitVarKeywordImpl(IRType* type, IRInst* varDecl) SLANG_OVERRIDE;
    virtual void emitDeclaratorImpl(DeclaratorInfo* declarator) SLANG_OVERRIDE;
    virtual void emitOperandImpl(IRInst* operand, EmitOpInfo const& outerPrec) SLANG_OVERRIDE;
    virtual void emitStructDeclarationSeparatorImpl() SLANG_OVERRIDE;
    virtual void emitLayoutQualifiersImpl(IRVarLayout* layout) SLANG_OVERRIDE;
    virtual void emitSimpleFuncParamImpl(IRParam* param) SLANG_OVERRIDE;
    virtual void emitParamTypeImpl(IRType* type, const String& name) SLANG_OVERRIDE;
    virtual void _emitType(IRType* type, DeclaratorInfo* declarator) SLANG_OVERRIDE;
    virtual void emitFrontMatterImpl(TargetRequest* targetReq) SLANG_OVERRIDE;
    virtual void emitSemanticsPrefixImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual void emitStructFieldAttributes(
        IRStructType* structType,
        IRStructField* field,
        bool allowOffsetLayout) SLANG_OVERRIDE;
    virtual void emitCallArg(IRInst* inst) SLANG_OVERRIDE;
    virtual void emitInterpolationModifiersImpl(
        IRInst* varInst,
        IRType* valueType,
        IRVarLayout* layout) SLANG_OVERRIDE;

    virtual void emitIntrinsicCallExprImpl(
        IRCall* inst,
        UnownedStringSlice intrinsicDefinition,
        IRInst* intrinsicInst,
        EmitOpInfo const& inOuterPrec) SLANG_OVERRIDE;
    virtual void emitGlobalParamDefaultVal(IRGlobalParam* varDecl) SLANG_OVERRIDE;

    virtual void emitRequireExtension(IRRequireTargetExtension* inst) SLANG_OVERRIDE;

    virtual void handleRequiredCapabilitiesImpl(IRInst* inst) SLANG_OVERRIDE;

    void emit(const AddressSpace addressSpace);

    virtual bool shouldFoldInstIntoUseSites(IRInst* inst) SLANG_OVERRIDE;

    virtual RefObject* getExtensionTracker() SLANG_OVERRIDE { return m_extensionTracker; }

private:
    bool maybeEmitSystemSemantic(IRInst* inst);

    // Emit the matrix type with 'rowCountWGSL' WGSL-rows and 'colCountWGSL' WGSL-columns
    void emitMatrixType(
        IRType* const elementType,
        const IRIntegerValue& rowCountWGSL,
        const IRIntegerValue& colCountWGSL);

    const char* getWgslImageFormat(IRTextureTypeBase* type);

    void _requireExtension(const UnownedStringSlice& name);

    bool m_f16ExtensionEnabled = false;

    RefPtr<ShaderExtensionTracker> m_extensionTracker;
};

} // namespace Slang
