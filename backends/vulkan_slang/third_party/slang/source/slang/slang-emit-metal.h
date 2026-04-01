// slang-emit-metal.h
#ifndef SLANG_EMIT_METAL_H
#define SLANG_EMIT_METAL_H

#include "slang-emit-c-like.h"

namespace Slang
{
class MetalExtensionTracker : public ExtensionTracker
{
};

class MetalSourceEmitter : public CLikeSourceEmitter
{
public:
    typedef CLikeSourceEmitter Super;

    MetalSourceEmitter(const Desc& desc)
        : Super(desc), m_extensionTracker(new MetalExtensionTracker())
    {
    }

    virtual RefObject* getExtensionTracker() SLANG_OVERRIDE { return m_extensionTracker; }

protected:
    RefPtr<MetalExtensionTracker> m_extensionTracker;

    virtual bool isResourceTypeBindless(IRType* type) SLANG_OVERRIDE
    {
        SLANG_UNUSED(type);
        return true;
    }

    void emitMemoryOrderOperand(IRInst* inst);
    virtual void emitParameterGroupImpl(IRGlobalParam* varDecl, IRUniformParameterGroupType* type)
        SLANG_OVERRIDE;
    virtual void emitEntryPointAttributesImpl(
        IRFunc* irFunc,
        IREntryPointDecoration* entryPointDecor) SLANG_OVERRIDE;

    virtual void emitFrontMatterImpl(TargetRequest* targetReq) SLANG_OVERRIDE;

    virtual void emitRateQualifiersAndAddressSpaceImpl(IRRate* rate, AddressSpace addressSpace)
        SLANG_OVERRIDE;
    virtual bool tryEmitGlobalParamImpl(IRGlobalParam* varDecl, IRType* varType) SLANG_OVERRIDE;
    virtual void emitSemanticsImpl(IRInst* inst, bool allowOffsets) SLANG_OVERRIDE;
    virtual void emitSimpleFuncParamImpl(IRParam* param) SLANG_OVERRIDE;
    virtual void emitPostDeclarationAttributesForType(IRInst* type) SLANG_OVERRIDE;

    virtual void emitInterpolationModifiersImpl(
        IRInst* varInst,
        IRType* valueType,
        IRVarLayout* layout) SLANG_OVERRIDE;
    virtual void emitPackOffsetModifier(
        IRInst* varInst,
        IRType* valueType,
        IRPackOffsetDecoration* decoration) SLANG_OVERRIDE;

    virtual void emitMeshShaderModifiersImpl(IRInst* varInst) SLANG_OVERRIDE;
    virtual void emitSimpleTypeImpl(IRType* type) SLANG_OVERRIDE;
    virtual void emitParamTypeImpl(IRType* type, String const& name) SLANG_OVERRIDE;
    virtual void emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount)
        SLANG_OVERRIDE;
    virtual void emitVarDecorationsImpl(IRInst* varDecl) SLANG_OVERRIDE;
    virtual void emitMatrixLayoutModifiersImpl(IRType* varType) SLANG_OVERRIDE;

    virtual bool tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec) SLANG_OVERRIDE;
    virtual bool tryEmitInstStmtImpl(IRInst* inst) SLANG_OVERRIDE;

    virtual void emitSimpleValueImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual void emitLoopControlDecorationImpl(IRLoopControlDecoration* decl) SLANG_OVERRIDE;
    virtual void emitFuncDecorationImpl(IRDecoration* decoration) SLANG_OVERRIDE;
    virtual void emitFuncDecorationsImpl(IRFunc* func) SLANG_OVERRIDE;

    virtual void emitSwitchDecorationsImpl(IRSwitch* switchInst) SLANG_OVERRIDE;
    virtual void emitIfDecorationsImpl(IRIfElse* ifInst) SLANG_OVERRIDE;

    virtual void handleRequiredCapabilitiesImpl(IRInst* inst) SLANG_OVERRIDE;

    virtual void emitGlobalInstImpl(IRInst* inst) SLANG_OVERRIDE;
    virtual bool doesTargetSupportPtrTypes() SLANG_OVERRIDE { return true; }

    void emitFuncParamLayoutImpl(IRInst* param);

    virtual void _emitType(IRType* type, DeclaratorInfo* declarator) SLANG_OVERRIDE;

    void _emitHLSLParameterGroup(IRGlobalParam* varDecl, IRUniformParameterGroupType* type);

    void _emitHLSLTextureType(IRTextureTypeBase* texType);

    void _emitHLSLSubpassInputType(IRSubpassInputType* subpassType);

    void _emitHLSLDecorationSingleString(const char* name, IRFunc* entryPoint, IRStringLit* val);
    void _emitHLSLDecorationSingleInt(const char* name, IRFunc* entryPoint, IRIntLit* val);

    void _emitStageAccessSemantic(IRStageAccessDecoration* decoration, const char* name);
    bool _emitUserSemantic(UnownedStringSlice semanticName, IRIntegerValue semanticIndex);
    bool maybeEmitSystemSemantic(IRInst* inst);

    void emitAtomicImageCoord(IRImageSubscript* subscript);
    void emitAtomicDestOperand(IRInst* operand);
    void emitAtomicSrcOperand(bool isImage, IRInst* operand);
    void emitAtomicSemanticOperand(IRInst* inst);
};

} // namespace Slang
#endif
