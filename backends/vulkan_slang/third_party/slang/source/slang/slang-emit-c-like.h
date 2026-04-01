// slang-emit-c-like.h
#ifndef SLANG_EMIT_C_LIKE_H
#define SLANG_EMIT_C_LIKE_H

#include "../core/slang-basic.h"
#include "slang-compiler.h"
#include "slang-emit-base.h"
#include "slang-emit-precedence.h"
#include "slang-emit-source-writer.h"
#include "slang-ir-insts.h"
#include "slang-ir-restructure.h"
#include "slang-ir.h"

namespace Slang
{

class CLikeSourceEmitter : public SourceEmitterBase
{
public:
    enum class EmitLayoutSemanticOption
    {
        kPreType,
        kPostType
    };

    struct Desc
    {
        CodeGenContext* codeGenContext = nullptr;

        /// The stage for the entry point we are being asked to compile
        Stage entryPointStage = Stage::Unknown;

        /// The "effective" profile that is being used to emit code,
        /// combining information from the target and entry point.
        Profile effectiveProfile = Profile::RawEnum::Unknown;

        /// The source writer to use
        SourceWriter* sourceWriter = nullptr;
    };

    enum
    {
        kThreadGroupAxisCount = 3,
    };

    typedef unsigned int ESemanticMask;
    enum
    {
        kESemanticMask_None = 0,
        kESemanticMask_NoPackOffset = 1 << 0,
        kESemanticMask_Default = kESemanticMask_NoPackOffset,
    };

    /// A C-style declarator, used for emitting types and declarations.
    ///
    /// A C-style declaration typically has a *type specifier* (like
    /// `int` or `MyType`) and a *declarator* (like `myVar` or
    /// `myArray[]` or `*myPtr`).
    ///
    /// The type of a declaration depends on both the type specifier
    /// the declarator, and we already have logic to "unwrap" the
    /// syntax of a declarator as part of the parser.
    ///
    /// A `DeclaratorInfo` is used for the inverse process: taking
    /// a complete type and splitting out the parts that need to be
    /// handled as declarators when emitting code in a C-like language.
    ///
    struct DeclaratorInfo
    {
    public:
        enum class Flavor
        {
            Name,
            Ptr,
            Ref,
            SizedArray,
            UnsizedArray,
            LiteralSizedArray,
            Attributed,
        };
        Flavor flavor;

    protected:
        DeclaratorInfo(Flavor flavor)
            : flavor(flavor)
        {
        }
    };

    /// A simple declarator that only includes a name
    struct NameDeclaratorInfo : DeclaratorInfo
    {
        const StringSliceLoc* nameAndLoc;

        NameDeclaratorInfo(StringSliceLoc const* nameAndLoc)
            : DeclaratorInfo(Flavor::Name), nameAndLoc(nameAndLoc)
        {
        }
    };

    /// A "chained" declarator that may a nested declarator.
    struct ChainedDeclaratorInfo : DeclaratorInfo
    {
        DeclaratorInfo* next = nullptr;

    protected:
        ChainedDeclaratorInfo(Flavor flavor, DeclaratorInfo* next)
            : DeclaratorInfo(flavor), next(next)
        {
        }
    };

    struct PtrDeclaratorInfo : ChainedDeclaratorInfo
    {
        PtrDeclaratorInfo(DeclaratorInfo* next)
            : ChainedDeclaratorInfo(Flavor::Ptr, next)
        {
        }
    };

    struct RefDeclaratorInfo : ChainedDeclaratorInfo
    {
        RefDeclaratorInfo(DeclaratorInfo* next)
            : ChainedDeclaratorInfo(Flavor::Ref, next)
        {
        }
    };

    struct SizedArrayDeclaratorInfo : ChainedDeclaratorInfo
    {
        IRInst* elementCount;

        SizedArrayDeclaratorInfo(DeclaratorInfo* next, IRInst* elementCount)
            : ChainedDeclaratorInfo(Flavor::SizedArray, next), elementCount(elementCount)
        {
        }
    };

    struct UnsizedArrayDeclaratorInfo : ChainedDeclaratorInfo
    {
        UnsizedArrayDeclaratorInfo(DeclaratorInfo* next)
            : ChainedDeclaratorInfo(Flavor::UnsizedArray, next)
        {
        }
    };

    struct LiteralSizedArrayDeclaratorInfo : ChainedDeclaratorInfo
    {
        IRIntegerValue elementCount;

        LiteralSizedArrayDeclaratorInfo(DeclaratorInfo* next, IRIntegerValue elementCount)
            : ChainedDeclaratorInfo(Flavor::LiteralSizedArray, next), elementCount(elementCount)
        {
        }
    };

    struct AttributedDeclaratorInfo : ChainedDeclaratorInfo
    {
        AttributedDeclaratorInfo(DeclaratorInfo* next, IRInst* instWithAttributes)
            : ChainedDeclaratorInfo(Flavor::Attributed, next)
            , instWithAttributes(instWithAttributes)
        {
        }

        IRInst* instWithAttributes;
    };

    struct FuncTypeDeclaratorInfo : ChainedDeclaratorInfo
    {
        FuncTypeDeclaratorInfo(DeclaratorInfo* next, IRFuncType* funcTypeInst)
            : ChainedDeclaratorInfo(Flavor::Attributed, next), funcType(funcTypeInst)
        {
        }

        IRFuncType* funcType;
    };

    struct ComputeEmitActionsContext;

    // An action to be performed during code emit.
    struct EmitAction
    {
        enum Level
        {
            ForwardDeclaration,
            Definition,
        };
        Level level;
        IRInst* inst;
    };

    // A chain of variables to use for emitting semantic/layout info
    struct EmitVarChain
    {
        IRVarLayout* varLayout;
        EmitVarChain* next;

        EmitVarChain()
            : varLayout(nullptr), next(nullptr)
        {
        }

        EmitVarChain(IRVarLayout* varLayout)
            : varLayout(varLayout), next(nullptr)
        {
        }

        EmitVarChain(IRVarLayout* varLayout, EmitVarChain* next)
            : varLayout(varLayout), next(next)
        {
        }
    };


    /// Must be called before used
    virtual SlangResult init();

    /// Ctor
    CLikeSourceEmitter(const Desc& desc);


    /// Get the source manager
    SourceManager* getSourceManager() { return m_codeGenContext->getSourceManager(); }

    /// Get the source writer used
    SourceWriter* getSourceWriter() const { return m_writer; }

    /// Get the diagnostic sink
    DiagnosticSink* getSink() { return m_codeGenContext->getSink(); }

    /// Get the code gen target
    CodeGenTarget getTarget() { return m_target; }
    /// Get the source style
    SLANG_FORCE_INLINE SourceLanguage getSourceLanguage() const { return m_sourceLanguage; }

    void noteInternalErrorLoc(SourceLoc loc) { return getSink()->noteInternalErrorLoc(loc); }

    CapabilitySet getTargetCaps() { return m_codeGenContext->getTargetCaps(); }

    CodeGenContext* getCodeGenContext() { return m_codeGenContext; }
    TargetRequest* getTargetReq() { return m_codeGenContext->getTargetReq(); }
    Session* getSession() { return m_codeGenContext->getSession(); }
    Linkage* getLinkage() { return m_codeGenContext->getLinkage(); }
    ComponentType* getProgram() { return m_codeGenContext->getProgram(); }
    TargetProgram* getTargetProgram() { return m_codeGenContext->getTargetProgram(); }
    //
    // Types
    //

    void ensureTypePrelude(IRType* type);
    void emitDeclarator(DeclaratorInfo* declarator) { emitDeclaratorImpl(declarator); }
    virtual void emitDeclaratorImpl(DeclaratorInfo* declarator);

    void emitType(IRType* type, const StringSliceLoc* nameLoc) { emitTypeImpl(type, nameLoc); }
    void emitType(IRType* type, Name* name);
    void emitType(IRType* type, String const& name);
    void emitType(IRType* type);
    void emitType(IRType* type, Name* name, SourceLoc const& nameLoc);
    void emitType(IRType* type, NameLoc const& nameAndLoc);
    bool hasExplicitConstantBufferOffset(IRInst* cbufferType);
    bool isSingleElementConstantBuffer(IRInst* cbufferType);
    bool shouldForceUnpackConstantBufferElements(IRInst* cbufferType);
    //
    // Expressions
    //

    bool maybeEmitParens(EmitOpInfo& outerPrec, const EmitOpInfo& prec);

    void maybeCloseParens(bool needClose);

    void emitStringLiteral(const String& value);

    void emitVal(IRInst* val, const EmitOpInfo& outerPrec);

    void emitStore(IRStore* store);
    virtual void _emitStoreImpl(IRStore* store);
    void _emitInstAsDefaultInitializedVar(IRInst* inst, IRType* type);
    void _emitInstAsVarInitializerImpl(IRInst* inst);

    UInt getBindingOffset(EmitVarChain* chain, LayoutResourceKind kind);
    UInt getBindingSpace(EmitVarChain* chain, LayoutResourceKind kind);

    /// Finds the binding offset for *all* the kinds that match the kindFlags
    /// Thus only meaningful if multiple kinds can be treated as the same as far as binding is
    /// concerned. In particular is useful for GLSL binding emit, where some HLSL resource kinds can
    /// appear but are in effect the same as DescriptorSlot
    UInt getBindingOffsetForKinds(EmitVarChain* chain, LayoutResourceKindFlags kindFlags);
    UInt getBindingSpaceForKinds(EmitVarChain* chain, LayoutResourceKindFlags kindFlags);

    // Utility code for generating unique IDs as needed
    // during the emit process (e.g., for declarations
    // that didn't originally have names, but now need to).
    UInt allocateUniqueID();

    // IR-level emit logic

    UInt getID(IRInst* value);

    /// "Scrub" a name so that it complies with restrictions of the target language.
    void appendScrubbedName(const UnownedStringSlice& name, StringBuilder& out);

    String generateName(IRInst* inst);
    virtual String generateEntryPointNameImpl(IREntryPointDecoration* entryPointDecor);

    String getName(IRInst* inst);
    String getUnmangledName(IRInst* inst);

    void emitSimpleValue(IRInst* inst) { emitSimpleValueImpl(inst); }

    virtual bool shouldFoldInstIntoUseSites(IRInst* inst);

    void emitOperand(IRInst* inst, EmitOpInfo const& outerPrec)
    {
        emitOperandImpl(inst, outerPrec);
    }

    void emitArgs(IRInst* inst);

    void emitRateQualifiers(IRInst* value);
    void emitRateQualifiersAndAddressSpace(IRInst* value);

    void emitInstResultDecl(IRInst* inst);

    template<typename T>
    IRTargetSpecificDecoration* findBestTargetDecoration(IRInst* inst);
    IRTargetIntrinsicDecoration* _findBestTargetIntrinsicDecoration(IRInst* inst);

    // Find the definition of a target intrinsic either from __target_intrinsic decoration, or from
    // a genericAsm inst in the function body. `outInst` is the decoration or the genericAsm inst.
    bool findTargetIntrinsicDefinition(
        IRInst* callee,
        UnownedStringSlice& outDefinition,
        IRInst*& outInst);

    // Check if the string being used to define a target intrinsic
    // is an "ordinary" name, such that we can simply emit a call
    // to the new name with the arguments of the old operation.
    static bool isOrdinaryName(const UnownedStringSlice& name);

    void emitComInterfaceCallExpr(IRCall* inst, EmitOpInfo const& inOuterPrec);

    void emitIntrinsicCallExpr(
        IRCall* inst,
        UnownedStringSlice intrinsicDefinition,
        IRInst* intrinsicInst,
        EmitOpInfo const& inOuterPrec);

    void emitCallExpr(IRCall* inst, EmitOpInfo outerPrec);

    void emitLiveness(IRInst* inst) { emitLivenessImpl(inst); }

    void emitInstExpr(IRInst* inst, EmitOpInfo const& inOuterPrec);
    void defaultEmitInstExpr(IRInst* inst, EmitOpInfo const& inOuterPrec);
    void diagnoseUnhandledInst(IRInst* inst);
    void emitInst(IRInst* inst);

    void emitSemanticsPrefix(IRInst* inst);
    void emitSemantics(IRInst* inst, bool allowOffsets = false);
    void emitSemanticsUsingVarLayout(IRVarLayout* varLayout);

    void emitDecorationLayoutSemantics(IRInst* inst, char const* uniformSemanticSpelling);
    void emitLayoutSemantics(IRInst* inst, char const* uniformSemanticSpelling);

    /// Emit high-level language statements from a structured region.
    void emitRegion(Region* inRegion);

    /// Emit high-level language statements from a structured region tree.
    void emitRegionTree(RegionTree* regionTree);

    // Is an IR function a definition? (otherwise it is a declaration)
    bool isDefinition(IRFunc* func);

    void emitEntryPointAttributes(IRFunc* irFunc, IREntryPointDecoration* entryPointDecor);

    /// Emit high-level statements for the body of a function.
    void emitFunctionBody(IRGlobalValueWithCode* code);

    void emitFuncHeader(IRFunc* func) { emitFuncHeaderImpl(func); }
    void emitSimpleFunc(IRFunc* func) { emitSimpleFuncImpl(func); }

    void emitSwitchCaseSelectors(const SwitchRegion::Case* currentCase, bool isDefault)
    {
        emitSwitchCaseSelectorsImpl(currentCase, isDefault);
    }

    void emitParamType(IRType* type, String const& name) { emitParamTypeImpl(type, name); }

    void emitFuncDecl(IRFunc* func);
    void emitFuncDecl(IRFunc* func, const String& name);


    IREntryPointLayout* getEntryPointLayout(IRFunc* func);

    IREntryPointLayout* asEntryPoint(IRFunc* func);

    // Detect if the given IR function/type represents a
    // declaration of an intrinsic/builtin for the
    // current code-generation target.
    bool isTargetIntrinsic(IRInst* func);

    void emitFunc(IRFunc* func);
    void emitFuncDecorations(IRFunc* func) { emitFuncDecorationsImpl(func); }

    void emitStruct(IRStructType* structType);
    // This is used independently of `emitStruct` by some GLSL parameter group
    // output functionality
    void emitStructDeclarationsBlock(IRStructType* structType, bool allowOffsetLayout);
    void emitClass(IRClassType* structType);

    void emitStructDeclarationSeparator() { emitStructDeclarationSeparatorImpl(); }
    virtual void emitStructDeclarationSeparatorImpl();

    /// Emit type attributes that should appear after, e.g., a `struct` keyword
    void emitPostKeywordTypeAttributes(IRInst* inst) { emitPostKeywordTypeAttributesImpl(inst); }

    virtual void emitMemoryQualifiers(IRInst* /*varInst*/){};
    virtual void emitStructFieldAttributes(
        IRStructType* /* structType */,
        IRStructField* /* field */,
        bool /* allowOffsetLayout */){};
    void emitInterpolationModifiers(IRInst* varInst, IRType* valueType, IRVarLayout* layout);
    void emitMeshShaderModifiers(IRInst* varInst);
    virtual void emitPackOffsetModifier(
        IRInst* /*varInst*/,
        IRType* /*valueType*/,
        IRPackOffsetDecoration* /*decoration*/
    ){};


    /// Emit modifiers that should apply even for a declaration of an SSA temporary.
    virtual void emitTempModifiers(IRInst* temp);

    void emitVarModifiers(IRVarLayout* layout, IRInst* varDecl, IRType* varType);

    /// Emit the array brackets that go on the end of a declaration of the given type.
    void emitArrayBrackets(IRType* inType);

    void emitParameterGroup(IRGlobalParam* varDecl, IRUniformParameterGroupType* type);

    void emitVar(IRVar* varDecl);
    void emitDereferenceOperand(IRInst* inst, EmitOpInfo const& outerPrec);

    void emitGlobalVar(IRGlobalVar* varDecl);
    void emitGlobalParam(IRGlobalParam* varDecl);

    void emitGlobalInst(IRInst* inst);
    virtual void emitGlobalInstImpl(IRInst* inst);

    void ensureInstOperand(
        ComputeEmitActionsContext* ctx,
        IRInst* inst,
        EmitAction::Level requiredLevel = EmitAction::Level::Definition);

    void ensureInstOperandsRec(ComputeEmitActionsContext* ctx, IRInst* inst);

    void ensureGlobalInst(
        ComputeEmitActionsContext* ctx,
        IRInst* inst,
        EmitAction::Level requiredLevel);

    void emitForwardDeclaration(IRInst* inst);

    void computeEmitActions(IRModule* module, List<EmitAction>& ioActions);

    void executeEmitActions(List<EmitAction> const& actions);

    // Emits front matter, that occurs before the prelude
    // Doesn't emit generated function/types that's handled by emitPreModule
    void emitFrontMatter(TargetRequest* targetReq) { emitFrontMatterImpl(targetReq); }

    void emitPreModule() { emitPreModuleImpl(); }
    void emitModule(IRModule* module, DiagnosticSink* sink)
    {
        m_irModule = module;
        emitModuleImpl(module, sink);
    }

    void emitSimpleType(IRType* type);

    void emitVectorTypeName(IRType* elementType, IRIntegerValue elementCount)
    {
        emitVectorTypeNameImpl(elementType, elementCount);
    }

    void emitTextureOrTextureSamplerType(IRTextureTypeBase* type, char const* baseName)
    {
        emitTextureOrTextureSamplerTypeImpl(type, baseName);
    }

    void emitSubpassInputType(IRSubpassInputType* type) { emitSubpassInputTypeImpl(type); }

    virtual RefObject* getExtensionTracker() { return nullptr; }

    /// Gets a source language for a target for a target. Returns Unknown if not a known target
    static SourceLanguage getSourceLanguage(CodeGenTarget target);

    /// Gets the default type name for built in scalar types. Different impls may require something
    /// different. Returns an empty slice if not a built in type
    static UnownedStringSlice getDefaultBuiltinTypeName(IROp op);

    /// Finds the IRNumThreadsDecoration and gets the size from that or sets all
    /// dimensions to 1
    IRNumThreadsDecoration* getComputeThreadGroupSize(
        IRFunc* func,
        Int outNumThreads[kThreadGroupAxisCount]);

    /// Finds the IRNumThreadsDecoration and gets the size from that or sets all
    /// dimensions to 1. If specialization constants are used for an axis, their
    /// IDs is reported in non-negative entries of outSpecializationConstantIds.
    static IRNumThreadsDecoration* getComputeThreadGroupSize(
        IRFunc* func,
        Int outNumThreads[kThreadGroupAxisCount],
        Int outSpecializationConstantIds[kThreadGroupAxisCount]);

    /// Finds the IRWaveSizeDecoration and gets the size from that.
    static IRWaveSizeDecoration* getComputeWaveSize(IRFunc* func, Int* outWaveSize);

protected:
    virtual void emitGlobalParamDefaultVal(IRGlobalParam* inst) { SLANG_UNUSED(inst); }
    virtual void emitPostDeclarationAttributesForType(IRInst* type) { SLANG_UNUSED(type); }
    virtual String getTargetBuiltinVarName(IRInst* inst, IRTargetBuiltinVarName builtinName);
    virtual bool doesTargetSupportPtrTypes() { return false; }
    virtual bool isResourceTypeBindless(IRType* type)
    {
        SLANG_UNUSED(type);
        return false;
    }
    virtual void emitLayoutSemanticsImpl(
        IRInst* inst,
        char const* uniformSemanticSpelling,
        EmitLayoutSemanticOption layoutSemanticOption)
    {
        SLANG_UNUSED(inst);
        SLANG_UNUSED(uniformSemanticSpelling);
        SLANG_UNUSED(layoutSemanticOption);
    }
    virtual void emitParameterGroupImpl(
        IRGlobalParam* varDecl,
        IRUniformParameterGroupType* type) = 0;
    virtual void emitEntryPointAttributesImpl(
        IRFunc* irFunc,
        IREntryPointDecoration* entryPointDecor) = 0;

    virtual void emitImageFormatModifierImpl(IRInst* varDecl, IRType* varType)
    {
        SLANG_UNUSED(varDecl);
        SLANG_UNUSED(varType);
    }
    virtual void emitLayoutQualifiersImpl(IRVarLayout* layout) { SLANG_UNUSED(layout); }

    /// Emit front matter inserting prelude where appropriate
    virtual void emitFrontMatterImpl(TargetRequest* targetReq);
    /// Emit any declarations, and other material that is needed before the modules contents
    /// For example on targets that don't have built in vector/matrix support, this is where
    /// the appropriate generated declarations occur.
    virtual void emitPreModuleImpl();

    virtual void emitSimpleTypeAndDeclaratorImpl(IRType* type, DeclaratorInfo* declarator);
    void emitSimpleTypeAndDeclarator(IRType* type, DeclaratorInfo* declarator)
    {
        emitSimpleTypeAndDeclaratorImpl(type, declarator);
    };
    virtual void emitVarKeywordImpl(IRType* type, IRInst* varDecl);
    void emitVarKeyword(IRType* type, IRInst* varDecl) { emitVarKeywordImpl(type, varDecl); }

    virtual void beforeComputeEmitActions(IRModule* module) { SLANG_UNUSED(module); };

    virtual void emitRateQualifiersAndAddressSpaceImpl(IRRate* rate, AddressSpace addressSpace)
    {
        SLANG_UNUSED(rate);
        SLANG_UNUSED(addressSpace);
    }
    virtual void emitSemanticsPrefixImpl(IRInst* inst) { SLANG_UNUSED(inst); }
    virtual void emitSemanticsImpl(IRInst* inst, bool allowOffsetLayout)
    {
        SLANG_UNUSED(inst);
        SLANG_UNUSED(allowOffsetLayout);
    }
    virtual void emitSimpleFuncParamImpl(IRParam* param);
    virtual void emitSimpleFuncParamsImpl(IRFunc* func);
    virtual void emitInterpolationModifiersImpl(
        IRInst* varInst,
        IRType* valueType,
        IRVarLayout* layout)
    {
        SLANG_UNUSED(varInst);
        SLANG_UNUSED(valueType);
        SLANG_UNUSED(layout);
    }

    virtual void emitMeshShaderModifiersImpl(IRInst* varInst) { SLANG_UNUSED(varInst) }
    virtual void emitSimpleTypeImpl(IRType* type) = 0;
    virtual void emitVarDecorationsImpl(IRInst* varDecl) { SLANG_UNUSED(varDecl); }
    virtual void emitMatrixLayoutModifiersImpl(IRType* varType) { SLANG_UNUSED(varType); }
    virtual void emitTypeImpl(IRType* type, const StringSliceLoc* nameLoc);
    virtual void emitSimpleValueImpl(IRInst* inst);
    virtual void emitModuleImpl(IRModule* module, DiagnosticSink* sink);
    virtual void emitFuncHeaderImpl(IRFunc* func);
    virtual void emitSimpleFuncImpl(IRFunc* func);
    virtual void emitVarExpr(IRInst* inst, EmitOpInfo const& outerPrec);
    virtual void emitOperandImpl(IRInst* inst, EmitOpInfo const& outerPrec);
    virtual void emitParamTypeImpl(IRType* type, String const& name);
    virtual void emitParamTypeModifier(IRType* type) { SLANG_UNUSED(type); }
    virtual void emitIntrinsicCallExprImpl(
        IRCall* inst,
        UnownedStringSlice intrinsicDefinition,
        IRInst* intrinsicInst,
        EmitOpInfo const& inOuterPrec);
    virtual void emitFunctionPreambleImpl(IRInst* inst) { SLANG_UNUSED(inst); }
    virtual void emitLoopControlDecorationImpl(IRLoopControlDecoration* decl)
    {
        SLANG_UNUSED(decl);
    }
    virtual void emitIfDecorationsImpl(IRIfElse* ifInst) { SLANG_UNUSED(ifInst); }
    virtual void emitSwitchDecorationsImpl(IRSwitch* switchInst) { SLANG_UNUSED(switchInst); }
    virtual void emitSwitchCaseSelectorsImpl(const SwitchRegion::Case* currentCase, bool isDefault);

    virtual void emitFuncDecorationImpl(IRDecoration* decoration) { SLANG_UNUSED(decoration); }
    virtual void emitLivenessImpl(IRInst* inst);

    virtual void emitFuncDecorationsImpl(IRFunc* func);

    // Only needed for glsl output with $ prefix intrinsics - so perhaps removable in the future
    virtual void emitTextureOrTextureSamplerTypeImpl(IRTextureTypeBase* type, char const* baseName)
    {
        SLANG_UNUSED(type);
        SLANG_UNUSED(baseName);
    }

    bool tryGetIntInfo(IRType* elementType, bool& isSigned, int& bitWidth);
    void emitVecNOrScalar(IRVectorType* vectorType, std::function<void()> func);
    virtual void emitBitfieldExtractImpl(IRInst* inst);
    virtual void emitBitfieldInsertImpl(IRInst* inst);

    virtual void emitSubpassInputTypeImpl(IRSubpassInputType* type) { SLANG_UNUSED(type); }

    // Again necessary for & prefix intrinsics. May be removable in the future
    virtual void emitVectorTypeNameImpl(IRType* elementType, IRIntegerValue elementCount) = 0;

    virtual void emitWitnessTable(IRWitnessTable* witnessTable);
    void emitComWitnessTable(IRWitnessTable* witnessTable);

    virtual void emitInterface(IRInterfaceType* interfaceType);
    virtual void emitRTTIObject(IRRTTIObject* rttiObject);

    virtual bool tryEmitGlobalParamImpl(IRGlobalParam* varDecl, IRType* varType)
    {
        SLANG_UNUSED(varDecl);
        SLANG_UNUSED(varType);
        return false;
    }
    virtual bool tryEmitInstExprImpl(IRInst* inst, const EmitOpInfo& inOuterPrec)
    {
        SLANG_UNUSED(inst);
        SLANG_UNUSED(inOuterPrec);
        return false;
    }
    virtual bool tryEmitInstStmtImpl(IRInst* inst)
    {
        SLANG_UNUSED(inst);
        return false;
    }

    void defaultEmitInstStmt(IRInst* inst);
    void emitInstStmt(IRInst* inst);

    virtual void emitPostKeywordTypeAttributesImpl(IRInst* inst) { SLANG_UNUSED(inst); }

    void _emitFuncTypeDeclaration(IRFuncType* type, IRAttributedType* attributes);

    virtual void _emitType(IRType* type, DeclaratorInfo* declarator);
    void _emitInst(IRInst* inst);

    virtual void _emitPrefixTypeAttr(IRAttr* attr);
    virtual void _emitPostfixTypeAttr(IRAttr* attr);

    // Emit the argument list (including paranthesis) in a `CallInst`
    void _emitCallArgList(IRCall* call, int startingOperandIndex = 1);
    virtual void emitCallArg(IRInst* arg);

    virtual void emitRequireExtension(IRRequireTargetExtension* inst) { SLANG_UNUSED(inst); }

    String _generateUniqueName(const UnownedStringSlice& slice);

    // Sort witnessTable entries according to the order defined in the witnessed interface type.
    List<IRWitnessTableEntry*> getSortedWitnessTableEntries(IRWitnessTable* witnessTable);

    // Special handling for swizzleStore call, save the right-handside vector to a temporary
    // variable first, then assign the corresponding elements to the left-handside vector one by
    // one.
    void _emitSwizzleStorePerElement(IRInst* inst);

    String _emitLiteralOneWithType(int bitWidth);


    virtual void ensurePrelude(const char* preludeText);

    CodeGenContext* m_codeGenContext = nullptr;
    IRModule* m_irModule = nullptr;

    // The stage for which we are emitting code.
    //
    // TODO: We should support emitting code that includes multiple
    // entry points for different stages, but this value is used
    // in some very specific cases to determine how a construct
    // should map to GLSL.
    //
    Stage m_entryPointStage = Stage::Unknown;

    // The target language we want to generate code for
    CodeGenTarget m_target;

    // Source language (based on the more nuanced m_target)
    SourceLanguage m_sourceLanguage;

    // Where source is written to
    SourceWriter* m_writer;

    UInt m_uniqueIDCounter = 1;
    Dictionary<IRInst*, UInt> m_mapIRValueToID;

    HashSet<String> m_irDeclsVisited;

    HashSet<String> m_irTupleTypes;

    // The "effective" profile that is being used to emit code,
    // combining information from the target and entry point.
    Profile m_effectiveProfile;

    // Map a string name to the number of times we have seen this
    // name used so far during code emission.
    Dictionary<String, UInt> m_uniqueNameCounters;

    // Map an IR instruction to the name that we've decided
    // to use for it when emitting code.
    Dictionary<IRInst*, String> m_mapInstToName;

    OrderedHashSet<IRStringLit*> m_requiredPreludes;

    Dictionary<const char*, IRStringLit*> m_builtinPreludes;
};

} // namespace Slang
#endif
