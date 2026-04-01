#include "slang-ir-glsl-liveness.h"

#include "slang-ir-dominators.h"
#include "slang-ir-insts.h"
#include "slang-ir.h"

namespace Slang
{

namespace
{ // anonymous

struct GLSLLivenessContext
{
    enum class Kind
    {
        Start,
        End,
        CountOf,
    };

    /// Process the module
    void processModule();

    GLSLLivenessContext(IRModule* module)
        : m_module(module), m_builder(module)
    {
    }

    void _replaceMarker(IRLiveRangeMarker* liveMarker);
    void _addDecorations(Kind kind, IRFunc* func);

    /// Process a function in the module
    void _processFunction(IRFunc* funcInst);

    IRType* _getReferencedType(IRInst* referenced);

    Kind getKind(IROp op)
    {
        switch (op)
        {
        case kIROp_LiveRangeStart:
            return Kind::Start;
        case kIROp_LiveRangeEnd:
            return Kind::End;
        default:
            break;
        }
        SLANG_UNREACHABLE("Invalid op");
    }

    // Entry holds information about each of the function kinds
    struct Entry
    {
        Dictionary<IRType*, IRFunc*>
            m_funcs; ///< Map of the parameter type to functions implementing (for the kind)
        IRStringLit* m_nameHintLiteral = nullptr; ///< Name hint string literal of the function
        IRInst* m_spirvOpLiteral = nullptr;       ///< The SPIR-V opcode for the kind
    };

    List<IRLiveRangeMarker*> m_markerInsts; ///< All of the liveness marker instrucitons found

    Entry m_entries[Index(Kind::CountOf)]; /// Entry for each kind of function

    IRInst* m_zeroIntLiteral = nullptr;      ///< Zero value literal
    IRType* m_spirvIntLiteralType = nullptr; ///< Int type that emits as `spirv_literal`

    IRModule* m_module;
    IRBuilder m_builder;
};

void GLSLLivenessContext::_processFunction(IRFunc* funcInst)
{
    // Iterate through blocks in the function, looking for variables to live track
    for (auto block = funcInst->getFirstBlock(); block; block = block->getNextBlock())
    {
        for (auto inst = block->getFirstChild(); inst; inst = inst->getNextInst())
        {
            IRLiveRangeMarker* marker = as<IRLiveRangeMarker>(inst);
            if (marker)
            {
                m_markerInsts.add(marker);
            }
        }
    }
}

void GLSLLivenessContext::_addDecorations(Kind kind, IRFunc* func)
{
    // We might(?) want to add a decoration saying this is GLSL specific, but at this point
    // we can only be in GLSL dependent IR.
    //
    // m_builder.addTargetDecoration();

    // We don't need to explictly add the "GL_EXT_spirv_intrinsics"
    // as it will be added on the GLSL emit, with the SPIRVOpDecoration is hit
    const auto& entry = m_entries[Index(kind)];
    if (entry.m_nameHintLiteral)
    {
        m_builder.addNameHintDecoration(func, entry.m_nameHintLiteral);
    }

    m_builder.addDecoration(func, kIROp_SPIRVOpDecoration, entry.m_spirvOpLiteral);
}

IRType* GLSLLivenessContext::_getReferencedType(IRInst* referenced)
{
    auto type = referenced->getDataType();

    if (type->getOp() == kIROp_PtrType)
    {
        type = static_cast<IRPtrType*>(type)->getValueType();
    }

    return type;
}

void GLSLLivenessContext::_replaceMarker(IRLiveRangeMarker* markerInst)
{
    const auto kind = getKind(markerInst->getOp());
    auto& entry = m_entries[Index(kind)];

    IRInst* referenced = markerInst->getReferenced();
    IRType* referencedType = _getReferencedType(referenced);

    IRFunc* func = nullptr;

    if (IRFunc** funcPtr = entry.m_funcs.tryGetValue(referencedType))
    {
        func = *funcPtr;
    }
    else
    {
        // We didn't find a function for the type, so lets create one. It has a signature of
        //
        // void func(Ref<ReferencedType> target, int sizeInBytes)

        IRType* paramTypes[] = {
            m_builder.getRefType(
                referencedType,
                AddressSpace::Generic), ///< Use a reference to the referenced type
            m_spirvIntLiteralType,      ///< The size type
        };

        func = m_builder.createFunc();

        auto funcType =
            m_builder.getFuncType(SLANG_COUNT_OF(paramTypes), paramTypes, m_builder.getVoidType());
        m_builder.setDataType(func, funcType);

        // Add any decorations to the new function
        _addDecorations(kind, func);

        // Add to the map
        entry.m_funcs.add(referencedType, func);
    }
    SLANG_ASSERT(func);

    // Create a call to the function in the form of...
    // func(referencedItem, 0);

    // As per the SPIR-V documentation around the OpLifetimeStart/OpLifetimeEnd
    // https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#OpLifetimeStart
    //
    // If the type is known the size should be passed as 0
    IRInst* args[] = {
        referenced,
        m_zeroIntLiteral,
    };

    // Set the location at the marker to add the call
    m_builder.setInsertLoc(IRInsertLoc::after(markerInst));
    m_builder.emitCallInst(m_builder.getVoidType(), func, SLANG_COUNT_OF(args), args);

    // We don't need the marker anymore
    markerInst->removeAndDeallocate();
}

void GLSLLivenessContext::processModule()
{
    // Find all of the liveness marker insts
    //
    // This is done prior to processing, so we don't need to worry about traversal when
    // instructions are replaced.

    IRModuleInst* moduleInst = m_module->getModuleInst();
    for (IRInst* child : moduleInst->getChildren())
    {
        // We want to find all of the functions, and process them
        if (auto funcInst = as<IRFunc>(child))
        {
            // Then we want to look through their definition
            // inserting instructions that mark the liveness start/end
            _processFunction(funcInst);
        }
    }

    // If we didn't find any liveness marker instructions then we are done
    if (!m_markerInsts.getCount())
    {
        return;
    }

    // Int type that is SPIRV Literal (ie prefixed with spirv_literal)
    m_spirvIntLiteralType = m_builder.getSPIRVLiteralType(m_builder.getIntType());

    // Zero value literal
    m_zeroIntLiteral = m_builder.getIntValue(m_builder.getIntType(), 0);

    // We don't need to explicitly add this decoration because it will be added as needed on GLSL
    // emit m_extensionStringLiteral =
    // m_builder.getStringValue(UnownedStringSlice::fromLiteral("GL_EXT_spirv_intrinsics"));

    // Set up some values that will be needed on instructions

    // The op values are from the SPIR-V spec
    // https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#OpLifetimeStart

    {
        auto& entry = m_entries[Index(Kind::Start)];
        entry.m_nameHintLiteral =
            m_builder.getStringValue(UnownedStringSlice::fromLiteral("livenessStart"));
        entry.m_spirvOpLiteral = m_builder.getIntValue(m_builder.getIntType(), 256);
    }
    {
        auto& entry = m_entries[Index(Kind::End)];
        entry.m_nameHintLiteral =
            m_builder.getStringValue(UnownedStringSlice::fromLiteral("livenessEnd"));
        entry.m_spirvOpLiteral = m_builder.getIntValue(m_builder.getIntType(), 257);
    }

    // Iterate across instructions, replacing with a call to a generated function (one that just is
    // a declaration defining the SPIR-V op)
    for (auto markerInst : m_markerInsts)
    {
        _replaceMarker(markerInst);
    }
}

} // namespace

void applyGLSLLiveness(IRModule* module)
{
    GLSLLivenessContext context(module);

    context.processModule();
}

} // namespace Slang
