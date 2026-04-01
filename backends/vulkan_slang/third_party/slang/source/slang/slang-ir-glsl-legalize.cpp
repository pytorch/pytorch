// slang-ir-glsl-legalize.cpp
#include "slang-ir-glsl-legalize.h"

#include "slang-extension-tracker.h"
#include "slang-ir-clone.h"
#include "slang-ir-inst-pass-base.h"
#include "slang-ir-insts.h"
#include "slang-ir-single-return.h"
#include "slang-ir-specialize-function-call.h"
#include "slang-ir-util.h"
#include "slang-ir.h"

#include <functional>

#ifdef SLANG_USE_SYSTEM_SPIRV_HEADER
#include <spirv/unified1/spirv.h>
#else
#include "../../external/spirv-headers/include/spirv/unified1/spirv.h"
#endif

namespace Slang
{
//
// Legalization of entry points for GLSL:
//

IRGlobalParam* addGlobalParam(IRModule* module, IRType* valueType)
{
    IRBuilder builder(module);

    return builder.createGlobalParam(valueType);
}

void moveValueBefore(IRInst* valueToMove, IRInst* placeBefore)
{
    valueToMove->removeFromParent();
    valueToMove->insertBefore(placeBefore);
}

IRType* getFieldType(IRType* baseType, IRStructKey* fieldKey)
{
    if (auto structType = as<IRStructType>(baseType))
    {
        for (auto ff : structType->getFields())
        {
            if (ff->getKey() == fieldKey)
                return ff->getFieldType();
        }
        SLANG_UNEXPECTED("no such field");
        UNREACHABLE_RETURN(nullptr);
    }
    SLANG_UNEXPECTED("not a struct");
    UNREACHABLE_RETURN(nullptr);
}


// When scalarizing shader inputs/outputs for GLSL, we need a way
// to refer to a conceptual "value" that might comprise multiple
// IR-level values. We could in principle introduce tuple types
// into the IR so that everything stays at the IR level, but
// it seems easier to just layer it over the top for now.
//
// The `ScalarizedVal` type deals with the "tuple or single value?"
// question, and also the "l-value or r-value?" question.
struct ScalarizedValImpl : RefObject
{
};
struct ScalarizedTupleValImpl;
struct ScalarizedTypeAdapterValImpl;
struct ScalarizedArrayIndexValImpl;

struct ScalarizedVal
{
    enum class Flavor
    {
        // no value (null pointer)
        none,

        // A simple `IRInst*` that represents the actual value
        value,

        // An `IRInst*` that represents the address of the actual value
        address,

        // A `TupleValImpl` that represents zero or more `ScalarizedVal`s
        tuple,

        // A `TypeAdapterValImpl` that wraps a single `ScalarizedVal` and
        // represents an implicit type conversion applied to it on read
        // or write.
        typeAdapter,

        // Array index to the irValue. The actual index is stored in impl as
        // ScalarizedArrayIndexValImpl
        arrayIndex,
    };

    // Create a value representing a simple value
    static ScalarizedVal value(IRInst* irValue)
    {
        ScalarizedVal result;
        result.flavor = Flavor::value;
        result.irValue = irValue;
        return result;
    }

    // Create a value representing an address
    static ScalarizedVal address(IRInst* irValue)
    {
        ScalarizedVal result;
        result.flavor = Flavor::address;
        result.irValue = irValue;
        return result;
    }

    static ScalarizedVal tuple(ScalarizedTupleValImpl* impl)
    {
        ScalarizedVal result;
        result.flavor = Flavor::tuple;
        result.impl = (ScalarizedValImpl*)impl;
        return result;
    }

    static ScalarizedVal typeAdapter(ScalarizedTypeAdapterValImpl* impl)
    {
        ScalarizedVal result;
        result.flavor = Flavor::typeAdapter;
        result.impl = (ScalarizedValImpl*)impl;
        return result;
    }
    static ScalarizedVal scalarizedArrayIndex(ScalarizedArrayIndexValImpl* impl)
    {
        ScalarizedVal result;
        result.flavor = Flavor::arrayIndex;
        result.irValue = nullptr;
        result.impl = (ScalarizedValImpl*)impl;
        return result;
    }

    List<IRInst*> leafAddresses();

    Flavor flavor = Flavor::none;
    IRInst* irValue = nullptr;
    RefPtr<ScalarizedValImpl> impl;
};

// This is the case for a value that is a "tuple" of other values
struct ScalarizedTupleValImpl : ScalarizedValImpl
{
    struct Element
    {
        IRStructKey* key;
        ScalarizedVal val;
    };

    IRType* type;
    List<Element> elements;
};

// This is the case for a value that is stored with one type,
// but needs to present itself as having a different type
struct ScalarizedTypeAdapterValImpl : ScalarizedValImpl
{
    ScalarizedVal val;
    IRType* actualType;  // the actual type of `val`
    IRType* pretendType; // the type this value pretends to have
};

struct ScalarizedArrayIndexValImpl : ScalarizedValImpl
{
    ScalarizedVal arrayVal;
    Index index;
    IRType* elementType;
};

ScalarizedVal extractField(
    IRBuilder* builder,
    ScalarizedVal const& val,
    UInt fieldIndex, // Pass ~0 in to search for the index via the key
    IRStructKey* fieldKey);
ScalarizedVal adaptType(IRBuilder* builder, IRInst* val, IRType* toType, IRType* fromType);
ScalarizedVal adaptType(
    IRBuilder* builder,
    ScalarizedVal const& val,
    IRType* toType,
    IRType* fromType);
IRInst* materializeValue(IRBuilder* builder, ScalarizedVal const& val);
ScalarizedVal getSubscriptVal(
    IRBuilder* builder,
    IRType* elementType,
    ScalarizedVal val,
    IRInst* indexVal);
ScalarizedVal getSubscriptVal(
    IRBuilder* builder,
    IRType* elementType,
    ScalarizedVal val,
    UInt index);

struct GlobalVaryingDeclarator
{
    enum class Flavor
    {
        array,
        meshOutputVertices,
        meshOutputIndices,
        meshOutputPrimitives,
    };

    Flavor flavor;
    IRInst* elementCount;
    GlobalVaryingDeclarator* next;
};

enum GLSLSystemValueKind
{
    General,
    PositionOutput,
    PositionInput,
};

struct GLSLSystemValueInfo
{
    // The name of the built-in GLSL variable
    char const* name;

    // The name of an outer array that wraps
    // the variable, in the case of a GS input
    char const* outerArrayName;

    // The required type of the built-in variable
    IRType* requiredType;

    // If the built in GLSL variable is an array, holds the index into the array.
    // If < 0, then there is no array indexing
    Index arrayIndex;

    // The kind of the system value that requires special treatment.
    GLSLSystemValueKind kind = GLSLSystemValueKind::General;

    // The target builtin name.
    IRTargetBuiltinVarName targetVarName = IRTargetBuiltinVarName::Unknown;
};

static void leafAddressesImpl(List<IRInst*>& ret, const ScalarizedVal& v)
{
    switch (v.flavor)
    {
    case ScalarizedVal::Flavor::none:
    case ScalarizedVal::Flavor::value:
        break;

    case ScalarizedVal::Flavor::address:
        {
            ret.add(v.irValue);
        }
        break;
    case ScalarizedVal::Flavor::tuple:
        {
            auto tupleVal = as<ScalarizedTupleValImpl>(v.impl);
            for (auto e : tupleVal->elements)
            {
                leafAddressesImpl(ret, e.val);
            }
        }
        break;
    case ScalarizedVal::Flavor::typeAdapter:
        {
            auto typeAdapterVal = as<ScalarizedTypeAdapterValImpl>(v.impl);
            leafAddressesImpl(ret, typeAdapterVal->val);
        }
        break;
    }
}

List<IRInst*> ScalarizedVal::leafAddresses()
{
    List<IRInst*> ret;
    leafAddressesImpl(ret, *this);
    return ret;
}

struct GLSLLegalizationContext
{
    Session* session;
    ShaderExtensionTracker* glslExtensionTracker;
    DiagnosticSink* sink;
    Stage stage;
    IRFunc* entryPointFunc;
    Dictionary<IRTargetBuiltinVarName, IRInst*> builtinVarMap;

    /// This dictionary stores all bindings of 'VaryingIn/VaryingOut'. We assume 'space' is 0.
    Dictionary<LayoutResourceKind, UIntSet> usedBindingIndex;

    GLSLLegalizationContext()
    {
        // Reserve for VaryingInput VaryingOutput
        usedBindingIndex.reserve(2);
    }

    struct SystemSemanticGlobal
    {
        void addIndex(Index index) { maxIndex = (index > maxIndex) ? index : maxIndex; }

        IRGlobalParam* globalParam;
        Count maxIndex;
    };

    // Currently only used for special cases of semantics which map to global variables
    Dictionary<UnownedStringSlice, SystemSemanticGlobal> systemNameToGlobalMap;

    // Map from a input parameter in fragment shader to its corresponding per-vertex array
    // to support the `GetAttributeAtVertex` intrinsic.
    Dictionary<IRInst*, IRInst*> mapVertexInputToPerVertexArray;

    void requireGLSLExtension(const UnownedStringSlice& name)
    {
        glslExtensionTracker->requireExtension(name);
    }

    void requireSPIRVVersion(const SemanticVersion& version)
    {
        glslExtensionTracker->requireSPIRVVersion(version);
    }

    void requireGLSLVersion(ProfileVersion version)
    {
        glslExtensionTracker->requireVersion(version);
    }

    Stage getStage() { return stage; }

    DiagnosticSink* getSink() { return sink; }

    IRBuilder* builder;
    IRBuilder* getBuilder() { return builder; }
};

// This examines the passed type and determines the GLSL mesh shader indices
// builtin name and type
GLSLSystemValueInfo* getMeshOutputIndicesSystemValueInfo(
    GLSLLegalizationContext* context,
    LayoutResourceKind kind,
    Stage stage,
    IRType* type,
    GlobalVaryingDeclarator* declarator,
    GLSLSystemValueInfo* inStorage)
{
    IRBuilder* builder = context->builder;
    if (stage != Stage::Mesh)
    {
        return nullptr;
    }
    if (kind != LayoutResourceKind::VaryingOutput)
    {
        return nullptr;
    }
    if (!declarator || declarator->flavor != GlobalVaryingDeclarator::Flavor::meshOutputIndices)
    {
        return nullptr;
    }

    inStorage->arrayIndex = -1;
    inStorage->outerArrayName = nullptr;

    // Points
    if (isIntegralType(type))
    {
        inStorage->name = "gl_PrimitivePointIndicesEXT";
        inStorage->requiredType = builder->getUIntType();
        return inStorage;
    }

    auto vectorCount = composeGetters<IRIntLit>(type, &IRVectorType::getElementCount);
    auto elemType = composeGetters<IRType>(type, &IRVectorType::getElementType);

    // Lines
    if (vectorCount->getValue() == 2 && isIntegralType(elemType))
    {
        inStorage->name = "gl_PrimitiveLineIndicesEXT";
        inStorage->requiredType = builder->getVectorType(
            builder->getUIntType(),
            builder->getIntValue(builder->getIntType(), 2));
        return inStorage;
    }

    // Triangles
    if (vectorCount->getValue() == 3 && isIntegralType(elemType))
    {
        inStorage->name = "gl_PrimitiveTriangleIndicesEXT";
        inStorage->requiredType = builder->getVectorType(
            builder->getUIntType(),
            builder->getIntValue(builder->getIntType(), 3));
        return inStorage;
    }

    SLANG_UNREACHABLE("Unhandled mesh output indices type");
}

GLSLSystemValueInfo* getGLSLSystemValueInfo(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRVarLayout* varLayout,
    LayoutResourceKind kind,
    Stage stage,
    IRType* type,
    GlobalVaryingDeclarator* declarator,
    GLSLSystemValueInfo* inStorage)
{
    SLANG_UNUSED(codeGenContext);

    if (auto indicesSemantic =
            getMeshOutputIndicesSystemValueInfo(context, kind, stage, type, declarator, inStorage))
    {
        return indicesSemantic;
    }

    char const* name = nullptr;
    char const* outerArrayName = nullptr;
    int arrayIndex = -1;
    GLSLSystemValueKind systemValueKind = GLSLSystemValueKind::General;
    IRTargetBuiltinVarName targetVarName = IRTargetBuiltinVarName::Unknown;
    auto semanticInst = varLayout->findSystemValueSemanticAttr();
    if (!semanticInst)
        return nullptr;

    String semanticNameSpelling = semanticInst->getName();
    auto semanticName = semanticNameSpelling.toLower();

    // HLSL semantic types can be found here
    // https://docs.microsoft.com/en-us/windows/desktop/direct3dhlsl/dx-graphics-hlsl-semantics
    /// NOTE! While there might be an "official" type for most of these in HLSL, in practice the
    /// user is allowed to declare almost anything that the HLSL compiler can implicitly convert
    /// to/from the correct type

    auto builder = context->getBuilder();
    IRType* requiredType = nullptr;

    if (semanticName == "sv_position")
    {
        // float4 in hlsl & glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_FragCoord.xhtml
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_Position.xhtml

        // This semantic can either work like `gl_FragCoord`
        // when it is used as a fragment shader input, or
        // like `gl_Position` when used in other stages.
        //
        // Note: This isn't as simple as testing input-vs-output,
        // because a user might have a VS output `SV_Position`,
        // and then pass it along to a GS that reads it as input.
        //
        if (stage == Stage::Fragment && kind == LayoutResourceKind::VaryingInput)
        {
            name = "gl_FragCoord";
            systemValueKind = GLSLSystemValueKind::PositionInput;
        }
        else if (stage == Stage::Geometry && kind == LayoutResourceKind::VaryingInput)
        {
            // As a GS input, the correct syntax is `gl_in[...].gl_Position`,
            // but that is not compatible with picking the array dimension later,
            // of course.
            outerArrayName = "gl_in";
            name = "gl_Position";
        }
        else
        {
            name = "gl_Position";
            if (kind == LayoutResourceKind::VaryingOutput)
            {
                systemValueKind = GLSLSystemValueKind::PositionOutput;
            }
        }

        requiredType = builder->getVectorType(
            builder->getBasicType(BaseType::Float),
            builder->getIntValue(builder->getIntType(), 4));
    }
    else if (semanticName == "sv_target")
    {
        // Note: we do *not* need to generate some kind of `gl_`
        // builtin for fragment-shader outputs: they are just
        // ordinary `out` variables, with ordinary `location`s,
        // as far as GLSL is concerned.
        return nullptr;
    }
    else if (semanticName == "sv_clipdistance")
    {
        // TODO: type conversion is required here.

        // float in hlsl & glsl.
        // "Clip distance data. SV_ClipDistance values are each assumed to be a float32 signed
        // distance to a plane." In glsl clipping value meaning is probably different
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_ClipDistance.xhtml

        name = "gl_ClipDistance";
        requiredType = builder->getBasicType(BaseType::Float);

        arrayIndex = int(semanticInst->getIndex());
    }
    else if (semanticName == "sv_culldistance")
    {
        // float in hlsl & glsl.
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_CullDistance.xhtml

        context->requireGLSLExtension(UnownedStringSlice::fromLiteral("ARB_cull_distance"));

        // TODO: type conversion is required here.
        name = "gl_CullDistance";
        requiredType = builder->getBasicType(BaseType::Float);
    }
    else if (semanticName == "sv_coverage")
    {
        // uint in hlsl, int in glsl
        // https://www.opengl.org/sdk/docs/manglsl/docbook4/xhtml/gl_SampleMask.xml

        requiredType = builder->getBasicType(BaseType::Int);

        // Note: `gl_SampleMask` is actually an *array* of `int`,
        // rather than a single scalar. Because HLSL `SV_Coverage`
        // on allows for a 32 bits worth of coverage, we will
        // only use the first array element in the generated GLSL.

        if (kind == LayoutResourceKind::VaryingInput)
        {
            name = "gl_SampleMaskIn";
        }
        else
        {
            name = "gl_SampleMask";
        }
        arrayIndex = 0;
    }
    else if (semanticName == "sv_innercoverage")
    {
        // uint in hlsl, bool in glsl
        // https://www.khronos.org/registry/OpenGL/extensions/NV/NV_conservative_raster_underestimation.txt

        context->requireGLSLExtension(
            UnownedStringSlice::fromLiteral("GL_NV_conservative_raster_underestimation"));

        name = "gl_FragFullyCoveredNV";
        requiredType = builder->getBasicType(BaseType::Bool);
    }
    else if (semanticName == "sv_depth")
    {
        // Float in hlsl & glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_FragDepth.xhtml
        name = "gl_FragDepth";
        requiredType = builder->getBasicType(BaseType::Float);
    }
    else if (semanticName == "sv_depthgreaterequal")
    {
        // TODO: layout(depth_greater) out float gl_FragDepth;

        // Type is 'unknown' in hlsl
        name = "gl_FragDepth";
        requiredType = builder->getBasicType(BaseType::Float);
    }
    else if (semanticName == "sv_depthlessequal")
    {
        // TODO: layout(depth_greater) out float gl_FragDepth;

        // 'unknown' in hlsl, float in glsl
        name = "gl_FragDepth";
        requiredType = builder->getBasicType(BaseType::Float);
    }
    else if (semanticName == "sv_dispatchthreadid")
    {
        // uint3 in hlsl, uvec3 in glsl
        // https://www.opengl.org/sdk/docs/manglsl/docbook4/xhtml/gl_GlobalInvocationID.xml
        name = "gl_GlobalInvocationID";

        requiredType = builder->getVectorType(
            builder->getBasicType(BaseType::UInt),
            builder->getIntValue(builder->getIntType(), 3));
    }
    else if (semanticName == "sv_domainlocation")
    {
        // float2|3 in hlsl, vec3 in glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_TessCoord.xhtml

        requiredType = builder->getVectorType(
            builder->getBasicType(BaseType::Float),
            builder->getIntValue(builder->getIntType(), 3));

        name = "gl_TessCoord";
    }
    else if (semanticName == "sv_groupid")
    {
        // uint3 in hlsl, uvec3 in glsl
        // https://www.opengl.org/sdk/docs/manglsl/docbook4/xhtml/gl_WorkGroupID.xml
        name = "gl_WorkGroupID";

        requiredType = builder->getVectorType(
            builder->getBasicType(BaseType::UInt),
            builder->getIntValue(builder->getIntType(), 3));
    }
    else if (semanticName == "sv_groupindex")
    {
        // uint in hlsl & in glsl
        name = "gl_LocalInvocationIndex";
        requiredType = builder->getBasicType(BaseType::UInt);
    }
    else if (semanticName == "sv_groupthreadid")
    {
        // uint3 in hlsl, uvec3 in glsl
        name = "gl_LocalInvocationID";

        requiredType = builder->getVectorType(
            builder->getBasicType(BaseType::UInt),
            builder->getIntValue(builder->getIntType(), 3));
    }
    else if (semanticName == "sv_gsinstanceid")
    {
        // uint in hlsl, int in glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_InvocationID.xhtml

        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_InvocationID";
    }
    else if (semanticName == "sv_instanceid")
    {
        // https://docs.microsoft.com/en-us/windows/desktop/direct3d11/d3d10-graphics-programming-guide-input-assembler-stage-using#instanceid
        // uint in hlsl, int in glsl

        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_InstanceIndex";
        targetVarName = IRTargetBuiltinVarName::HlslInstanceID;
        context->requireSPIRVVersion(SemanticVersion(1, 3));
        context->requireGLSLVersion(ProfileVersion::GLSL_460);
        context->requireGLSLExtension(toSlice("GL_ARB_shader_draw_parameters"));
    }
    else if (semanticName == "sv_isfrontface")
    {
        // bool in hlsl & glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_FrontFacing.xhtml
        name = "gl_FrontFacing";
        requiredType = builder->getBasicType(BaseType::Bool);
    }
    else if (semanticName == "sv_outputcontrolpointid")
    {
        // uint in hlsl, int in glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_InvocationID.xhtml

        name = "gl_InvocationID";

        requiredType = builder->getBasicType(BaseType::Int);
    }
    else if (semanticName == "sv_pointsize")
    {
        // float in hlsl & glsl
        name = "gl_PointSize";
        requiredType = builder->getBasicType(BaseType::Float);
    }
    else if (semanticName == "sv_pointcoord")
    {
        name = "gl_PointCoord";
        requiredType = builder->getVectorType(
            builder->getBasicType(BaseType::Float),
            builder->getIntValue(builder->getIntType(), 2));
    }
    else if (semanticName == "sv_drawindex")
    {
        name = "gl_DrawID";
        requiredType = builder->getBasicType(BaseType::Int);
    }
    else if (semanticName == "sv_primitiveid")
    {
        // uint in hlsl, int in glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_PrimitiveID.xhtml
        requiredType = builder->getBasicType(BaseType::Int);

        switch (context->getStage())
        {
        default:
            name = "gl_PrimitiveID";
            break;

        case Stage::Geometry:
            // GLSL makes a confusing design choice here.
            //
            // All the non-GS stages use `gl_PrimitiveID` to access
            // the *input* primitive ID, but a GS uses `gl_PrimitiveID`
            // to acces an *output* primitive ID (that will be passed
            // along to the fragment shader).
            //
            // For a GS to get an input primitive ID (the thing that
            // other stages access with `gl_PrimitiveID`), the
            // programmer must write `gl_PrimitiveIDIn`.
            //
            if (kind == LayoutResourceKind::VaryingInput)
            {
                name = "gl_PrimitiveIDIn";
            }
            else
            {
                name = "gl_PrimitiveID";
            }
            break;
        }
    }
    else if (semanticName == "sv_rendertargetarrayindex")
    {
        // uint on hlsl, int on glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_Layer.xhtml

        switch (context->getStage())
        {
        case Stage::Geometry:
            context->requireGLSLVersion(ProfileVersion::GLSL_150);
            break;

        case Stage::Fragment:
            context->requireGLSLVersion(ProfileVersion::GLSL_430);
            break;

        default:
            context->requireGLSLVersion(ProfileVersion::GLSL_450);
            context->requireSPIRVVersion(SemanticVersion(1, 5, 0));
            context->requireGLSLExtension(
                UnownedStringSlice::fromLiteral("GL_ARB_shader_viewport_layer_array"));
            break;
        }

        name = "gl_Layer";
        requiredType = builder->getBasicType(BaseType::Int);
    }
    else if (semanticName == "sv_sampleindex")
    {
        // uint in hlsl, int in glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_SampleID.xhtml

        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_SampleID";
    }
    else if (semanticName == "sv_stencilref")
    {
        // uint in hlsl, int in glsl
        // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_stencil_export.txt

        requiredType = builder->getBasicType(BaseType::Int);

        context->requireGLSLExtension(UnownedStringSlice::fromLiteral("ARB_shader_stencil_export"));
        name = "gl_FragStencilRef";
    }
    else if (semanticName == "sv_tessfactor")
    {
        // TODO(JS): Adjust type does *not* handle the conversion correctly. More specifically a
        // float array hlsl parameter goes through code to make SOA in createGLSLGlobalVaryingsImpl.
        //
        // Can be input and output.
        //
        // https://docs.microsoft.com/en-us/windows/desktop/direct3dhlsl/sv-tessfactor
        // "Tessellation factors must be declared as an array; they cannot be packed into a single
        // vector."
        //
        // float[2|3|4] in hlsl, float[4] on glsl (ie both are arrays but might be different size)
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_TessLevelOuter.xhtml

        name = "gl_TessLevelOuter";

        // float[4] on glsl
        requiredType = builder->getArrayType(
            builder->getBasicType(BaseType::Float),
            builder->getIntValue(builder->getIntType(), 4));
    }
    else if (semanticName == "sv_insidetessfactor")
    {
        name = "gl_TessLevelInner";

        // float[2] on glsl
        requiredType = builder->getArrayType(
            builder->getBasicType(BaseType::Float),
            builder->getIntValue(builder->getIntType(), 2));
    }
    else if (semanticName == "sv_vertexid")
    {
        // uint in hlsl, int in glsl (https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL))
        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_VertexIndex";
    }
    else if (semanticName == "sv_viewid")
    {
        // uint in hlsl, int in glsl
        // https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_multiview.txt
        requiredType = builder->getBasicType(BaseType::Int);
        context->requireGLSLExtension(UnownedStringSlice::fromLiteral("GL_EXT_multiview"));
        name = "gl_ViewIndex";
    }
    else if (semanticName == "sv_viewportarrayindex")
    {
        // uint on hlsl, int on glsl
        // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/gl_ViewportIndex.xhtml

        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_ViewportIndex";
    }
    else if (semanticName == "nv_x_right")
    {
        context->requireGLSLVersion(ProfileVersion::GLSL_450);
        context->requireGLSLExtension(
            UnownedStringSlice::fromLiteral("GL_NVX_multiview_per_view_attributes"));

        // The actual output in GLSL is:
        //
        //    vec4 gl_PositionPerViewNV[];
        //
        // and is meant to support an arbitrary number of views,
        // while the HLSL case just defines a second position
        // output.
        //
        // For now we will hack this by:
        //   1. Mapping an `NV_X_Right` output to `gl_PositionPerViewNV[1]`
        //      (that is, just one element of the output array)
        //   2. Adding logic to copy the traditional `gl_Position` output
        //      over to `gl_PositionPerViewNV[0]`
        //

        name = "gl_PositionPerViewNV[1]";
        arrayIndex = 1;

        //            shared->requiresCopyGLPositionToPositionPerView = true;
    }
    else if (semanticName == "nv_viewport_mask")
    {
        // TODO: This doesn't seem to work correctly on it's own between hlsl/glsl

        // Indeed on slang issue 109 claims this remains a problem
        // https://github.com/shader-slang/slang/issues/109

        // On hlsl it's UINT related. "higher 16 bits for the right view, lower 16 bits for the left
        // view." There is use in hlsl shader code as uint4 - not clear if that varies
        // https://github.com/KhronosGroup/GLSL/blob/master/extensions/nvx/GL_NVX_multiview_per_view_attributes.txt
        // On glsl its highp int gl_ViewportMaskPerViewNV[];

        context->requireGLSLVersion(ProfileVersion::GLSL_450);
        context->requireGLSLExtension(
            UnownedStringSlice::fromLiteral("GL_NVX_multiview_per_view_attributes"));

        name = "gl_ViewportMaskPerViewNV";
        //            globalVarExpr = createGLSLBuiltinRef("gl_ViewportMaskPerViewNV",
        //                getUnsizedArrayType(getIntType()));
    }
    else if (semanticName == "sv_barycentrics")
    {
        context->requireGLSLVersion(ProfileVersion::GLSL_450);
        context->requireGLSLExtension(
            UnownedStringSlice::fromLiteral("GL_EXT_fragment_shader_barycentric"));
        name = "gl_BaryCoordEXT";

        // TODO: There is also the `gl_BaryCoordNoPerspNV` builtin, which
        // we ought to use if the `noperspective` modifier has been
        // applied to this varying input.
    }
    else if (semanticName == "sv_cullprimitive")
    {
        name = "gl_CullPrimitiveEXT";
    }
    else if (semanticName == "sv_shadingrate")
    {
        if (kind == LayoutResourceKind::VaryingInput)
        {
            name = "gl_ShadingRateEXT";
        }
        else
        {
            name = "gl_PrimitiveShadingRateEXT";
        }
    }
    else if (semanticName == "sv_startvertexlocation")
    {
        context->requireGLSLVersion(ProfileVersion::GLSL_460);

        // uint in hlsl, int in glsl (https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL))
        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_BaseVertex";
    }
    else if (semanticName == "sv_startinstancelocation")
    {
        context->requireGLSLVersion(ProfileVersion::GLSL_460);

        // uint in hlsl, int in glsl (https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL))
        requiredType = builder->getBasicType(BaseType::Int);
        name = "gl_BaseInstance";
    }

    inStorage->targetVarName = targetVarName;
    if (name)
    {
        inStorage->name = name;
        inStorage->outerArrayName = outerArrayName;
        inStorage->requiredType = requiredType;
        inStorage->arrayIndex = arrayIndex;
        inStorage->kind = systemValueKind;
        return inStorage;
    }

    context->getSink()->diagnose(
        varLayout->sourceLoc,
        Diagnostics::unknownSystemValueSemantic,
        semanticNameSpelling);
    return nullptr;
}

// Hold the in-stack linked list that represents the access chain
// to the current global varying parameter being created.
// e.g. if the user code has:
//    struct Params { in float member; }
//    void main(in Params inParams);
// Then the `outerParamInfo` when we get to `createSimpleGLSLVarying` for `member`
// will be:  {IRStructField member} -> {IRParam inParams} -> {IRFunc main}.
//
struct OuterParamInfoLink
{
    IRInst* outerParam;
    OuterParamInfoLink* next;
};

void createVarLayoutForLegalizedGlobalParam(
    GLSLLegalizationContext* context,
    IRInst* globalParam,
    IRBuilder* builder,
    IRVarLayout* inVarLayout,
    IRTypeLayout* typeLayout,
    LayoutResourceKind kind,
    UInt bindingIndex,
    UInt bindingSpace,
    GlobalVaryingDeclarator* declarator,
    OuterParamInfoLink* outerParamInfo,
    GLSLSystemValueInfo* systemValueInfo)
{
    context->usedBindingIndex[kind].add(bindingIndex);

    // We need to construct a fresh layout for the variable, even
    // if the original had its own layout, because it might be
    // an `inout` parameter, and we only want to deal with the case
    // described by our `kind` parameter.
    //
    IRVarLayout::Builder varLayoutBuilder(builder, typeLayout);
    varLayoutBuilder.cloneEverythingButOffsetsFrom(inVarLayout);
    auto varOffsetInfo = varLayoutBuilder.findOrAddResourceInfo(kind);
    varOffsetInfo->offset = bindingIndex;
    varOffsetInfo->space = bindingSpace;
    IRVarLayout* varLayout = varLayoutBuilder.build();
    builder->addLayoutDecoration(globalParam, varLayout);

    // Traverse the entire access chain for the current leaf var and see if
    // there are interpolation mode decorations along the way.
    // Make sure we respect the decoration on the inner most node.
    // So that the decoration on a struct field overrides the outer decoration
    // on a parameter of the struct type.
    for (; outerParamInfo; outerParamInfo = outerParamInfo->next)
    {
        auto paramInfo = outerParamInfo->outerParam;
        auto decorParent = paramInfo;
        if (auto field = as<IRStructField>(decorParent))
            decorParent = field->getKey();
        if (auto interpolationModeDecor =
                decorParent->findDecoration<IRInterpolationModeDecoration>())
        {
            builder->addInterpolationModeDecoration(globalParam, interpolationModeDecor->getMode());
            break;
        }
    }

    if (declarator && declarator->flavor == GlobalVaryingDeclarator::Flavor::meshOutputPrimitives)
    {
        builder->addDecoration(globalParam, kIROp_GLSLPrimitivesRateDecoration);
    }

    if (systemValueInfo)
    {
        builder->addImportDecoration(
            globalParam,
            UnownedTerminatedStringSlice(systemValueInfo->name));

        if (auto outerArrayName = systemValueInfo->outerArrayName)
        {
            builder->addGLSLOuterArrayDecoration(
                globalParam,
                UnownedTerminatedStringSlice(outerArrayName));
        }

        switch (systemValueInfo->kind)
        {
        case GLSLSystemValueKind::PositionOutput:
            builder->addGLPositionOutputDecoration(globalParam);
            break;
        case GLSLSystemValueKind::PositionInput:
            builder->addGLPositionInputDecoration(globalParam);
            break;
        default:
            break;
        }

        if (systemValueInfo->targetVarName != IRTargetBuiltinVarName::Unknown)
        {
            builder->addTargetBuiltinVarDecoration(globalParam, systemValueInfo->targetVarName);
            context->builtinVarMap[systemValueInfo->targetVarName] = globalParam;
        }
    }
}

IRInst* getOrCreateBuiltinParamForHullShader(
    GLSLLegalizationContext* context,
    UnownedStringSlice builtinSemantic)
{
    IRInst* outputControlPointIdParam = nullptr;
    if (context->stage == Stage::Hull)
    {
        for (auto param : context->entryPointFunc->getParams())
        {
            auto layout = findVarLayout(param);
            if (!layout)
                continue;
            auto sysAttr = layout->findSystemValueSemanticAttr();
            if (!sysAttr)
                continue;
            if (sysAttr->getName().caseInsensitiveEquals(builtinSemantic))
            {
                outputControlPointIdParam = param;
                break;
            }
        }
        if (!outputControlPointIdParam)
        {
            IRBuilder builder(context->entryPointFunc);
            auto paramType = builder.getIntType();
            builder.setInsertBefore(
                context->entryPointFunc->getFirstBlock()->getFirstOrdinaryInst());
            outputControlPointIdParam = builder.emitParam(paramType);
            IRStructTypeLayout::Builder typeBuilder(&builder);
            auto typeLayout = typeBuilder.build();
            IRVarLayout::Builder varLayoutBuilder(&builder, typeLayout);
            varLayoutBuilder.setSystemValueSemantic(builtinSemantic, 0);
            varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::VaryingInput);
            auto varLayout = varLayoutBuilder.build();
            builder.addLayoutDecoration(outputControlPointIdParam, varLayout);
        }
    }
    return outputControlPointIdParam;
}

IRTypeLayout* createPatchConstantFuncResultTypeLayout(
    GLSLLegalizationContext* context,
    IRBuilder& irBuilder,
    IRType* type)
{
    if (auto structType = as<IRStructType>(type))
    {
        IRStructTypeLayout::Builder builder(&irBuilder);
        for (auto field : structType->getFields())
        {
            auto fieldType = field->getFieldType();
            IRTypeLayout* fieldTypeLayout =
                createPatchConstantFuncResultTypeLayout(context, irBuilder, fieldType);
            IRVarLayout::Builder fieldVarLayoutBuilder(&irBuilder, fieldTypeLayout);
            auto decoration = field->getKey()->findDecoration<IRSemanticDecoration>();
            if (decoration)
            {
                if (decoration->getSemanticName().startsWithCaseInsensitive(toSlice("sv_")))
                    fieldVarLayoutBuilder.setSystemValueSemantic(decoration->getSemanticName(), 0);
            }
            else
            {
                auto varLayoutForKind =
                    fieldVarLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::VaryingOutput);

                UInt space = 0;
                varLayoutForKind->space = space;

                auto unusedBinding =
                    context->usedBindingIndex[LayoutResourceKind::VaryingOutput].getLSBZero();
                varLayoutForKind->offset = unusedBinding;
                context->usedBindingIndex[LayoutResourceKind::VaryingOutput].add(unusedBinding);
            }
            builder.addField(field->getKey(), fieldVarLayoutBuilder.build());
        }
        auto typeLayout = builder.build();
        return typeLayout;
    }
    else if (auto arrayType = as<IRArrayTypeBase>(type))
    {
        auto elementTypeLayout = createPatchConstantFuncResultTypeLayout(
            context,
            irBuilder,
            arrayType->getElementType());
        IRArrayTypeLayout::Builder builder(&irBuilder, elementTypeLayout);
        return builder.build();
    }
    else
    {
        IRTypeLayout::Builder builder(&irBuilder);
        builder.addResourceUsage(LayoutResourceKind::VaryingOutput, LayoutSize::fromRaw(1));
        return builder.build();
    }
}

ScalarizedVal legalizeEntryPointReturnValueForGLSL(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRBuilder& builder,
    IRFunc* func,
    IRVarLayout* resultLayout);

void invokePathConstantFuncInHullShader(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    ScalarizedVal outputPatchVal)
{
    auto entryPoint = context->entryPointFunc;
    auto patchConstantFuncDecor = entryPoint->findDecoration<IRPatchConstantFuncDecoration>();
    if (!patchConstantFuncDecor)
        return;
    IRInst* inputPatchArg = nullptr;
    for (auto param : entryPoint->getParams())
    {
        if (as<IRHLSLInputPatchType>(param->getDataType()))
        {
            inputPatchArg = param;
            break;
        }
    }
    IRBuilder builder(entryPoint);
    builder.setInsertInto(entryPoint);
    IRBlock* conditionBlock = builder.emitBlock();
    for (auto block : entryPoint->getBlocks())
    {
        if (auto returnInst = as<IRReturn>(block->getTerminator()))
        {
            builder.setInsertBefore(returnInst);
            builder.emitBranch(conditionBlock);
            returnInst->removeAndDeallocate();
        }
    }
    builder.setInsertInto(conditionBlock);
    builder.emitIntrinsicInst(builder.getVoidType(), kIROp_ControlBarrier, 0, nullptr);
    auto index = getOrCreateBuiltinParamForHullShader(context, toSlice("SV_OutputControlPointID"));
    auto condition = builder.emitEql(index, builder.getIntValue(builder.getIntType(), 0));
    auto outputPatchArg = materializeValue(&builder, outputPatchVal);

    List<IRInst*> args;
    auto constantFunc = as<IRFunc>(patchConstantFuncDecor->getFunc());
    for (auto param : constantFunc->getParams())
    {
        if (as<IRHLSLOutputPatchType>(param->getDataType()))
        {
            if (!outputPatchArg)
            {
                context->getSink()->diagnose(
                    param->sourceLoc,
                    Diagnostics::unknownPatchConstantParameter,
                    param);
                return;
            }
            param->setFullType(outputPatchArg->getDataType());
            args.add(outputPatchArg);
        }
        else if (auto inputPatchType = as<IRHLSLInputPatchType>(param->getDataType()))
        {
            if (!inputPatchArg)
            {
                context->getSink()->diagnose(
                    param->sourceLoc,
                    Diagnostics::unknownPatchConstantParameter,
                    param);
                return;
            }
            auto arrayType = builder.getArrayType(
                inputPatchType->getElementType(),
                inputPatchType->getElementCount());
            param->setFullType(arrayType);
            args.add(inputPatchArg);
        }
        else
        {
            auto layout = findVarLayout(param);
            if (!layout)
            {
                context->getSink()->diagnose(
                    param->sourceLoc,
                    Diagnostics::unknownPatchConstantParameter,
                    param);
                return;
            }
            auto sysAttr = layout->findSystemValueSemanticAttr();
            if (!sysAttr)
            {
                context->getSink()->diagnose(
                    param->sourceLoc,
                    Diagnostics::unknownPatchConstantParameter,
                    param);
                return;
            }
            if (sysAttr->getName().caseInsensitiveEquals(toSlice("SV_OutputControlPointID")))
            {
                args.add(getOrCreateBuiltinParamForHullShader(
                    context,
                    toSlice("SV_OutputControlPointID")));
            }
            else if (sysAttr->getName().caseInsensitiveEquals(toSlice("SV_PrimitiveID")))
            {
                args.add(getOrCreateBuiltinParamForHullShader(context, toSlice("SV_PrimitiveID")));
            }
            else
            {
                context->getSink()->diagnose(
                    param->sourceLoc,
                    Diagnostics::unknownPatchConstantParameter,
                    param);
                return;
            }
        }
    }

    IRBlock* trueBlock;
    IRBlock* mergeBlock;
    builder.emitIfWithBlocks(condition, trueBlock, mergeBlock);
    builder.setInsertInto(trueBlock);
    builder.emitCallInst(builder.getVoidType(), constantFunc, args.getArrayView());
    builder.emitBranch(mergeBlock);
    builder.setInsertInto(mergeBlock);
    builder.emitReturn();
    fixUpFuncType(entryPoint, builder.getVoidType());

    if (auto readNoneDecor = constantFunc->findDecoration<IRReadNoneDecoration>())
        readNoneDecor->removeAndDeallocate();
    if (auto noSideEffectDecor = constantFunc->findDecoration<IRNoSideEffectDecoration>())
        noSideEffectDecor->removeAndDeallocate();

    builder.setInsertBefore(constantFunc->getFirstBlock()->getFirstOrdinaryInst());

    auto constantOutputType = constantFunc->getResultType();
    IRTypeLayout* constantOutputLayout =
        createPatchConstantFuncResultTypeLayout(context, builder, constantOutputType);
    IRVarLayout::Builder resultVarLayoutBuilder(&builder, constantOutputLayout);
    if (auto semanticDecor = constantFunc->findDecoration<IRSemanticDecoration>())
        resultVarLayoutBuilder.setSystemValueSemantic(semanticDecor->getSemanticName(), 0);

    context->entryPointFunc = constantFunc;
    context->stage = Stage::Unknown;
    legalizeEntryPointReturnValueForGLSL(
        context,
        codeGenContext,
        builder,
        constantFunc,
        resultVarLayoutBuilder.build());
    context->entryPointFunc = entryPoint;
    context->stage = Stage::Hull;

    fixUpFuncType(constantFunc);
}

ScalarizedVal createSimpleGLSLGlobalVarying(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRBuilder* builder,
    IRType* inType,
    IRVarLayout* inVarLayout,
    IRTypeLayout* inTypeLayout,
    LayoutResourceKind kind,
    Stage stage,
    UInt bindingIndex,
    UInt bindingSpace,
    GlobalVaryingDeclarator* declarator,
    OuterParamInfoLink* outerParamInfo,
    StringBuilder& nameHintSB)
{
    // Check if we have a system value on our hands.
    GLSLSystemValueInfo systemValueInfoStorage;
    auto systemValueInfo = getGLSLSystemValueInfo(
        context,
        codeGenContext,
        inVarLayout,
        kind,
        stage,
        inType,
        declarator,
        &systemValueInfoStorage);

    {

        auto systemSemantic = inVarLayout->findAttr<IRSystemValueSemanticAttr>();
        // Validate the system value, convert to a regular parameter if this is not a valid system
        // value for a given target.
        if (systemSemantic && systemValueInfo && isSPIRV(codeGenContext->getTargetFormat()) &&
            systemValueInfo->targetVarName == IRTargetBuiltinVarName::HlslInstanceID &&
            ((stage == Stage::Fragment) ||
             (stage == Stage::Vertex &&
              inVarLayout->usesResourceKind(LayoutResourceKind::VaryingOutput))))
        {
            ShortList<IRInst*> newOperands;
            auto opCount = inVarLayout->getOperandCount();
            newOperands.reserveOverflowBuffer(opCount);
            for (UInt i = 0; i < opCount; ++i)
            {
                auto op = inVarLayout->getOperand(i);
                if (op == systemSemantic)
                    continue;
                newOperands.add(op);
            }

            auto newVarLayout = builder->emitIntrinsicInst(
                inVarLayout->getFullType(),
                inVarLayout->getOp(),
                newOperands.getCount(),
                newOperands.getArrayView().getBuffer());

            newVarLayout->sourceLoc = inVarLayout->sourceLoc;

            inVarLayout->replaceUsesWith(newVarLayout);
            systemValueInfo->targetVarName = IRTargetBuiltinVarName::Unknown;
        }
    }

    IRType* type = inType;
    IRType* peeledRequiredType = nullptr;
    ShortList<IRInst*> peeledRequiredArraySizes;
    bool peeledRequiredArrayLevelMatchesUserDeclaredType = false;
    // A system-value semantic might end up needing to override the type
    // that the user specified.
    if (systemValueInfo && systemValueInfo->requiredType)
    {
        type = systemValueInfo->requiredType;
        peeledRequiredType = type;
        peeledRequiredArrayLevelMatchesUserDeclaredType = true;
        // Unpeel `type` using declarators so that it matches `inType`.
        for (auto dd = declarator; dd; dd = dd->next)
        {
            switch (dd->flavor)
            {
            case GlobalVaryingDeclarator::Flavor::array:
                {
                    if (auto arrayType = as<IRArrayTypeBase>(type))
                    {
                        type = arrayType->getElementType();
                        peeledRequiredArraySizes.add(arrayType->getElementCount());
                        peeledRequiredType = type;
                    }
                    else
                    {
                        peeledRequiredArrayLevelMatchesUserDeclaredType = false;
                    }
                    break;
                }
            }
        }
    }

    AddressSpace addrSpace = AddressSpace::Uniform;
    IROp ptrOpCode = kIROp_PtrType;
    switch (kind)
    {
    case LayoutResourceKind::VaryingInput:
        addrSpace = systemValueInfo ? AddressSpace::BuiltinInput : AddressSpace::Input;
        break;
    case LayoutResourceKind::VaryingOutput:
        addrSpace = systemValueInfo ? AddressSpace::BuiltinOutput : AddressSpace::Output;
        ptrOpCode = kIROp_OutType;
        break;
    default:
        break;
    }


    // If we have a declarator, we just use the normal logic, as that seems to work correctly
    //
    if (systemValueInfo && systemValueInfo->arrayIndex >= 0 && declarator == nullptr)
    {
        // If declarator is set we have a problem, because we can't have an array of arrays
        // so for now that's just an error
        if (kind != LayoutResourceKind::VaryingOutput && kind != LayoutResourceKind::VaryingInput)
        {
            SLANG_UNIMPLEMENTED_X("Can't handle anything but VaryingOutput and VaryingInput.");
        }

        // Let's see if it has been already created

        // Note! Assumes that the memory backing the name stays in scope! Does if the memory is
        // string constants
        UnownedTerminatedStringSlice systemValueName(systemValueInfo->name);

        auto semanticGlobal = context->systemNameToGlobalMap.tryGetValue(systemValueName);

        if (semanticGlobal == nullptr)
        {
            // Otherwise we just create and add
            GLSLLegalizationContext::SystemSemanticGlobal semanticGlobalTmp;

            // We need to create the global. For now we *don't* know how many indices will be used.
            // So we will

            // Create the array type, but *don't* set the array size, because at this point we don't
            // know. We can at the end replace any accesses to this variable with the correctly
            // sized global

            semanticGlobalTmp.maxIndex = Count(systemValueInfo->arrayIndex);

            // Set the array size to 0, to mean it is unsized
            auto arrayType = builder->getArrayType(type, 0);

            IRType* paramType = builder->getPtrType(ptrOpCode, arrayType, addrSpace);

            auto globalParam = addGlobalParam(builder->getModule(), paramType);
            moveValueBefore(globalParam, builder->getFunc());

            builder->addImportDecoration(globalParam, systemValueName);

            createVarLayoutForLegalizedGlobalParam(
                context,
                globalParam,
                builder,
                inVarLayout,
                inTypeLayout,
                kind,
                bindingIndex,
                bindingSpace,
                declarator,
                outerParamInfo,
                systemValueInfo);

            semanticGlobalTmp.globalParam = globalParam;

            semanticGlobal =
                &context->systemNameToGlobalMap.getOrAddValue(systemValueName, semanticGlobalTmp);
        }

        // Update the max
        semanticGlobal->addIndex(systemValueInfo->arrayIndex);

        // Make it an array index
        ScalarizedVal val = ScalarizedVal::address(semanticGlobal->globalParam);
        RefPtr<ScalarizedArrayIndexValImpl> arrayImpl = new ScalarizedArrayIndexValImpl();
        arrayImpl->arrayVal = val;
        arrayImpl->index = systemValueInfo->arrayIndex;
        arrayImpl->elementType = type;
        val = ScalarizedVal::scalarizedArrayIndex(arrayImpl);

        // We need to make this access, an array access to the global
        if (auto fromType = systemValueInfo->requiredType)
        {
            // We may need to adapt from the declared type to/from
            // the actual type of the GLSL global.
            auto toType = inType;

            if (!isTypeEqual(fromType, toType))
            {
                RefPtr<ScalarizedTypeAdapterValImpl> typeAdapter = new ScalarizedTypeAdapterValImpl;
                typeAdapter->actualType = systemValueInfo->requiredType;
                typeAdapter->pretendType = inType;
                typeAdapter->val = val;

                val = ScalarizedVal::typeAdapter(typeAdapter);
            }
        }

        return val;
    }

    // Construct the actual type and type-layout for the global variable
    //
    IRTypeLayout* typeLayout = inTypeLayout;
    Index requiredArraySizeIndex = peeledRequiredArraySizes.getCount() - 1;
    for (auto dd = declarator; dd; dd = dd->next)
    {
        switch (dd->flavor)
        {
        case GlobalVaryingDeclarator::Flavor::array:
            {
                auto elementCount = peeledRequiredArrayLevelMatchesUserDeclaredType
                                        ? peeledRequiredArraySizes[requiredArraySizeIndex]
                                        : dd->elementCount;

                auto arrayType = builder->getArrayType(type, elementCount);
                requiredArraySizeIndex--;

                IRArrayTypeLayout::Builder arrayTypeLayoutBuilder(builder, typeLayout);
                if (auto resInfo = inTypeLayout->findSizeAttr(kind))
                {
                    // TODO: it is kind of gross to be re-running some
                    // of the type layout logic here.

                    arrayTypeLayoutBuilder.addResourceUsage(
                        kind,
                        resInfo->getSize() * getIntVal(elementCount));
                }
                auto arrayTypeLayout = arrayTypeLayoutBuilder.build();

                type = arrayType;
                typeLayout = arrayTypeLayout;
            }
            break;
        case GlobalVaryingDeclarator::Flavor::meshOutputVertices:
        case GlobalVaryingDeclarator::Flavor::meshOutputIndices:
        case GlobalVaryingDeclarator::Flavor::meshOutputPrimitives:
            {
                // It's legal to declare these as unsized arrays, but by sizing
                // them by the (max) max size GLSL allows us to index into them
                // with variable index.
                SLANG_ASSERT(
                    dd->elementCount && "Mesh output declarator didn't specify element count");
                auto arrayType = builder->getArrayType(type, dd->elementCount);

                IRArrayTypeLayout::Builder arrayTypeLayoutBuilder(builder, typeLayout);
                if (auto resInfo = inTypeLayout->findSizeAttr(kind))
                {
                    // Although these are arrays, they consume slots as though
                    // they're scalar parameters, so don't multiply the usage by the
                    // (runtime) array size.
                    arrayTypeLayoutBuilder.addResourceUsage(kind, resInfo->getSize());
                }
                auto arrayTypeLayout = arrayTypeLayoutBuilder.build();

                type = arrayType;
                typeLayout = arrayTypeLayout;
            }
            break;
        }
    }

    // We are going to be creating a global parameter to replace
    // the function parameter, but we need to handle the case
    // where the parameter represents a varying *output* and not
    // just an input.
    //
    // Our IR global shader parameters are read-only, just
    // like our IR function parameters, and need a wrapper
    // `Out<...>` type to represent outputs.
    //

    // Non system value varying inputs shall be passed as pointers.
    IRType* paramType = builder->getPtrType(ptrOpCode, type, addrSpace);

    auto globalParam = addGlobalParam(builder->getModule(), paramType);
    moveValueBefore(globalParam, builder->getFunc());

    ScalarizedVal val = ScalarizedVal::address(globalParam);

    if (systemValueInfo)
    {
        if (systemValueInfo->requiredType)
        {
            // We may need to adapt from the declared type to/from
            // the actual type of the GLSL global.
            if (!isTypeEqual(peeledRequiredType, inType))
            {
                RefPtr<ScalarizedTypeAdapterValImpl> typeAdapter = new ScalarizedTypeAdapterValImpl;
                typeAdapter->actualType = peeledRequiredType;
                typeAdapter->pretendType = inType;
                typeAdapter->val = val;

                val = ScalarizedVal::typeAdapter(typeAdapter);
            }

            if (auto requiredArrayType = as<IRArrayTypeBase>(systemValueInfo->requiredType))
            {
                // Find first array declarator and handle size mismatch
                for (auto dd = declarator; dd; dd = dd->next)
                {
                    if (dd->flavor != GlobalVaryingDeclarator::Flavor::array)
                        continue;

                    // Compare the array size
                    auto declaredArraySize = dd->elementCount;
                    auto requiredArraySize = requiredArrayType->getElementCount();
                    if (declaredArraySize == requiredArraySize)
                        break;

                    auto toSize = getIntVal(requiredArraySize);
                    auto fromSize = getIntVal(declaredArraySize);
                    if (toSize < fromSize)
                    {
                        context->getSink()->diagnose(
                            inVarLayout,
                            Diagnostics::cannotConvertArrayOfSmallerToLargerSize,
                            fromSize,
                            toSize);
                    }

                    // Array sizes differ, need type adapter
                    RefPtr<ScalarizedTypeAdapterValImpl> typeAdapter =
                        new ScalarizedTypeAdapterValImpl;
                    typeAdapter->actualType = systemValueInfo->requiredType;
                    typeAdapter->pretendType = builder->getArrayType(inType, declaredArraySize);
                    typeAdapter->val = val;

                    val = ScalarizedVal::typeAdapter(typeAdapter);
                    break;
                }
            }
        }
    }
    else
    {
        if (nameHintSB.getLength())
        {
            builder->addNameHintDecoration(globalParam, nameHintSB.getUnownedSlice());
        }
    }

    createVarLayoutForLegalizedGlobalParam(
        context,
        globalParam,
        builder,
        inVarLayout,
        typeLayout,
        kind,
        bindingIndex,
        bindingSpace,
        declarator,
        outerParamInfo,
        systemValueInfo);
    return val;
}

ScalarizedVal createGLSLGlobalVaryingsImpl(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRBuilder* builder,
    IRType* type,
    IRVarLayout* varLayout,
    IRTypeLayout* typeLayout,
    LayoutResourceKind kind,
    Stage stage,
    UInt bindingIndex,
    UInt bindingSpace,
    GlobalVaryingDeclarator* declarator,
    OuterParamInfoLink* outerParamInfo,
    IRInst* leafVar,
    StringBuilder& nameHintSB)
{
    if (as<IRVoidType>(type))
    {
        return ScalarizedVal();
    }
    else if (as<IRBasicType>(type))
    {
        return createSimpleGLSLGlobalVarying(
            context,
            codeGenContext,
            builder,
            type,
            varLayout,
            typeLayout,
            kind,
            stage,
            bindingIndex,
            bindingSpace,
            declarator,
            outerParamInfo,
            nameHintSB);
    }
    else if (as<IRVectorType>(type))
    {
        return createSimpleGLSLGlobalVarying(
            context,
            codeGenContext,
            builder,
            type,
            varLayout,
            typeLayout,
            kind,
            stage,
            bindingIndex,
            bindingSpace,
            declarator,
            outerParamInfo,
            nameHintSB);
    }
    else if (as<IRMatrixType>(type))
    {
        // TODO: a matrix-type varying should probably be handled like an array of rows
        return createSimpleGLSLGlobalVarying(
            context,
            codeGenContext,
            builder,
            type,
            varLayout,
            typeLayout,
            kind,
            stage,
            bindingIndex,
            bindingSpace,
            declarator,
            outerParamInfo,
            nameHintSB);
    }
    else if (auto arrayType = as<IRArrayType>(type))
    {
        // We will need to SOA-ize any nested types.

        auto elementType = arrayType->getElementType();
        auto elementCount = arrayType->getElementCount();
        auto arrayLayout = as<IRArrayTypeLayout>(typeLayout);
        SLANG_ASSERT(arrayLayout);
        auto elementTypeLayout = arrayLayout->getElementTypeLayout();

        GlobalVaryingDeclarator arrayDeclarator;
        arrayDeclarator.flavor = GlobalVaryingDeclarator::Flavor::array;
        arrayDeclarator.elementCount = elementCount;
        arrayDeclarator.next = declarator;

        return createGLSLGlobalVaryingsImpl(
            context,
            codeGenContext,
            builder,
            elementType,
            varLayout,
            elementTypeLayout,
            kind,
            stage,
            bindingIndex,
            bindingSpace,
            &arrayDeclarator,
            outerParamInfo,
            leafVar,
            nameHintSB);
    }
    else if (auto meshOutputType = as<IRMeshOutputType>(type))
    {
        // We will need to SOA-ize any nested types.
        // TODO: Ellie, deduplicate with the above case?

        auto elementType = meshOutputType->getElementType();
        auto arrayLayout = as<IRArrayTypeLayout>(typeLayout);
        SLANG_ASSERT(arrayLayout);
        auto elementTypeLayout = arrayLayout->getElementTypeLayout();

        GlobalVaryingDeclarator arrayDeclarator;
        switch (type->getOp())
        {
            using F = GlobalVaryingDeclarator::Flavor;
        case kIROp_VerticesType:
            arrayDeclarator.flavor = F::meshOutputVertices;
            break;
        case kIROp_IndicesType:
            arrayDeclarator.flavor = F::meshOutputIndices;
            break;
        case kIROp_PrimitivesType:
            arrayDeclarator.flavor = F::meshOutputPrimitives;
            break;
        default:
            SLANG_UNEXPECTED("Unhandled mesh output type");
        }
        arrayDeclarator.elementCount = meshOutputType->getMaxElementCount();
        arrayDeclarator.next = declarator;

        return createGLSLGlobalVaryingsImpl(
            context,
            codeGenContext,
            builder,
            elementType,
            varLayout,
            elementTypeLayout,
            kind,
            stage,
            bindingIndex,
            bindingSpace,
            &arrayDeclarator,
            outerParamInfo,
            leafVar,
            nameHintSB);
    }
    else if (auto streamType = as<IRHLSLStreamOutputType>(type))
    {
        auto elementType = streamType->getElementType();
        auto streamLayout = as<IRStreamOutputTypeLayout>(typeLayout);
        SLANG_ASSERT(streamLayout);
        auto elementTypeLayout = streamLayout->getElementTypeLayout();

        return createGLSLGlobalVaryingsImpl(
            context,
            codeGenContext,
            builder,
            elementType,
            varLayout,
            elementTypeLayout,
            kind,
            stage,
            bindingIndex,
            bindingSpace,
            declarator,
            outerParamInfo,
            leafVar,
            nameHintSB);
    }
    else if (auto structType = as<IRStructType>(type))
    {
        // We need to recurse down into the individual fields,
        // and generate a variable for each of them.

        auto structTypeLayout = as<IRStructTypeLayout>(typeLayout);
        SLANG_ASSERT(structTypeLayout);
        RefPtr<ScalarizedTupleValImpl> tupleValImpl = new ScalarizedTupleValImpl();

        // Since we are going to recurse into struct fields,
        // we need to create a new node in `outerParamInfo` to keep track of
        // the access chain to get to the new leafVar.
        OuterParamInfoLink fieldParentInfo;
        fieldParentInfo.next = outerParamInfo;

        // Construct the actual type for the tuple (including any outer arrays)
        IRType* fullType = type;
        for (auto dd = declarator; dd; dd = dd->next)
        {
            switch (dd->flavor)
            {
            case GlobalVaryingDeclarator::Flavor::meshOutputVertices:
            case GlobalVaryingDeclarator::Flavor::meshOutputIndices:
            case GlobalVaryingDeclarator::Flavor::meshOutputPrimitives:
            case GlobalVaryingDeclarator::Flavor::array:
                {
                    fullType = builder->getArrayType(fullType, dd->elementCount);
                }
                break;
            }
        }

        tupleValImpl->type = fullType;

        // Okay, we want to walk through the fields here, and
        // generate one variable for each.
        UInt fieldCounter = 0;
        auto nameSBLength = nameHintSB.getLength();

        for (auto field : structType->getFields())
        {
            UInt fieldIndex = fieldCounter++;

            auto fieldLayout = structTypeLayout->getFieldLayout(fieldIndex);

            UInt fieldBindingIndex = bindingIndex;
            UInt fieldBindingSpace = bindingSpace;
            if (auto fieldResInfo = fieldLayout->findOffsetAttr(kind))
            {
                fieldBindingIndex += fieldResInfo->getOffset();
                fieldBindingSpace += fieldResInfo->getSpace();
            }
            nameHintSB.reduceLength(nameSBLength);
            if (auto fieldNameHint = field->getKey()->findDecoration<IRNameHintDecoration>())
            {
                if (nameHintSB.getLength() != 0)
                    nameHintSB << ".";
                nameHintSB << fieldNameHint->getName();
            }
            fieldParentInfo.outerParam = field;
            auto fieldVal = createGLSLGlobalVaryingsImpl(
                context,
                codeGenContext,
                builder,
                field->getFieldType(),
                fieldLayout,
                fieldLayout->getTypeLayout(),
                kind,
                stage,
                fieldBindingIndex,
                fieldBindingSpace,
                declarator,
                &fieldParentInfo,
                field,
                nameHintSB);
            if (fieldVal.flavor != ScalarizedVal::Flavor::none)
            {
                ScalarizedTupleValImpl::Element element;
                element.val = fieldVal;
                element.key = field->getKey();

                tupleValImpl->elements.add(element);
            }
        }

        return ScalarizedVal::tuple(tupleValImpl);
    }

    // Default case is to fall back on the simple behavior
    return createSimpleGLSLGlobalVarying(
        context,
        codeGenContext,
        builder,
        type,
        varLayout,
        typeLayout,
        kind,
        stage,
        bindingIndex,
        bindingSpace,
        declarator,
        outerParamInfo,
        nameHintSB);
}

ScalarizedVal createGLSLGlobalVaryings(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRBuilder* builder,
    IRType* type,
    IRVarLayout* layout,
    LayoutResourceKind kind,
    Stage stage,
    IRInst* leafVar)
{
    UInt bindingIndex = 0;
    UInt bindingSpace = 0;
    if (auto rr = layout->findOffsetAttr(kind))
    {
        bindingIndex = rr->getOffset();
        bindingSpace = rr->getSpace();
    }
    StringBuilder namehintSB;
    if (auto nameHint = leafVar->findDecoration<IRNameHintDecoration>())
    {
        if (leafVar->getOp() == kIROp_Func)
            namehintSB << "entryPointParam_";
        namehintSB << nameHint->getName();
    }
    OuterParamInfoLink outerParamInfo;
    outerParamInfo.next = nullptr;
    outerParamInfo.outerParam = leafVar;

    GlobalVaryingDeclarator* declarator = nullptr;
    GlobalVaryingDeclarator arrayDeclarator;
    if (stage == Stage::Hull && kind == LayoutResourceKind::VaryingOutput)
    {
        // Hull shader's output should be materialized into an array.
        auto outputControlPointsDecor =
            context->entryPointFunc->findDecoration<IROutputControlPointsDecoration>();
        if (outputControlPointsDecor)
        {
            arrayDeclarator.flavor = GlobalVaryingDeclarator::Flavor::array;
            arrayDeclarator.next = nullptr;
            arrayDeclarator.elementCount = outputControlPointsDecor->getControlPointCount();
            declarator = &arrayDeclarator;
        }
    }

    return createGLSLGlobalVaryingsImpl(
        context,
        codeGenContext,
        builder,
        type,
        layout,
        layout->getTypeLayout(),
        kind,
        stage,
        bindingIndex,
        bindingSpace,
        declarator,
        &outerParamInfo,
        leafVar,
        namehintSB);
}

ScalarizedVal extractField(
    IRBuilder* builder,
    ScalarizedVal const& val,
    // Pass ~0 in to search for the index via the key
    UInt fieldIndex,
    IRStructKey* fieldKey)
{
    switch (val.flavor)
    {
    case ScalarizedVal::Flavor::value:
        return ScalarizedVal::value(builder->emitFieldExtract(
            getFieldType(val.irValue->getDataType(), fieldKey),
            val.irValue,
            fieldKey));

    case ScalarizedVal::Flavor::address:
        {
            auto ptrType = as<IRPtrTypeBase>(val.irValue->getDataType());
            auto valType = ptrType->getValueType();
            auto fieldType = getFieldType(valType, fieldKey);
            auto fieldPtrType = builder->getPtrType(ptrType->getOp(), fieldType);
            return ScalarizedVal::address(
                builder->emitFieldAddress(fieldPtrType, val.irValue, fieldKey));
        }

    case ScalarizedVal::Flavor::tuple:
        {
            auto tupleVal = as<ScalarizedTupleValImpl>(val.impl);
            const auto& es = tupleVal->elements;
            if (fieldIndex == kMaxUInt)
            {
                for (fieldIndex = 0; fieldIndex < (UInt)es.getCount(); ++fieldIndex)
                {
                    if (es[fieldIndex].key == fieldKey)
                    {
                        break;
                    }
                }
                if (fieldIndex >= (UInt)es.getCount())
                {
                    SLANG_UNEXPECTED("Unable to find field index from struct key");
                }
            }
            return es[fieldIndex].val;
        }

    default:
        SLANG_UNEXPECTED("unimplemented");
        UNREACHABLE_RETURN(ScalarizedVal());
    }
}

ScalarizedVal adaptType(IRBuilder* builder, IRInst* val, IRType* toType, IRType* fromType)
{
    if (auto fromVector = as<IRVectorType>(fromType))
    {
        if (auto toVector = as<IRVectorType>(toType))
        {
            if (fromVector->getElementCount() != toVector->getElementCount())
            {
                fromType = builder->getVectorType(
                    fromVector->getElementType(),
                    toVector->getElementCount());
                val = builder->emitVectorReshape(fromType, val);
            }
        }
        else if (as<IRBasicType>(toType))
        {
            UInt index = 0;
            val = builder->emitSwizzle(fromVector->getElementType(), val, 1, &index);
        }
    }
    else if (auto fromArray = as<IRArrayTypeBase>(fromType))
    {
        if (as<IRBasicType>(toType))
        {
            val = builder->emitElementExtract(
                fromArray->getElementType(),
                val,
                builder->getIntValue(builder->getIntType(), 0));
        }
        else if (auto toArray = as<IRArrayTypeBase>(toType))
        {
            // If array sizes differ, we need to reshape the array
            if (fromArray->getElementCount() != toArray->getElementCount())
            {
                List<IRInst*> elements;

                // Get array sizes once
                auto fromSize = getIntVal(fromArray->getElementCount());
                auto toSize = getIntVal(toArray->getElementCount());

                // Extract elements one at a time up to the minimum
                // size, between the source and destination.
                //
                auto limit = fromSize < toSize ? fromSize : toSize;
                for (Index i = 0; i < limit; i++)
                {
                    auto element = builder->emitElementExtract(
                        fromArray->getElementType(),
                        val,
                        builder->getIntValue(builder->getIntType(), i));
                    elements.add(element);
                }

                if (fromSize < toSize)
                {
                    // Fill remaining elements with default value up to target size
                    auto elementType = toArray->getElementType();
                    auto defaultValue = builder->emitDefaultConstruct(elementType);
                    for (Index i = fromSize; i < toSize; i++)
                    {
                        elements.add(defaultValue);
                    }
                }

                val = builder->emitMakeArray(toType, elements.getCount(), elements.getBuffer());
            }
        }
    }
    // TODO: actually consider what needs to go on here...
    return ScalarizedVal::value(builder->emitCast(toType, val));
}

ScalarizedVal adaptType(
    IRBuilder* builder,
    ScalarizedVal const& val,
    IRType* toType,
    IRType* fromType)
{
    switch (val.flavor)
    {
    case ScalarizedVal::Flavor::value:
        return adaptType(builder, val.irValue, toType, fromType);
        break;

    case ScalarizedVal::Flavor::address:
        {
            auto loaded = builder->emitLoad(val.irValue);
            return adaptType(builder, loaded, toType, fromType);
        }
        break;
    case ScalarizedVal::Flavor::arrayIndex:
        {
            auto arrayImpl = as<ScalarizedArrayIndexValImpl>(val.impl);
            auto elementVal =
                getSubscriptVal(builder, fromType, arrayImpl->arrayVal, arrayImpl->index);
            return adaptType(builder, elementVal, toType, fromType);
        }
        break;
    default:
        SLANG_UNEXPECTED("unimplemented");
        UNREACHABLE_RETURN(ScalarizedVal());
    }
}

void assign(
    IRBuilder* builder,
    ScalarizedVal const& left,
    ScalarizedVal const& right,
    // Pass nullptr for an unindexed write (for everything but mesh shaders)
    IRInst* index = nullptr)
{
    switch (left.flavor)
    {
    case ScalarizedVal::Flavor::arrayIndex:
        {
            // Get the rhs value
            auto rhs = materializeValue(builder, right);

            // Determine the index
            auto leftArrayIndexVal = as<ScalarizedArrayIndexValImpl>(left.impl);
            auto leftVal = getSubscriptVal(
                builder,
                leftArrayIndexVal->elementType,
                leftArrayIndexVal->arrayVal,
                leftArrayIndexVal->index);
            builder->emitStore(leftVal.irValue, rhs);

            break;
        }
    case ScalarizedVal::Flavor::address:
        {
            switch (right.flavor)
            {
            case ScalarizedVal::Flavor::value:
                {
                    auto address = left.irValue;
                    if (index)
                    {
                        address = builder->emitElementAddress(
                            builder->getPtrType(right.irValue->getFullType()),
                            left.irValue,
                            index);
                    }
                    builder->emitStore(address, right.irValue);
                    break;
                }
            case ScalarizedVal::Flavor::address:
                {
                    auto val = builder->emitLoad(right.irValue);
                    builder->emitStore(left.irValue, val);
                    break;
                }
            case ScalarizedVal::Flavor::tuple:
                {
                    // We are assigning from a tuple to a destination
                    // that is not a tuple. We will perform assignment
                    // element-by-element.
                    auto rightTupleVal = as<ScalarizedTupleValImpl>(right.impl);
                    Index elementCount = rightTupleVal->elements.getCount();

                    for (Index ee = 0; ee < elementCount; ++ee)
                    {
                        auto rightElement = rightTupleVal->elements[ee];
                        auto leftElementVal = extractField(builder, left, ee, rightElement.key);
                        assign(builder, leftElementVal, rightElement.val, index);
                    }
                    break;
                }

            default:
                SLANG_UNEXPECTED("unimplemented");
                break;
            }
            break;
        }
    case ScalarizedVal::Flavor::tuple:
        {
            // We have a tuple, so we are going to need to try and assign
            // to each of its constituent fields.
            auto leftTupleVal = as<ScalarizedTupleValImpl>(left.impl);
            Index elementCount = leftTupleVal->elements.getCount();

            for (Index ee = 0; ee < elementCount; ++ee)
            {
                auto rightElementVal =
                    extractField(builder, right, ee, leftTupleVal->elements[ee].key);
                assign(builder, leftTupleVal->elements[ee].val, rightElementVal, index);
            }
            break;
        }
    case ScalarizedVal::Flavor::typeAdapter:
        {
            // We are trying to assign to something that had its type adjusted,
            // so we will need to adjust the type of the right-hand side first.
            //
            // In this case we are converting to the actual type of the GLSL variable,
            // from the "pretend" type that it had in the IR before.
            auto typeAdapter = as<ScalarizedTypeAdapterValImpl>(left.impl);
            auto adaptedRight =
                adaptType(builder, right, typeAdapter->actualType, typeAdapter->pretendType);
            assign(builder, typeAdapter->val, adaptedRight, index);
            break;
        }
    default:
        {
            SLANG_UNEXPECTED("unimplemented");
            break;
        }
    }
}

ScalarizedVal getSubscriptVal(
    IRBuilder* builder,
    IRType* elementType,
    ScalarizedVal val,
    IRInst* indexVal)
{
    switch (val.flavor)
    {
    case ScalarizedVal::Flavor::value:
        return ScalarizedVal::value(
            builder->emitElementExtract(elementType, val.irValue, indexVal));

    case ScalarizedVal::Flavor::address:
        return ScalarizedVal::address(
            builder->emitElementAddress(builder->getPtrType(elementType), val.irValue, indexVal));

    case ScalarizedVal::Flavor::tuple:
        {
            auto inputTuple = val.impl.as<ScalarizedTupleValImpl>();

            RefPtr<ScalarizedTupleValImpl> resultTuple = new ScalarizedTupleValImpl();
            resultTuple->type = elementType;

            Index elementCount = inputTuple->elements.getCount();
            Index elementCounter = 0;

            auto structType = as<IRStructType>(elementType);
            for (auto field : structType->getFields())
            {
                auto tupleElementType = field->getFieldType();

                Index elementIndex = elementCounter++;

                SLANG_RELEASE_ASSERT(elementIndex < elementCount);
                auto inputElement = inputTuple->elements[elementIndex];

                ScalarizedTupleValImpl::Element resultElement;
                resultElement.key = inputElement.key;
                resultElement.val =
                    getSubscriptVal(builder, tupleElementType, inputElement.val, indexVal);

                resultTuple->elements.add(resultElement);
            }
            SLANG_RELEASE_ASSERT(elementCounter == elementCount);

            return ScalarizedVal::tuple(resultTuple);
        }
    case ScalarizedVal::Flavor::typeAdapter:
        {
            auto inputAdapter = val.impl.as<ScalarizedTypeAdapterValImpl>();
            RefPtr<ScalarizedTypeAdapterValImpl> resultAdapter = new ScalarizedTypeAdapterValImpl();

            resultAdapter->pretendType = inputAdapter->pretendType;
            resultAdapter->actualType = inputAdapter->actualType;

            resultAdapter->val =
                getSubscriptVal(builder, inputAdapter->actualType, inputAdapter->val, indexVal);
            return ScalarizedVal::typeAdapter(resultAdapter);
        }

    default:
        SLANG_UNEXPECTED("unimplemented");
        UNREACHABLE_RETURN(ScalarizedVal());
    }
}

ScalarizedVal getSubscriptVal(
    IRBuilder* builder,
    IRType* elementType,
    ScalarizedVal val,
    UInt index)
{
    return getSubscriptVal(
        builder,
        elementType,
        val,
        builder->getIntValue(builder->getIntType(), index));
}

IRInst* materializeValue(IRBuilder* builder, ScalarizedVal const& val);

IRInst* materializeTupleValue(IRBuilder* builder, ScalarizedVal val)
{
    auto tupleVal = val.impl.as<ScalarizedTupleValImpl>();
    SLANG_ASSERT(tupleVal);

    Index elementCount = tupleVal->elements.getCount();
    auto type = tupleVal->type;

    if (auto arrayType = as<IRArrayType>(type))
    {
        // The tuple represent an array, which means that the
        // individual elements are expected to yield arrays as well.
        //
        // We will extract a value for each array element, and
        // then use these to construct our result.

        List<IRInst*> arrayElementVals;
        UInt arrayElementCount = (UInt)getIntVal(arrayType->getElementCount());

        for (UInt ii = 0; ii < arrayElementCount; ++ii)
        {
            auto arrayElementPseudoVal =
                getSubscriptVal(builder, arrayType->getElementType(), val, ii);

            auto arrayElementVal = materializeValue(builder, arrayElementPseudoVal);

            arrayElementVals.add(arrayElementVal);
        }

        return builder->emitMakeArray(
            arrayType,
            arrayElementVals.getCount(),
            arrayElementVals.getBuffer());
    }
    else
    {
        // The tuple represents a value of some aggregate type,
        // so we can simply materialize the elements and then
        // construct a value of that type.
        //
        SLANG_RELEASE_ASSERT(as<IRStructType>(type));

        List<IRInst*> elementVals;
        for (Index ee = 0; ee < elementCount; ++ee)
        {
            auto elementVal = materializeValue(builder, tupleVal->elements[ee].val);
            elementVals.add(elementVal);
        }

        return builder->emitMakeStruct(
            tupleVal->type,
            elementVals.getCount(),
            elementVals.getBuffer());
    }
}

IRInst* materializeValue(IRBuilder* builder, ScalarizedVal const& val)
{
    switch (val.flavor)
    {
    case ScalarizedVal::Flavor::value:
        return val.irValue;

    case ScalarizedVal::Flavor::address:
        {
            auto loadInst = builder->emitLoad(val.irValue);
            return loadInst;
        }
        break;

    case ScalarizedVal::Flavor::arrayIndex:
        {
            auto impl = as<ScalarizedArrayIndexValImpl>(val.impl);
            auto elementVal =
                getSubscriptVal(builder, impl->elementType, impl->arrayVal, impl->index);
            return materializeValue(builder, elementVal);
        }
    case ScalarizedVal::Flavor::tuple:
        {
            // auto tupleVal = as<ScalarizedTupleValImpl>(val.impl);
            return materializeTupleValue(builder, val);
        }
        break;

    case ScalarizedVal::Flavor::typeAdapter:
        {
            // Somebody is trying to use a value where its actual type
            // doesn't match the type it pretends to have. To make this
            // work we need to adapt the type from its actual type over
            // to its pretend type.
            auto typeAdapter = as<ScalarizedTypeAdapterValImpl>(val.impl);
            auto adapted = adaptType(
                builder,
                typeAdapter->val,
                typeAdapter->pretendType,
                typeAdapter->actualType);
            return materializeValue(builder, adapted);
        }
        break;

    default:
        SLANG_UNEXPECTED("unimplemented");
        break;
    }
}

void handleSingleParam(
    GLSLLegalizationContext* context,
    IRFunc* func,
    IRParam* pp,
    IRVarLayout* paramLayout)
{
    auto builder = context->getBuilder();
    auto paramType = pp->getDataType();

    // The parameter might be either an `in` parameter,
    // or an `out` or `in out` parameter, and in those
    // latter cases its IR-level type will include a
    // wrapping "pointer-like" type (e.g., `Out<Float>`
    // instead of just `Float`).
    //
    // Because global shader parameters are read-only
    // in the same way function types are, we can take
    // care of that detail here just by allocating a
    // global shader parameter with exactly the type
    // of the original function parameter.
    //
    auto globalParam = addGlobalParam(builder->getModule(), paramType);
    builder->addLayoutDecoration(globalParam, paramLayout);
    moveValueBefore(globalParam, builder->getFunc());
    pp->replaceUsesWith(globalParam);

    // Because linkage between ray-tracing shaders is
    // based on the type of incoming/outgoing payload
    // and attribute parameters, it would be an error to
    // eliminate the global parameter *even if* it is
    // not actually used inside the entry point.
    //
    // We attach a decoration to the entry point that
    // makes note of the dependency, so that steps
    // like dead code elimination cannot get rid of
    // the parameter.
    //
    // TODO: We could consider using a structure like
    // this for *all* of the entry point parameters
    // that get moved to the global scope, since SPIR-V
    // ends up requiring such information on an `OpEntryPoint`.
    //
    // As a further alternative, we could decide to
    // keep entry point varying input/outtput attached
    // to the parameter list through all of the Slang IR
    // steps, and only declare it as global variables at
    // the last minute when emitting a GLSL `main` or
    // SPIR-V for an entry point.
    //
    builder->addDependsOnDecoration(func, globalParam);
}

static void consolidateParameters(GLSLLegalizationContext* context, List<IRParam*>& params)
{
    auto builder = context->getBuilder();

    // Create a struct type to hold all parameters
    IRInst* consolidatedVar = nullptr;
    auto structType = builder->createStructType();

    // Inside the structure, add fields for each parameter
    for (auto _param : params)
    {
        auto _paramType = _param->getDataType();
        IRType* valueType = _paramType;

        if (as<IROutTypeBase>(_paramType))
            valueType = as<IROutTypeBase>(_paramType)->getValueType();

        auto key = builder->createStructKey();
        if (auto nameDecor = _param->findDecoration<IRNameHintDecoration>())
            builder->addNameHintDecoration(key, nameDecor->getName());
        auto field = builder->createStructField(structType, key, valueType);
        field->removeFromParent();
        field->insertAtEnd(structType);
    }

    // Create a global variable to hold the consolidated struct
    consolidatedVar = builder->createGlobalVar(structType);
    auto ptrType = builder->getPtrType(kIROp_PtrType, structType, AddressSpace::IncomingRayPayload);
    consolidatedVar->setFullType(ptrType);
    consolidatedVar->moveToEnd();

    // Add the ray payload decoration and assign location 0.
    builder->addVulkanRayPayloadDecoration(consolidatedVar, 0);

    // Replace each parameter with a field in the consolidated struct
    for (Index i = 0; i < params.getCount(); ++i)
    {
        auto _param = params[i];

        // Find the i-th field
        IRStructField* targetField = nullptr;
        Index fieldIndex = 0;
        for (auto field : structType->getFields())
        {
            if (fieldIndex == i)
            {
                targetField = field;
                break;
            }
            fieldIndex++;
        }
        SLANG_ASSERT(targetField);

        // Create the field address with the correct type
        auto _paramType = _param->getDataType();
        auto fieldType = targetField->getFieldType();

        // If the parameter is an out/inout type, we need to create a pointer type
        IRType* fieldPtrType = nullptr;
        if (as<IROutType>(_paramType))
        {
            fieldPtrType = builder->getPtrType(kIROp_OutType, fieldType);
        }
        else if (as<IRInOutType>(_paramType))
        {
            fieldPtrType = builder->getPtrType(kIROp_InOutType, fieldType);
        }

        auto fieldAddr =
            builder->emitFieldAddress(fieldPtrType, consolidatedVar, targetField->getKey());

        // Replace parameter uses with field address
        _param->replaceUsesWith(fieldAddr);
    }
}

// Consolidate ray tracing parameters for an entry point function
void consolidateRayTracingParameters(GLSLLegalizationContext* context, IRFunc* func)
{
    auto builder = context->getBuilder();
    auto firstBlock = func->getFirstBlock();
    if (!firstBlock)
        return;

    // Collect all out/inout parameters that need to be consolidated
    List<IRParam*> outParams;
    List<IRParam*> params;

    for (auto param = firstBlock->getFirstParam(); param; param = param->getNextParam())
    {
        builder->setInsertBefore(firstBlock->getFirstOrdinaryInst());
        if (as<IROutType>(param->getDataType()) || as<IRInOutType>(param->getDataType()))
        {
            outParams.add(param);
        }
        params.add(param);
    }

    // We don't need consolidation here.
    if (outParams.getCount() <= 1)
    {
        for (auto param : params)
        {
            auto paramLayoutDecoration = param->findDecoration<IRLayoutDecoration>();
            SLANG_ASSERT(paramLayoutDecoration);
            auto paramLayout = as<IRVarLayout>(paramLayoutDecoration->getLayout());
            handleSingleParam(context, func, param, paramLayout);
        }
        return;
    }
    else
    {
        // We need consolidation here, but before that, handle parameters other than inout/out.
        for (auto param : params)
        {
            if (outParams.contains(param))
            {
                continue;
            }
            auto paramLayoutDecoration = param->findDecoration<IRLayoutDecoration>();
            SLANG_ASSERT(paramLayoutDecoration);
            auto paramLayout = as<IRVarLayout>(paramLayoutDecoration->getLayout());
            handleSingleParam(context, func, param, paramLayout);
        }

        // Now, consolidate the inout/out parameters
        consolidateParameters(context, outParams);
    }
}

static void legalizeMeshPayloadInputParam(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRParam* pp)
{
    auto builder = context->getBuilder();
    auto stage = context->getStage();
    SLANG_ASSERT(
        stage == Stage::Mesh && "legalizing mesh payload input, but we're not a mesh shader");
    IRBuilderInsertLocScope locScope{builder};
    builder->setInsertInto(builder->getModule());

    const auto ptrType = cast<IRPtrTypeBase>(pp->getDataType());
    const auto g =
        builder->createGlobalVar(ptrType->getValueType(), AddressSpace::TaskPayloadWorkgroup);
    g->setFullType(builder->getRateQualifiedType(builder->getGroupSharedRate(), g->getFullType()));
    // moveValueBefore(g, builder->getFunc());
    builder->addNameHintDecoration(g, pp->findDecoration<IRNameHintDecoration>()->getName());
    pp->replaceUsesWith(g);
    struct MeshPayloadInputSpecializationCondition : FunctionCallSpecializeCondition
    {
        bool doesParamWantSpecialization(IRParam*, IRInst* arg) { return arg == g; }
        IRInst* g;
    } condition;
    condition.g = g;
    specializeFunctionCalls(codeGenContext, builder->getModule(), &condition);
}

static void legalizePatchParam(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRFunc* func,
    IRParam* pp,
    IRVarLayout* paramLayout,
    IRHLSLPatchType* patchType)
{
    auto builder = context->getBuilder();
    auto elementType = patchType->getElementType();
    auto elementCount = patchType->getElementCount();
    auto arrayType = builder->getArrayType(elementType, elementCount);

    auto globalPatchVal = createGLSLGlobalVaryings(
        context,
        codeGenContext,
        builder,
        arrayType,
        paramLayout,
        LayoutResourceKind::VaryingInput,
        Stage::Hull, // Doesn't matter whether we are in Hull or Domain shader.
        pp);

    builder->setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
    auto materializedVal = materializeValue(builder, globalPatchVal);
    pp->transferDecorationsTo(materializedVal);
    pp->replaceUsesWith(materializedVal);
}

static void legalizeMeshOutputParam(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRFunc* func,
    IRParam* pp,
    IRVarLayout* paramLayout,
    IRMeshOutputType* meshOutputType)
{
    auto builder = context->getBuilder();
    auto stage = context->getStage();
    SLANG_ASSERT(stage == Stage::Mesh && "legalizing mesh output, but we're not a mesh shader");
    IRBuilderInsertLocScope locScope{builder};
    builder->setInsertInto(func);

    auto globalOutputVal = createGLSLGlobalVaryings(
        context,
        codeGenContext,
        builder,
        meshOutputType,
        paramLayout,
        LayoutResourceKind::VaryingOutput,
        stage,
        pp);

    switch (globalOutputVal.flavor)
    {
    case ScalarizedVal::Flavor::tuple:
        {
            auto v = as<ScalarizedTupleValImpl>(globalOutputVal.impl);

            Index elementCount = v->elements.getCount();
            for (Index ee = 0; ee < elementCount; ++ee)
            {
                auto e = v->elements[ee];
                auto leftElementVal = extractField(builder, globalOutputVal, ee, e.key);
            }
        }
        break;
    case ScalarizedVal::Flavor::value:
    case ScalarizedVal::Flavor::address:
    case ScalarizedVal::Flavor::typeAdapter:
        break;
    }

    //
    // Introduce a global parameter to drive the specialization machinery
    //
    // It would potentially be nicer to orthogonalize the SOA-ization
    // and the entry point parameter legalization, however this isn't
    // possible in the general case as we need to know which members
    // map to builtin values and which to user defined varyings;
    // information which is only readily available given the entry
    // point.
    // The AOS handling of builtins is a quirk specific to GLSL and
    // once we've moved to a SPIR-V direct backend we can neaten this
    // up.
    //
    auto g = addGlobalParam(builder->getModule(), pp->getFullType());
    moveValueBefore(g, builder->getFunc());
    builder->addNameHintDecoration(g, pp->findDecoration<IRNameHintDecoration>()->getName());
    pp->replaceUsesWith(g);
    // pp is only removed later on, so sadly we have to keep it around for now
    struct MeshOutputSpecializationCondition : FunctionCallSpecializeCondition
    {
        bool doesParamWantSpecialization(IRParam*, IRInst* arg) { return arg == g; }
        IRInst* g;
    } condition;
    condition.g = g;
    specializeFunctionCalls(codeGenContext, builder->getModule(), &condition);

    //
    // Remove this global by making all writes actually write to the
    // newly introduced out variables.
    //
    // Sadly it's not as simple as just using this file's `assign` function as
    // the writes may only be writing to parts of the output struct, or may not
    // be writes at all (i.e. being passed as an out paramter).
    //
    std::function<void(ScalarizedVal&, IRInst*)> assignUses = [&](ScalarizedVal& d, IRInst* a)
    {
        // If we're just writing to an address, we can seamlessly
        // replace it with the address to the SOA representation.
        // GLSL's `out` function parameters have copy-out semantics, so
        // this is all above board.
        if (d.flavor == ScalarizedVal::Flavor::address)
        {
            IRBuilderInsertLocScope locScope{builder};
            builder->setInsertBefore(a);
            a->replaceUsesWith(d.irValue);
            a->removeAndDeallocate();
            return;
        }
        // Otherwise, go through the uses one by one and see what we can do
        traverseUsers(
            a,
            [&](IRInst* s)
            {
                IRBuilderInsertLocScope locScope{builder};
                builder->setInsertBefore(s);
                if (auto m = as<IRFieldAddress>(s))
                {
                    auto key = as<IRStructKey>(m->getField());
                    SLANG_ASSERT(key && "Result of getField wasn't a struct key");

                    auto d_ = extractField(builder, d, kMaxUInt, key);
                    assignUses(d_, m);
                }
                else if (auto ref = as<IRMeshOutputRef>(s))
                {
                    auto elemType = composeGetters<IRType>(
                        ref,
                        &IRInst::getFullType,
                        &IRPtrTypeBase::getValueType);
                    auto d_ = getSubscriptVal(builder, elemType, d, ref->getIndex());
                    assignUses(d_, ref);
                }
                else if (auto set = as<IRMeshOutputSet>(s))
                {
                    auto elemType =
                        composeGetters<IRType>(set->getElementValue(), &IRInst::getFullType);
                    auto d_ = getSubscriptVal(builder, elemType, d, set->getIndex());
                    assign(builder, d_, ScalarizedVal::value(set->getElementValue()));
                    set->removeAndDeallocate();
                }
                else if (auto g = as<IRGetElementPtr>(s))
                {
                    // Writing to something like `struct Vertex{ Foo foo[10]; }`
                    // This case is also what's taken in the initial
                    // traversal, as every mesh output is an array.
                    auto elemType = composeGetters<IRType>(
                        g,
                        &IRInst::getFullType,
                        &IRPtrTypeBase::getValueType);
                    auto d_ = getSubscriptVal(builder, elemType, d, g->getIndex());
                    assignUses(d_, g);
                }
                else if (auto store = as<IRStore>(s))
                {
                    // Store using the SOA representation

                    assign(builder, d, ScalarizedVal::value(store->getVal()));

                    // Stores aren't used, safe to remove here without checking
                    store->removeAndDeallocate();
                }
                else if (auto c = as<IRCall>(s))
                {
                    // Translate
                    //   foo(vertices[n])
                    // to
                    //   tmp
                    //   foo(tmp)
                    //   vertices[n] = tmp;
                    //
                    // This has copy-out semantics, which is really the
                    // best we can hope for without going and
                    // specializing foo.
                    auto ptr = as<IRPtrTypeBase>(a->getFullType());
                    SLANG_ASSERT(ptr && "Mesh output parameter was passed by value");
                    auto t = ptr->getValueType();
                    auto tmp = builder->emitVar(t);
                    for (UInt i = 0; i < c->getOperandCount(); i++)
                    {
                        if (c->getOperand(i) == a)
                        {
                            c->setOperand(i, tmp);
                        }
                    }
                    builder->setInsertAfter(c);
                    assign(builder, d, ScalarizedVal::value(builder->emitLoad(tmp)));
                }
                else if (const auto load = as<IRLoad>(s))
                {
                    // Handles the case where a `this` points to a IRMeshOutputRef.
                    auto t = as<IRPtrType>(load->getPtr()->getDataType())->getValueType();
                    auto tmp = builder->emitVar(t);
                    assign(builder, ScalarizedVal::address(tmp), d);

                    s->replaceUsesWith(builder->emitLoad(tmp));
                    s->removeAndDeallocate();
                }
                else if (const auto swiz = as<IRSwizzledStore>(s))
                {
                    SLANG_UNEXPECTED("Swizzled store to a non-address ScalarizedVal");
                }
                else
                {
                    SLANG_UNEXPECTED(
                        "Unhandled use of mesh output parameter during GLSL legalization");
                }
            });

        SLANG_ASSERT(!a->hasUses());
        a->removeAndDeallocate();
    };
    assignUses(globalOutputVal, g);

    //
    // GLSL requires that builtins are written to a block named
    // gl_MeshVerticesEXT or gl_MeshPrimitivesEXT. Once we've done the
    // specialization above, we can fix this.
    //
    // This part is split into a separate step so as not to infect
    // ScalarizedVal and also so it can be excised easily when moving
    // to a SPIR-V direct.
    //
    // It's tempting to move this into a separate IR pass which looks for
    // global mesh output params and coalesces them, however the precise
    // definitions of gl_MeshPerVertexEXT can differ depending on the entry
    // point, consider sizing the gl_ClipDistance array for different number of
    // clip planes used by different entry points.
    //
    // We are allowed to redeclare these with just the necessary subset of
    // members.
    //
    // out gl_MeshPerVertexEXT {
    //   vec4  gl_Position;
    //   float gl_PointSize;
    //   float gl_ClipDistance[];
    //   float gl_CullDistance[];
    // } gl_MeshVerticesEXT[];
    //
    // perprimitiveEXT out gl_MeshPerPrimitiveEXT {
    //   int  gl_PrimitiveID;
    //   int  gl_Layer;
    //   int  gl_ViewportIndex;
    //   bool gl_CullPrimitiveEXT;
    //   int  gl_PrimitiveShadingRateEXT;
    // } gl_MeshPrimitivesEXT[];
    //

    // First, collect the subset of outputs being used
    if (!isSPIRV(codeGenContext->getTargetFormat()))
    {
        auto isMeshOutputBuiltin = [](IRInst* g)
        {
            if (const auto s = composeGetters<IRStringLit>(
                    g,
                    &IRInst::findDecoration<IRImportDecoration>,
                    &IRImportDecoration::getMangledNameOperand))
            {
                const auto n = s->getStringSlice();
                if (n == "gl_Position" || n == "gl_PointSize" || n == "gl_ClipDistance" ||
                    n == "gl_CullDistance" || n == "gl_PrimitiveID" || n == "gl_Layer" ||
                    n == "gl_ViewportIndex" || n == "gl_CullPrimitiveEXT" ||
                    n == "gl_PrimitiveShadingRateEXT")
                {
                    return s;
                }
            }
            return (IRStringLit*)nullptr;
        };
        auto leaves = globalOutputVal.leafAddresses();
        struct BuiltinOutputInfo
        {
            IRInst* param;
            IRStringLit* nameDecoration;
            IRType* type;
            IRStructKey* key;
        };
        List<BuiltinOutputInfo> builtins;
        for (auto leaf : leaves)
        {
            if (auto decoration = isMeshOutputBuiltin(leaf))
            {
                builtins.add({leaf, decoration, nullptr, nullptr});
            }
        }
        if (builtins.getCount() == 0)
        {
            return;
        }
        const auto _locScope = IRBuilderInsertLocScope{builder};
        builder->setInsertBefore(func);
        auto meshOutputBlockType = builder->createStructType();
        {
            const auto _locScope2 = IRBuilderInsertLocScope{builder};
            builder->setInsertInto(meshOutputBlockType);
            for (auto& builtin : builtins)
            {
                auto t = composeGetters<IRType>(
                    builtin.param,
                    &IRInst::getFullType,
                    &IROutTypeBase::getValueType,
                    &IRArrayTypeBase::getElementType);
                auto key = builder->createStructKey();
                auto n = builtin.nameDecoration->getStringSlice();
                builder->addImportDecoration(key, n);
                builder->createStructField(meshOutputBlockType, key, t);
                builtin.type = t;
                builtin.key = key;
            }
        }

        // No emitter actually handles GLSLOutputParameterGroupTypes, this isn't a
        // problem as it's used as an intrinsic.
        // GLSL does permit redeclaring these particular ones, so it might be nice to
        // add a linkage decoration instead of it being an intrinsic in the event
        // that we start outputting the linkage decoration instead of it being an
        // intrinsic in the event that we start outputting these.
        auto blockParamType = builder->getGLSLOutputParameterGroupType(
            builder->getArrayType(meshOutputBlockType, meshOutputType->getMaxElementCount()));
        auto blockParam = builder->createGlobalParam(blockParamType);
        bool isPerPrimitive = as<IRPrimitivesType>(meshOutputType);
        auto typeName = isPerPrimitive ? "gl_MeshPerPrimitiveEXT" : "gl_MeshPerVertexEXT";
        auto arrayName = isPerPrimitive ? "gl_MeshPrimitivesEXT" : "gl_MeshVerticesEXT";
        builder->addTargetIntrinsicDecoration(
            meshOutputBlockType,
            CapabilitySet(CapabilityName::glsl),
            UnownedStringSlice(typeName));
        builder->addImportDecoration(blockParam, UnownedStringSlice(arrayName));
        if (isPerPrimitive)
        {
            builder->addDecoration(blockParam, kIROp_GLSLPrimitivesRateDecoration);
        }
        // // While this is probably a correct thing to do, LRK::VaryingOutput
        // // isn't really used for redeclaraion of builtin outputs, and assumes
        // // that it's got a layout location, the correct fix might be to add
        // // LRK::BuiltinVaryingOutput, but that would be polluting LRK for no
        // // real gain.
        // //
        // IRVarLayout::Builder varLayoutBuilder{builder, IRTypeLayout::Builder{builder}.build()};
        // varLayoutBuilder.findOrAddResourceInfo(LayoutResourceKind::VaryingOutput);
        // varLayoutBuilder.setStage(Stage::Mesh);
        // builder->addLayoutDecoration(blockParam, varLayoutBuilder.build());

        for (auto builtin : builtins)
        {
            traverseUsers(
                builtin.param,
                [&](IRInst* u)
                {
                    IRBuilderInsertLocScope locScope{builder};
                    builder->setInsertBefore(u);
                    IRInst* index;
                    if (const auto p = as<IRGetElementPtr>(u))
                        index = p->getIndex();
                    else if (const auto m = as<IRMeshOutputRef>(u))
                        index = m->getIndex();
                    else
                        SLANG_UNEXPECTED("Illegal use of mesh output parameter");
                    auto e = builder->emitElementAddress(
                        builder->getPtrType(meshOutputBlockType),
                        blockParam,
                        index);
                    auto a = builder->emitFieldAddress(
                        builder->getPtrType(builtin.type),
                        e,
                        builtin.key);
                    u->replaceUsesWith(a);
                });
        }
    }

    SLANG_ASSERT(!g->hasUses());
    g->removeAndDeallocate();
}

IRInst* getOrCreatePerVertexInputArray(GLSLLegalizationContext* context, IRInst* inputVertexAttr)
{
    IRInst* arrayInst = nullptr;
    if (context->mapVertexInputToPerVertexArray.tryGetValue(inputVertexAttr, arrayInst))
        return arrayInst;
    IRBuilder builder(inputVertexAttr);
    builder.setInsertBefore(inputVertexAttr);
    auto arrayType = builder.getArrayType(
        tryGetPointedToType(&builder, inputVertexAttr->getDataType()),
        builder.getIntValue(builder.getIntType(), 3));
    arrayInst = builder.createGlobalParam(builder.getPtrType(arrayType, AddressSpace::Input));
    context->mapVertexInputToPerVertexArray[inputVertexAttr] = arrayInst;
    builder.addDecoration(arrayInst, kIROp_PerVertexDecoration);

    // Clone decorations from original input.
    for (auto decoration : inputVertexAttr->getDecorations())
    {
        switch (decoration->getOp())
        {
        case kIROp_InterpolationModeDecoration:
            continue;
        default:
            cloneDecoration(decoration, arrayInst);
            break;
        }
    }
    return arrayInst;
}

void tryReplaceUsesOfStageInput(
    GLSLLegalizationContext* context,
    ScalarizedVal val,
    IRInst* originalVal)
{
    switch (val.flavor)
    {
    case ScalarizedVal::Flavor::value:
        {
            traverseUses(
                originalVal,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    IRBuilder builder(user);
                    builder.setInsertBefore(user);
                    builder.replaceOperand(use, val.irValue);
                });
        }
        break;
    case ScalarizedVal::Flavor::address:
        {
            bool needMaterialize = false;
            if (as<IRPtrTypeBase>(val.irValue->getDataType()))
            {
                if (!as<IRPtrTypeBase>(originalVal->getDataType()))
                {
                    needMaterialize = true;
                }
            }
            traverseUses(
                originalVal,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    if (user->getOp() == kIROp_GetPerVertexInputArray)
                    {
                        auto arrayInst = getOrCreatePerVertexInputArray(context, val.irValue);
                        user->replaceUsesWith(arrayInst);
                        user->removeAndDeallocate();
                        return;
                    }
                    IRBuilder builder(user);
                    builder.setInsertBefore(user);
                    if (needMaterialize)
                    {
                        auto materializedVal = materializeValue(&builder, val);
                        builder.replaceOperand(use, materializedVal);
                    }
                    else
                    {
                        builder.replaceOperand(use, val.irValue);
                    }
                });
        }
        break;
    case ScalarizedVal::Flavor::typeAdapter:
        {
            traverseUses(
                originalVal,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    IRBuilder builder(user);
                    builder.setInsertBefore(user);
                    auto typeAdapter = as<ScalarizedTypeAdapterValImpl>(val.impl);
                    auto materializedInner = materializeValue(&builder, typeAdapter->val);
                    auto adapted = adaptType(
                        &builder,
                        materializedInner,
                        typeAdapter->pretendType,
                        typeAdapter->actualType);
                    if (user->getOp() == kIROp_Load)
                    {
                        user->replaceUsesWith(adapted.irValue);
                        user->removeAndDeallocate();
                    }
                    else
                    {
                        use->set(adapted.irValue);
                    }
                });
        }
        break;
    case ScalarizedVal::Flavor::arrayIndex:
        {
            traverseUses(
                originalVal,
                [&](IRUse* use)
                {
                    auto arrayIndexImpl = as<ScalarizedArrayIndexValImpl>(val.impl);
                    auto user = use->getUser();
                    IRBuilder builder(user);
                    builder.setInsertBefore(user);
                    auto subscriptVal = getSubscriptVal(
                        &builder,
                        arrayIndexImpl->elementType,
                        arrayIndexImpl->arrayVal,
                        arrayIndexImpl->index);
                    builder.setInsertBefore(user);
                    auto materializedInner = materializeValue(&builder, subscriptVal);
                    if (user->getOp() == kIROp_Load)
                    {
                        user->replaceUsesWith(materializedInner);
                        user->removeAndDeallocate();
                    }
                    else
                    {
                        use->set(materializedInner);
                    }
                });
            break;
        }
    case ScalarizedVal::Flavor::tuple:
        {
            auto tupleVal = as<ScalarizedTupleValImpl>(val.impl);
            traverseUses(
                originalVal,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    switch (user->getOp())
                    {
                    case kIROp_FieldExtract:
                    case kIROp_FieldAddress:
                        {
                            auto fieldKey = user->getOperand(1);
                            ScalarizedVal fieldVal;
                            for (auto element : tupleVal->elements)
                            {
                                if (element.key == fieldKey)
                                {
                                    fieldVal = element.val;
                                    break;
                                }
                                if (auto tupleValType =
                                        as<ScalarizedTupleValImpl>(element.val.impl))
                                {
                                    for (auto tupleElement : tupleValType->elements)
                                    {
                                        if (tupleElement.key == fieldKey)
                                        {
                                            fieldVal = tupleElement.val;
                                            break;
                                        }
                                    }
                                }
                                if (fieldVal.flavor != ScalarizedVal::Flavor::none)
                                {
                                    break;
                                }
                            }
                            if (fieldVal.flavor != ScalarizedVal::Flavor::none)
                            {
                                tryReplaceUsesOfStageInput(context, fieldVal, user);
                            }
                        }
                        break;
                    case kIROp_Load:
                        {
                            IRBuilder builder(user);
                            builder.setInsertBefore(user);
                            auto materializedVal = materializeTupleValue(&builder, val);
                            user->replaceUsesWith(materializedVal);
                            user->removeAndDeallocate();
                        }
                        break;
                    }
                });
        }
        break;
    }
}

void legalizeEntryPointParameterForGLSL(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRFunc* func,
    IRParam* pp,
    IRVarLayout* paramLayout)
{
    auto builder = context->getBuilder();
    auto stage = context->getStage();

    // (JS): In the legalization process parameters are moved from the entry point.
    // So when we get to emit we have a problem in that we can't use parameters to find important
    // decorations And in the future we will not have front end 'Layout' available. To work around
    // this, we take the decorations that need special handling from parameters and put them on the
    // IRFunc.
    //
    // This is only appropriate of course if there is only one of each for all parameters...
    // which is what current emit code assumes, but may not be more generally applicable.
    if (auto geomDecor = pp->findDecoration<IRGeometryInputPrimitiveTypeDecoration>())
    {
        if (!func->findDecoration<IRGeometryInputPrimitiveTypeDecoration>())
        {
            builder->addDecoration(func, geomDecor->getOp());
        }
        else
        {
            SLANG_UNEXPECTED("Only expected a single parameter to have "
                             "IRGeometryInputPrimitiveTypeDecoration decoration");
        }
    }

    if (stage == Stage::Geometry)
    {
        // If the user provided no parameters with a input primitive type qualifier, we
        // default to `triangle`.
        if (!func->findDecoration<IRGeometryInputPrimitiveTypeDecoration>())
        {
            builder->addDecoration(func, kIROp_TriangleInputPrimitiveTypeDecoration);
        }
    }

    // There *can* be multiple streamout parameters, to an entry point (points if nothing else)
    {
        IRType* type = pp->getFullType();
        // Strip out type
        if (auto outType = as<IROutTypeBase>(type))
        {
            type = outType->getValueType();
        }

        if (auto streamType = as<IRHLSLStreamOutputType>(type))
        {
            if ([[maybe_unused]] auto decor = func->findDecoration<IRStreamOutputTypeDecoration>())
            {
                // If it has the same stream out type, we *may* be ok (might not work for all types
                // of streams)
                SLANG_ASSERT(decor->getStreamType()->getOp() == streamType->getOp());
            }
            else
            {
                builder->addDecoration(func, kIROp_StreamOutputTypeDecoration, streamType);
            }
        }
    }

    // Lift Mesh Output decorations to the function
    // TODO: Ellie, check for duplication and assert consistency
    if (stage == Stage::Mesh)
    {
        if (auto d = pp->findDecoration<IRMeshOutputDecoration>())
        {
            // It's illegal to have differently sized indices and primitives
            // outputs, for consistency, only attach a PrimitivesDecoration to
            // the function.
            auto op = as<IRIndicesDecoration>(d) ? kIROp_PrimitivesDecoration : d->getOp();
            builder->addDecoration(func, op, d->getMaxSize());
        }
    }

    // We need to create a global variable that will replace the parameter.
    // It seems superficially obvious that the variable should have
    // the same type as the parameter.
    // However, if the parameter was a pointer, in order to
    // support `out` or `in out` parameter passing, we need
    // to be sure to allocate a variable of the pointed-to
    // type instead.
    //
    // We also need to replace uses of the parameter with
    // uses of the variable, and the exact logic there
    // will differ a bit between the pointer and non-pointer
    // cases.
    auto paramType = pp->getDataType();
    auto valueType = paramType;
    // First we will special-case stage input/outputs that
    // don't fit into the standard varying model.
    // - Geometry shader output streams
    // - Mesh shader outputs
    // - Mesh shader payload input
    if (auto paramPtrType = as<IRPtrTypeBase>(paramType))
    {
        valueType = paramPtrType->getValueType();
    }
    if (const auto gsStreamType = as<IRHLSLStreamOutputType>(valueType))
    {
        // An output stream type like `TriangleStream<Foo>` should
        // more or less translate into `out Foo` (plus scalarization).

        auto globalOutputVal = createGLSLGlobalVaryings(
            context,
            codeGenContext,
            builder,
            valueType,
            paramLayout,
            LayoutResourceKind::VaryingOutput,
            stage,
            pp);

        // A GS output stream might be passed into other
        // functions, so that we should really be modifying
        // any function that has one of these in its parameter
        // list.
        //
        HashSet<IRInst*> workListSet;
        List<IRFunc*> workList;
        workList.add(func);
        workListSet.add(func);
        for (Index i = 0; i < workList.getCount(); i++)
        {
            auto f = workList[i];
            for (auto bb = f->getFirstBlock(); bb; bb = bb->getNextBlock())
            {
                for (auto ii = bb->getFirstInst(); ii; ii = ii->getNextInst())
                {
                    // Is it a call?
                    if (ii->getOp() != kIROp_Call)
                        continue;

                    // Is it calling the append operation?
                    auto callee = getResolvedInstForDecorations(ii->getOperand(0));
                    if (callee->getOp() != kIROp_Func)
                        continue;

                    if (getBuiltinFuncName(callee) !=
                        UnownedStringSlice::fromLiteral("GeometryStreamAppend"))
                    {
                        // If we are calling a function that takes a output stream as a parameter,
                        // we need to add it to the work list to be processed.
                        for (UInt a = 1; a < ii->getOperandCount(); a++)
                        {
                            auto argType = ii->getOperand(a)->getDataType();
                            if (auto ptrTypeBase = as<IRPtrTypeBase>(argType))
                                argType = ptrTypeBase->getValueType();
                            if (as<IRHLSLStreamOutputType>(argType))
                            {
                                if (workListSet.add(callee))
                                    workList.add(as<IRFunc>(callee));
                                break;
                            }
                        }
                        continue;
                    }

                    // Okay, we have a declaration, and we want to modify it!

                    builder->setInsertBefore(ii);

                    assign(builder, globalOutputVal, ScalarizedVal::value(ii->getOperand(2)));
                }
            }
        }

        // We will still have references to the parameter coming
        // from the `EmitVertex` calls, so we need to replace it
        // with something. There isn't anything reasonable to
        // replace it with that would have the right type, so
        // we will replace it with an undefined value, knowing
        // that the emitted code will not actually reference it.
        //
        // TODO: This approach to generating geometry shader code
        // is not ideal, and we should strive to find a better
        // approach that involes coding the `EmitVertex` operation
        // directly in the core module, similar to how ray-tracing
        // operations like `TraceRay` are handled.
        //
        builder->setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
        auto undefinedVal = builder->emitUndefined(pp->getFullType());
        pp->replaceUsesWith(undefinedVal);

        return;
    }
    if (auto meshOutputType = as<IRMeshOutputType>(valueType))
    {
        return legalizeMeshOutputParam(
            context,
            codeGenContext,
            func,
            pp,
            paramLayout,
            meshOutputType);
    }
    if (auto patchType = as<IRHLSLPatchType>(valueType))
    {
        return legalizePatchParam(context, codeGenContext, func, pp, paramLayout, patchType);
    }
    if (pp->findDecoration<IRHLSLMeshPayloadDecoration>())
    {
        return legalizeMeshPayloadInputParam(context, codeGenContext, pp);
    }

    // When we have an HLSL ray tracing shader entry point,
    // we don't want to translate the inputs/outputs for GLSL/SPIR-V
    // according to our default rules, for two reasons:
    //
    // 1. The input and output for these stages are expected to
    // be packaged into `struct` types rather than be scalarized,
    // so the usual scalarization approach we take here should
    // not be applied.
    //
    // 2. An `in out` parameter isn't just sugar for a combination
    // of an `in` and an `out` parameter, and instead represents the
    // read/write "payload" that was passed in. It should legalize
    // to a single variable, and we can lower reads/writes of it
    // directly, rather than introduce an intermediate temporary.
    //
    switch (stage)
    {
    default:
        break;

    case Stage::AnyHit:
    case Stage::Callable:
    case Stage::ClosestHit:
    case Stage::Intersection:
    case Stage::Miss:
    case Stage::RayGeneration:
        return;
    }

    // Is the parameter type a special pointer type
    // that indicates the parameter is used for `out`
    // or `inout` access?
    if (as<IROutTypeBase>(paramType))
    {
        // Okay, we have the more interesting case here,
        // where the parameter was being passed by reference.
        // We are going to create a local variable of the appropriate
        // type, which will replace the parameter, along with
        // one or more global variables for the actual input/output.
        setInsertAfterOrdinaryInst(builder, pp);
        auto localVariable = builder->emitVar(valueType);
        auto localVal = ScalarizedVal::address(localVariable);

        if (const auto inOutType = as<IRInOutType>(paramType))
        {
            // In the `in out` case we need to declare two
            // sets of global variables: one for the `in`
            // side and one for the `out` side.
            auto globalInputVal = createGLSLGlobalVaryings(
                context,
                codeGenContext,
                builder,
                valueType,
                paramLayout,
                LayoutResourceKind::VaryingInput,
                stage,
                pp);

            assign(builder, localVal, globalInputVal);
        }

        // Any places where the original parameter was used inside
        // the function body should instead use the new local variable.
        // Since the parameter was a pointer, we use the variable instruction
        // itself (which is an `alloca`d pointer) directly:
        pp->replaceUsesWith(localVariable);

        // We also need one or more global variables to write the output to
        // when the function is done. We create them here.
        auto globalOutputVal = createGLSLGlobalVaryings(
            context,
            codeGenContext,
            builder,
            valueType,
            paramLayout,
            LayoutResourceKind::VaryingOutput,
            stage,
            pp);

        // Now we need to iterate over all the blocks in the function looking
        // for any `return*` instructions, so that we can write to the output variable
        for (auto bb = func->getFirstBlock(); bb; bb = bb->getNextBlock())
        {
            auto terminatorInst = bb->getLastInst();
            if (!terminatorInst)
                continue;

            switch (terminatorInst->getOp())
            {
            default:
                continue;

            case kIROp_Return:
                break;
            }

            // We dont' re-use `builder` here because we don't want to
            // disrupt the source location it is using for inserting
            // temporary variables at the top of the function.
            //
            IRBuilder terminatorBuilder(func);
            terminatorBuilder.setInsertBefore(terminatorInst);

            // Assign from the local variabel to the global output
            // variable before the actual `return` takes place.
            assign(&terminatorBuilder, globalOutputVal, localVal);
        }
    }
    else if (auto ptrType = as<IRPtrTypeBase>(paramType))
    {
        // This is the case where the parameter is passed by const
        // reference. We simply replace existing uses of the parameter
        // with the real global variable.
        SLANG_ASSERT(
            ptrType->getOp() == kIROp_ConstRefType ||
            ptrType->getAddressSpace() == AddressSpace::Input ||
            ptrType->getAddressSpace() == AddressSpace::BuiltinInput);

        auto globalValue = createGLSLGlobalVaryings(
            context,
            codeGenContext,
            builder,
            valueType,
            paramLayout,
            LayoutResourceKind::VaryingInput,
            stage,
            pp);
        tryReplaceUsesOfStageInput(context, globalValue, pp);
        for (auto dec : pp->getDecorations())
        {
            if (dec->getOp() != kIROp_GlobalVariableShadowingGlobalParameterDecoration)
                continue;
            auto globalVar = dec->getOperand(0);
            auto globalVarType = cast<IRPtrTypeBase>(globalVar->getDataType())->getValueType();
            if (as<IRStructType>(globalVarType))
            {
                tryReplaceUsesOfStageInput(context, globalValue, globalVar);
            }
            else
            {

                auto key = dec->getOperand(1);
                IRInst* realGlobalVar = nullptr;
                if (globalValue.flavor != ScalarizedVal::Flavor::tuple)
                    continue;
                if (auto tupleVal = as<ScalarizedTupleValImpl>(globalValue.impl))
                {
                    for (auto elem : tupleVal->elements)
                    {
                        if (elem.key == key)
                        {
                            realGlobalVar = elem.val.irValue;
                            if (!realGlobalVar &&
                                ScalarizedVal::Flavor::typeAdapter == elem.val.flavor)
                            {
                                if (auto typeAdapterVal =
                                        as<ScalarizedTypeAdapterValImpl>(elem.val.impl))
                                {
                                    realGlobalVar = typeAdapterVal->val.irValue;
                                }
                            }
                            break;
                        }
                    }
                }
                SLANG_ASSERT(realGlobalVar);

                // Remove all stores into the global var introduced during
                // the initial glsl global var translation pass since we are
                // going to replace the global var with a pointer to the real
                // input, and it makes no sense to store values into such real
                // input locations.
                traverseUses(
                    globalVar,
                    [&](IRUse* use)
                    {
                        auto user = use->getUser();
                        if (auto store = as<IRStore>(user))
                        {
                            if (store->getPtrUse() == use)
                            {
                                store->removeAndDeallocate();
                            }
                        }
                    });
                // we will be replacing uses of `globalVarToReplace`. We need
                // globalVarToReplaceNextUse to catch the next use before it is removed from the
                // list of uses.
                globalVar->replaceUsesWith(realGlobalVar);
                globalVar->removeAndDeallocate();
            }
        }
    }
    else
    {
        // This is the "easy" case where the parameter wasn't
        // being passed by reference. We start by just creating
        // one or more global variables to represent the parameter,
        // and attach the required layout information to it along
        // the way.

        auto globalValue = createGLSLGlobalVaryings(
            context,
            codeGenContext,
            builder,
            paramType,
            paramLayout,
            LayoutResourceKind::VaryingInput,
            stage,
            pp);

        tryReplaceUsesOfStageInput(context, globalValue, pp);

        // we have a simple struct which represents all materialized GlobalParams, this
        // struct will replace the no longer needed global variable which proxied as a
        // GlobalParam.
        IRInst* materialized = materializeValue(builder, globalValue);

        // We next need to replace all uses of the proxy variable with the actual GlobalParam
        pp->replaceUsesWith(materialized);

        // GlobalParams use use a OpStore to copy its data into a global
        // variable intermediary. We will follow the uses of this intermediary
        // and replace all some of the uses (function calls and SPIRV Operands)
        Dictionary<IRBlock*, IRInst*> blockToMaterialized;
        IRBuilder replaceBuilder(materialized);
        for (auto dec : pp->getDecorations())
        {
            if (dec->getOp() != kIROp_GlobalVariableShadowingGlobalParameterDecoration)
                continue;
            auto globalVar = dec->getOperand(0);
            auto globalVarType = cast<IRPtrTypeBase>(globalVar->getDataType())->getValueType();
            auto key = dec->getOperand(1);

            // we will be replacing uses of `globalVarToReplace`. We need globalVarToReplaceNextUse
            // to catch the next use before it is removed from the list of uses.
            IRUse* globalVarToReplaceNextUse;
            for (auto globalVarUse = globalVar->firstUse; globalVarUse;
                 globalVarUse = globalVarToReplaceNextUse)
            {
                globalVarToReplaceNextUse = globalVarUse->nextUse;
                auto user = globalVarUse->getUser();
                switch (user->getOp())
                {
                case kIROp_SPIRVAsmOperandInst:
                case kIROp_Call:
                    {
                        for (Slang::UInt operandIndex = 0; operandIndex < user->getOperandCount();
                             operandIndex++)
                        {
                            auto operand = user->getOperand(operandIndex);
                            auto operandUse = user->getOperands() + operandIndex;
                            if (operand != globalVar)
                                continue;

                            // a GlobalParam may be used across functions/blocks, we need to
                            // materialize at a minimum 1 struct per block.
                            auto callingBlock = getBlock(user);
                            bool found =
                                blockToMaterialized.tryGetValue(callingBlock, materialized);
                            if (!found)
                            {
                                replaceBuilder.setInsertBefore(callingBlock->getFirstInst());
                                materialized = materializeValue(&replaceBuilder, globalValue);
                                blockToMaterialized.set(callingBlock, materialized);
                            }

                            replaceBuilder.setInsertBefore(user);
                            auto field =
                                replaceBuilder.emitFieldExtract(globalVarType, materialized, key);
                            replaceBuilder.replaceOperand(operandUse, field);
                            break;
                        }
                        break;
                    }
                default:
                    break;
                }
                continue;
            }
        }
    }
}

bool shouldUseOriginalEntryPointName(CodeGenContext* codeGenContext)
{
    if (auto hlslOptions = codeGenContext->getTargetProgram()->getHLSLToVulkanLayoutOptions())
    {
        if (hlslOptions->getUseOriginalEntryPointName())
        {
            return true;
        }
    }
    return false;
}

void getAllNullLocationRayObjectsAndUsedLocations(
    IRModule* module,
    List<IRInst*>* nullRayObjects,
    HashSet<IRIntegerValue>* rayPayload,
    HashSet<IRIntegerValue>* callablePayload,
    HashSet<IRIntegerValue>* hitObjectAttribute)
{
    for (auto inst : module->getGlobalInsts())
    {
        auto instOp = inst->getOp();
        IRIntegerValue intLitVal = 0;
        if (instOp != kIROp_GlobalParam && instOp != kIROp_GlobalVar)
            continue;
        for (auto decor : inst->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_VulkanRayPayloadDecoration:
            case kIROp_VulkanRayPayloadInDecoration:
                intLitVal = as<IRIntLit>(decor->getOperand(0))->getValue();
                if (intLitVal == -1)
                {
                    nullRayObjects->add(inst);
                    goto getAllNullLocationRayObjectsAndUsedLocations_end;
                }
                rayPayload->add(intLitVal);
                goto getAllNullLocationRayObjectsAndUsedLocations_end;
            case kIROp_VulkanCallablePayloadDecoration:
            case kIROp_VulkanCallablePayloadInDecoration:
                intLitVal = as<IRIntLit>(decor->getOperand(0))->getValue();
                if (intLitVal == -1)
                {
                    nullRayObjects->add(inst);
                    goto getAllNullLocationRayObjectsAndUsedLocations_end;
                }
                callablePayload->add(intLitVal);
                goto getAllNullLocationRayObjectsAndUsedLocations_end;
            case kIROp_VulkanHitObjectAttributesDecoration:
                intLitVal = as<IRIntLit>(decor->getOperand(0))->getValue();
                if (intLitVal == -1)
                {
                    nullRayObjects->add(inst);
                    goto getAllNullLocationRayObjectsAndUsedLocations_end;
                }
                hitObjectAttribute->add(intLitVal);
                goto getAllNullLocationRayObjectsAndUsedLocations_end;
            }
        }
    getAllNullLocationRayObjectsAndUsedLocations_end:;
    }
}
void assignRayPayloadHitObjectAttributeLocations(IRModule* module)
{
    List<IRInst*> nullRayObjects;
    HashSet<IRIntegerValue> rayPayloadLocations;
    HashSet<IRIntegerValue> callablePayloadLocations;
    HashSet<IRIntegerValue> hitObjectAttributeLocations;
    getAllNullLocationRayObjectsAndUsedLocations(
        module,
        &nullRayObjects,
        &rayPayloadLocations,
        &callablePayloadLocations,
        &hitObjectAttributeLocations);

    IRIntegerValue rayPayloadCounter = 0;
    IRIntegerValue callablePayloadCounter = 0;
    IRIntegerValue hitObjectAttributeCounter = 0;

    IRBuilder builder(module);
    for (auto inst : nullRayObjects)
    {
        IRInst* location = nullptr;
        IRIntegerValue intLitVal = 0;
        for (auto decor : inst->getDecorations())
        {
            switch (decor->getOp())
            {
            case kIROp_VulkanRayPayloadDecoration:
            case kIROp_VulkanRayPayloadInDecoration:
                intLitVal = as<IRIntLit>(decor->getOperand(0))->getValue();
                if (intLitVal >= 0)
                    goto assignRayPayloadHitObjectAttributeLocations_end;
                while (rayPayloadLocations.contains(rayPayloadCounter))
                {
                    rayPayloadCounter++;
                }
                builder.setInsertBefore(inst);
                location = builder.getIntValue(builder.getIntType(), rayPayloadCounter);
                decor->setOperand(0, location);
                rayPayloadCounter++;
                goto assignRayPayloadHitObjectAttributeLocations_end;
            case kIROp_VulkanCallablePayloadDecoration:
            case kIROp_VulkanCallablePayloadInDecoration:
                intLitVal = as<IRIntLit>(decor->getOperand(0))->getValue();
                if (intLitVal >= 0)
                    goto assignRayPayloadHitObjectAttributeLocations_end;
                while (callablePayloadLocations.contains(callablePayloadCounter))
                {
                    callablePayloadCounter++;
                }
                builder.setInsertBefore(inst);
                location = builder.getIntValue(builder.getIntType(), callablePayloadCounter);
                decor->setOperand(0, location);
                callablePayloadCounter++;
                goto assignRayPayloadHitObjectAttributeLocations_end;
            case kIROp_VulkanHitObjectAttributesDecoration:
                intLitVal = as<IRIntLit>(decor->getOperand(0))->getValue();
                if (intLitVal >= 0)
                    goto assignRayPayloadHitObjectAttributeLocations_end;
                while (hitObjectAttributeLocations.contains(hitObjectAttributeCounter))
                {
                    hitObjectAttributeCounter++;
                }
                builder.setInsertBefore(inst);
                location = builder.getIntValue(builder.getIntType(), hitObjectAttributeCounter);
                decor->setOperand(0, location);
                hitObjectAttributeCounter++;
                goto assignRayPayloadHitObjectAttributeLocations_end;
            default:
                break;
            }
        }
    assignRayPayloadHitObjectAttributeLocations_end:;
    }
}

void rewriteReturnToOutputStore(IRBuilder& builder, IRFunc* func, ScalarizedVal resultGlobal)
{
    for (auto bb = func->getFirstBlock(); bb; bb = bb->getNextBlock())
    {
        auto returnInst = as<IRReturn>(bb->getTerminator());
        if (!returnInst)
            continue;

        IRInst* returnValue = returnInst->getVal();

        // Make sure we add these instructions to the right block
        builder.setInsertInto(bb);

        // Write to our global variable(s) from the value being returned.
        assign(&builder, resultGlobal, ScalarizedVal::value(returnValue));

        // Emit a `return void_val` to end the block
        builder.emitReturn();

        // Remove the old `returnVal` instruction.
        returnInst->removeAndDeallocate();
    }
}

ScalarizedVal legalizeEntryPointReturnValueForGLSL(
    GLSLLegalizationContext* context,
    CodeGenContext* codeGenContext,
    IRBuilder& builder,
    IRFunc* func,
    IRVarLayout* resultLayout)
{
    ScalarizedVal result;
    auto resultType = func->getResultType();
    if (as<IRVoidType>(resultType))
    {
        // In this case, the function doesn't return a value
        // so we don't need to transform its `return` sites.
        //
        // We can also use this opportunity to quickly
        // check if the function has any parameters, and if
        // it doesn't use the chance to bail out immediately.
        if (func->getParamCount() == 0)
        {
            // This function is already legal for GLSL
            // (at least in terms of parameter/result signature),
            // so we won't bother doing anything at all.
            return result;
        }

        // If the function does have parameters, then we need
        // to let the logic later in this function handle them.
    }
    else
    {
        // Function returns a value, so we need
        // to introduce a new global variable
        // to hold that value, and then replace
        // any `returnVal` instructions with
        // code to write to that variable.

        ScalarizedVal resultGlobal = createGLSLGlobalVaryings(
            context,
            codeGenContext,
            &builder,
            resultType,
            resultLayout,
            LayoutResourceKind::VaryingOutput,
            context->stage,
            func);
        result = resultGlobal;

        if (auto entryPointDecor = func->findDecoration<IREntryPointDecoration>())
        {
            if (entryPointDecor->getProfile().getStage() == Stage::Hull)
            {
                builder.setInsertBefore(func->getFirstBlock()->getFirstOrdinaryInst());
                auto index = getOrCreateBuiltinParamForHullShader(
                    context,
                    toSlice("SV_OutputControlPointID"));
                resultGlobal = getSubscriptVal(&builder, resultType, resultGlobal, index);
            }
        }
        rewriteReturnToOutputStore(builder, func, resultGlobal);
    }
    return result;
}

void legalizeTargetBuiltinVar(GLSLLegalizationContext& context)
{
    List<KeyValuePair<IRTargetBuiltinVarName, IRInst*>> workItems;
    for (auto [builtinVarName, varInst] : context.builtinVarMap)
    {
        if (builtinVarName == IRTargetBuiltinVarName::HlslInstanceID)
        {
            workItems.add(KeyValuePair(builtinVarName, varInst));
        }
    }

    auto getOrCreateBuiltinVar = [&](IRTargetBuiltinVarName name, IRType* type)
    {
        if (auto var = context.builtinVarMap.tryGetValue(name))
            return *var;
        IRBuilder builder(context.entryPointFunc);
        builder.setInsertBefore(context.entryPointFunc);
        IRInst* var = builder.createGlobalParam(type);
        builder.addTargetBuiltinVarDecoration(var, name);
        return var;
    };
    for (auto& kv : workItems)
    {
        auto builtinVarName = kv.key;
        auto varInst = kv.value;

        // Repalce SV_InstanceID with gl_InstanceIndex - gl_BaseInstance.
        if (builtinVarName == IRTargetBuiltinVarName::HlslInstanceID)
        {
            auto instanceIndex = getOrCreateBuiltinVar(
                IRTargetBuiltinVarName::SpvInstanceIndex,
                varInst->getDataType());
            auto baseInstance = getOrCreateBuiltinVar(
                IRTargetBuiltinVarName::SpvBaseInstance,
                varInst->getDataType());
            traverseUses(
                varInst,
                [&](IRUse* use)
                {
                    auto user = use->getUser();
                    if (user->getOp() == kIROp_Load)
                    {
                        IRBuilder builder(use->getUser());
                        builder.setInsertBefore(use->getUser());
                        auto sub = builder.emitSub(
                            tryGetPointedToType(&builder, varInst->getDataType()),
                            builder.emitLoad(instanceIndex),
                            builder.emitLoad(baseInstance));
                        user->replaceUsesWith(sub);
                    }
                });
        }
    }
}

void legalizeEntryPointForGLSL(
    Session* session,
    IRModule* module,
    IRFunc* func,
    CodeGenContext* codeGenContext,
    ShaderExtensionTracker* glslExtensionTracker)
{
    auto entryPointDecor = func->findDecoration<IREntryPointDecoration>();
    SLANG_ASSERT(entryPointDecor);

    auto stage = entryPointDecor->getProfile().getStage();

    auto layoutDecoration = func->findDecoration<IRLayoutDecoration>();
    SLANG_ASSERT(layoutDecoration);

    auto entryPointLayout = as<IREntryPointLayout>(layoutDecoration->getLayout());
    SLANG_ASSERT(entryPointLayout);


    GLSLLegalizationContext context;
    context.session = session;
    context.stage = stage;
    context.entryPointFunc = func;
    context.sink = codeGenContext->getSink();
    context.glslExtensionTracker = glslExtensionTracker;

    // We require that the entry-point function has no calls,
    // because otherwise we'd invalidate the signature
    // at all existing call sites.
    //
    // TODO: the right thing to do here is to split any
    // function that both gets called as an entry point
    // and as an ordinary function.
    for (auto use = func->firstUse; use; use = use->nextUse)
        SLANG_ASSERT(use->getUser()->getOp() != kIROp_Call);

    // Require SPIRV version based on the stage.
    switch (stage)
    {
    case Stage::Mesh:
    case Stage::Amplification:
        glslExtensionTracker->requireSPIRVVersion(SemanticVersion(1, 4, 0));
        break;
    case Stage::AnyHit:
    case Stage::Callable:
    case Stage::Miss:
    case Stage::RayGeneration:
    case Stage::Intersection:
    case Stage::ClosestHit:
        glslExtensionTracker->requireSPIRVVersion(SemanticVersion(1, 4, 0));
        break;
    }

    // For hull shaders, we need to convert it to single return form, because
    // we need to insert a barrier after the main body, then invoke the
    // patch constant function after the barrier.
    if (stage == Stage::Hull)
    {
        convertFuncToSingleReturnForm(module, func);
    }

    // We create a dummy IR builder, since some of
    // the functions require it.
    //
    // TODO: make some of these free functions...
    //
    IRBuilder builder(module);
    builder.setInsertInto(func);

    context.builder = &builder;

    // Rename the entrypoint to "main" to conform to GLSL standard,
    // if the compile options require us to do it.
    if (!shouldUseOriginalEntryPointName(codeGenContext) &&
        codeGenContext->getEntryPointCount() == 1)
    {
        entryPointDecor->setName(builder.getStringValue(UnownedStringSlice("main")));
    }

    // We will start by looking at the return type of the
    // function, because that will enable us to do an
    // early-out check to avoid more work.
    //
    // Specifically, we need to check if the function has
    // a `void` return type, because there is no work
    // to be done on its return value in that case.
    auto scalarizedGlobalOutput = legalizeEntryPointReturnValueForGLSL(
        &context,
        codeGenContext,
        builder,
        func,
        entryPointLayout->getResultLayout());

    // For hull shaders, insert the invocation of the patch constant function
    // at the end of the entrypoint now.
    if (stage == Stage::Hull)
    {
        invokePathConstantFuncInHullShader(&context, codeGenContext, scalarizedGlobalOutput);
    }

    // Special handling for ray tracing shaders
    bool isRayTracingShader = false;
    switch (stage)
    {
    case Stage::AnyHit:
    case Stage::Callable:
    case Stage::ClosestHit:
    case Stage::Intersection:
    case Stage::Miss:
    case Stage::RayGeneration:
        isRayTracingShader = true;
        consolidateRayTracingParameters(&context, func);
        break;
    default:
        break;
    }

    // Next we will walk through any parameters of the entry-point function,
    // and turn them into global variables.
    if (auto firstBlock = func->getFirstBlock())
    {
        for (auto pp = firstBlock->getFirstParam(); pp; pp = pp->getNextParam())
        {
            if (isRayTracingShader)
            {
                continue;
            }
            // Any initialization code we insert for parameters needs
            // to be at the start of the "ordinary" instructions in the block:
            builder.setInsertBefore(firstBlock->getFirstOrdinaryInst());

            // We assume that the entry-point parameters will all have
            // layout information attached to them, which is kept up-to-date
            // by any transformations affecting the parameter list.
            //
            auto paramLayoutDecoration = pp->findDecoration<IRLayoutDecoration>();
            SLANG_ASSERT(paramLayoutDecoration);
            auto paramLayout = as<IRVarLayout>(paramLayoutDecoration->getLayout());
            SLANG_ASSERT(paramLayout);

            legalizeEntryPointParameterForGLSL(&context, codeGenContext, func, pp, paramLayout);
        }

        // At this point we should have eliminated all uses of the
        // parameters of the entry block. Also, our control-flow
        // rules mean that the entry block cannot be the target
        // of any branches in the code, so there can't be
        // any control-flow ops that try to match the parameter
        // list.
        //
        // We can safely go through and destroy the parameters
        // themselves, and then clear out the parameter list.

        for (auto pp = firstBlock->getFirstParam(); pp;)
        {
            auto next = pp->getNextParam();
            pp->removeAndDeallocate();
            pp = next;
        }
    }

    // Finally, we need to patch up the type of the entry point,
    // because it is no longer accurate.

    IRFuncType* voidFuncType = builder.getFuncType(0, nullptr, builder.getVoidType());
    func->setFullType(voidFuncType);

    // TODO: we should technically be constructing
    // a new `EntryPointLayout` here to reflect
    // the way that things have been moved around.

    // Let's fix the size array type globals now that we know the maximum index
    {
        for (const auto& [_, value] : context.systemNameToGlobalMap)
        {
            auto type = value.globalParam->getDataType();

            // Strip ptr if there is one.
            auto ptrType = as<IRPtrTypeBase>(type);
            if (ptrType)
            {
                type = ptrType->getValueType();
            }

            // Get the array type
            auto arrayType = as<IRArrayType>(type);
            if (!arrayType)
            {
                continue;
            }

            // Get the element type
            auto elementType = arrayType->getElementType();

            // Create an new array type
            auto elementCountInst = builder.getIntValue(builder.getIntType(), value.maxIndex + 1);
            IRType* sizedArrayType = builder.getArrayType(elementType, elementCountInst);

            // Re-add ptr if there was one on the input
            if (ptrType)
            {
                sizedArrayType = builder.getPtrType(
                    ptrType->getOp(),
                    sizedArrayType,
                    ptrType->getAddressSpace());
            }

            // Change the globals type
            value.globalParam->setFullType(sizedArrayType);
        }
    }

    // Some system value vars can't be mapped 1:1 to a GLSL/Vulkan builtin,
    // for example, SV_InstanceID should map to gl_InstanceIndex - gl_BaseInstance,
    // we will replace these builtins with additional compute logic here.
    legalizeTargetBuiltinVar(context);
}

void decorateModuleWithSPIRVVersion(IRModule* module, SemanticVersion spirvVersion)
{
    CapabilityName atom = CapabilityName::_spirv_1_0;
    switch (spirvVersion.m_major)
    {
    case 1:
        {
            switch (spirvVersion.m_minor)
            {
            case 0:
                atom = CapabilityName::_spirv_1_0;
                break;
            case 1:
                atom = CapabilityName::_spirv_1_1;
                break;
            case 2:
                atom = CapabilityName::_spirv_1_2;
                break;
            case 3:
                atom = CapabilityName::_spirv_1_3;
                break;
            case 4:
                atom = CapabilityName::_spirv_1_4;
                break;
            case 5:
                atom = CapabilityName::_spirv_1_5;
                break;
            case 6:
                atom = CapabilityName::_spirv_1_6;
                break;
            default:
                SLANG_UNEXPECTED("Unknown SPIRV version");
            }
            break;
        }
    }
    IRBuilder builder(module);
    builder.addRequireCapabilityAtomDecoration(module->getModuleInst(), atom);
}

void legalizeEntryPointsForGLSL(
    Session* session,
    IRModule* module,
    const List<IRFunc*>& funcs,
    CodeGenContext* context,
    ShaderExtensionTracker* glslExtensionTracker)
{
    for (auto func : funcs)
    {
        legalizeEntryPointForGLSL(session, module, func, context, glslExtensionTracker);
    }

    assignRayPayloadHitObjectAttributeLocations(module);

    decorateModuleWithSPIRVVersion(module, glslExtensionTracker->getSPIRVVersion());
}

void legalizeConstantBufferLoadForGLSL(IRModule* module)
{
    // Constant buffers and parameter blocks are represented as `uniform` blocks
    // in GLSL. These uniform blocks can't be used directly as a value of the underlying
    // struct type. If we see a direct load of the constant buffer pointer,
    // we need to replace it with a `MakeStruct` inst where each field is separately
    // loaded.
    IRBuilder builder(module);
    for (auto globalInst : module->getGlobalInsts())
    {
        if (auto func = as<IRGlobalValueWithCode>(globalInst))
        {
            for (auto block : func->getBlocks())
            {
                for (auto inst = block->getFirstInst(); inst;)
                {
                    auto load = as<IRLoad>(inst);
                    inst = inst->next;
                    if (!load)
                        continue;
                    auto bufferType = load->getPtr()->getDataType();
                    if (as<IRConstantBufferType>(bufferType) ||
                        as<IRParameterBlockType>(bufferType))
                    {
                        auto parameterGroupType = as<IRUniformParameterGroupType>(bufferType);
                        auto elementType = as<IRStructType>(parameterGroupType->getElementType());
                        if (!elementType)
                            continue;
                        List<IRInst*> elements;
                        builder.setInsertBefore(load);
                        for (auto field : elementType->getFields())
                        {
                            auto fieldAddr = builder.emitFieldAddress(
                                builder.getPtrType(field->getFieldType()),
                                load->getPtr(),
                                field->getKey());
                            auto fieldValue = builder.emitLoad(field->getFieldType(), fieldAddr);
                            elements.add(fieldValue);
                        }
                        auto makeStruct = builder.emitMakeStruct(
                            elementType,
                            elements.getCount(),
                            elements.getBuffer());
                        load->replaceUsesWith(makeStruct);
                        load->removeAndDeallocate();
                    }
                }
            }
        }
    }
}


void legalizeDispatchMeshPayloadForGLSL(IRModule* module)
{
    // Find out DispatchMesh function
    IRGlobalValueWithCode* dispatchMeshFunc = nullptr;
    for (const auto globalInst : module->getGlobalInsts())
    {
        if (const auto func = as<IRGlobalValueWithCode>(globalInst))
        {
            if (const auto dec = func->findDecoration<IRKnownBuiltinDecoration>())
            {
                if (dec->getName() == "DispatchMesh")
                {
                    SLANG_ASSERT(!dispatchMeshFunc && "Multiple DispatchMesh functions found");
                    dispatchMeshFunc = func;
                }
            }
        }
    }

    if (!dispatchMeshFunc)
        return;

    IRBuilder builder{module};
    builder.setInsertBefore(dispatchMeshFunc);

    // We'll rewrite the calls to call EmitMeshTasksEXT
    traverseUses(
        dispatchMeshFunc,
        [&](const IRUse* use)
        {
            if (const auto call = as<IRCall>(use->getUser()))
            {
                SLANG_ASSERT(call->getArgCount() == 4);
                const auto payload = call->getArg(3);

                const auto payloadPtrType =
                    composeGetters<IRPtrTypeBase>(payload, &IRInst::getDataType);
                SLANG_ASSERT(payloadPtrType);
                const auto payloadType = payloadPtrType->getValueType();
                SLANG_ASSERT(payloadType);

                const bool isGroupsharedGlobal =
                    payload->getParent() == module->getModuleInst() &&
                    composeGetters<IRGroupSharedRate>(payload, &IRInst::getRate);
                if (isGroupsharedGlobal)
                {
                    // If it's a groupshared global, then we put it in the address
                    // space we know to emit as taskPayloadSharedEXT instead (or
                    // naturally fall through correctly for SPIR-V emit)
                    //
                    // Keep it as a groupshared rate qualified type so we don't
                    // miss out on any further legalization requirement or
                    // optimization opportunities.
                    const auto payloadSharedPtrType = builder.getRateQualifiedType(
                        builder.getGroupSharedRate(),
                        builder.getPtrType(
                            payloadPtrType->getOp(),
                            payloadPtrType->getValueType(),
                            AddressSpace::TaskPayloadWorkgroup));
                    payload->setFullType(payloadSharedPtrType);
                }
                else
                {
                    // ...
                    // If it's not a groupshared global, then create such a
                    // parameter and store into the value being passed to this
                    // call.
                    builder.setInsertInto(module->getModuleInst());
                    const auto v =
                        builder.createGlobalVar(payloadType, AddressSpace::TaskPayloadWorkgroup);
                    v->setFullType(builder.getRateQualifiedType(
                        builder.getGroupSharedRate(),
                        v->getFullType()));
                    builder.setInsertBefore(call);
                    builder.emitStore(v, builder.emitLoad(payload));

                    // Then, make sure that it's this new global which is being
                    // passed into the call to DispatchMesh, this is unimportant
                    // for GLSL which ignores such a parameter, but the SPIR-V
                    // backend depends on it being the global
                    call->getArgs()[3].set(v);
                }
            }
        });
}

void legalizeDynamicResourcesForGLSL(CodeGenContext* context, IRModule* module)
{
    List<IRInst*> toRemove;

    // At this stage, we can safely remove the generic `getDescriptorFromHandle` function
    // despite it being marked `export`.
    for (auto inst : module->getGlobalInsts())
    {
        if (auto genFunc = as<IRGeneric>(inst))
        {
            if (!genFunc->hasUses())
            {
                toRemove.add(genFunc);
            }
        }
    }
    for (auto inst : toRemove)
    {
        inst->removeAndDeallocate();
    }

    for (auto inst : module->getGlobalInsts())
    {
        auto param = as<IRGlobalParam>(inst);

        if (!param)
            continue;

        // We are only interested in parameters involving `DynamicResource`, or arrays of it.
        auto arrayType = as<IRArrayTypeBase>(param->getDataType());
        auto type = arrayType ? arrayType->getElementType() : param->getDataType();

        if (!as<IRDynamicResourceType>(type))
            continue;

        Dictionary<IRType*, IRGlobalParam*> aliasedParams;
        IRBuilder builder(module);

        auto getAliasedParam = [&](IRType* type)
        {
            IRGlobalParam* newParam;

            if (!aliasedParams.tryGetValue(type, newParam))
            {
                newParam = builder.createGlobalParam(type);

                for (auto decoration : param->getDecorations())
                    cloneDecoration(decoration, newParam);

                aliasedParams[type] = newParam;
            }
            return newParam;
        };

        // Try to rewrite all uses leading to `CastDynamicResource`.
        // Later, we will diagnose an error if the parameter still has uses.
        traverseUsers(
            param,
            [&](IRInst* user)
            {
                if (user->getOp() == kIROp_CastDynamicResource && !arrayType)
                {
                    builder.setInsertBefore(user);

                    user->replaceUsesWith(getAliasedParam(user->getDataType()));
                    user->removeAndDeallocate();
                }
                else if (user->getOp() == kIROp_GetElement && arrayType)
                {
                    traverseUsers(
                        user,
                        [&](IRInst* elementUser)
                        {
                            if (elementUser->getOp() == kIROp_CastDynamicResource)
                            {
                                builder.setInsertBefore(elementUser);

                                auto paramType = builder.getArrayTypeBase(
                                    arrayType->getOp(),
                                    elementUser->getDataType(),
                                    arrayType->getElementCount());

                                auto newAccess = builder.emitElementExtract(
                                    paramType->getElementType(),
                                    getAliasedParam(paramType),
                                    user->getOperand(1));

                                elementUser->replaceUsesWith(newAccess);
                                elementUser->removeAndDeallocate();
                            }
                        });

                    if (!user->hasUses())
                    {
                        user->removeAndDeallocate();
                    }
                }
            });
        toRemove.add(param);
    }

    // Remove unused parameters later to avoid invalidating iterator.
    for (auto param : toRemove)
    {
        if (!param->hasUses())
        {
            param->removeAndDeallocate();
        }
        else
        {
            context->getSink()->diagnose(
                param->firstUse->getUser(),
                Diagnostics::ambiguousReference,
                param);
        }
    }
}

} // namespace Slang
