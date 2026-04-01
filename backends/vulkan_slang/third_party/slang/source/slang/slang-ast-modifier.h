// slang-ast-modifier.h
#pragma once

#include "slang-ast-base.h"
#include "slang-ast-modifier.h.fiddle"

FIDDLE()
namespace Slang
{

// Syntax class definitions for modifiers.

// Simple modifiers have no state beyond their identity

FIDDLE()
class InModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class OutModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class ConstModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class BuiltinModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class InlineModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE(abstract)
class VisibilityModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class PublicModifier : public VisibilityModifier
{
    FIDDLE(...)
};

FIDDLE()
class PrivateModifier : public VisibilityModifier
{
    FIDDLE(...)
};

FIDDLE()
class InternalModifier : public VisibilityModifier
{
    FIDDLE(...)
};

FIDDLE()
class RequireModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class ParamModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class ExternModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLExportModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class TransparentModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class FromCoreModuleModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class PrefixModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class PostfixModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class ExportedModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class ConstExprModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class ExternCppModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLPrecisionModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLModuleModifier : public Modifier
{
    FIDDLE(...)
};

// Marks that the definition of a decl is not yet synthesized.
FIDDLE()
class ToBeSynthesizedModifier : public Modifier
{
    FIDDLE(...)
};

// Marks that the definition of a decl is synthesized.
FIDDLE()
class SynthesizedModifier : public Modifier
{
    FIDDLE(...)
};

// Marks a synthesized variable as local temporary variable.
FIDDLE()
class LocalTempVarModifier : public Modifier
{
    FIDDLE(...)
};

// An `extern` variable in an extension is used to introduce additional attributes on an existing
// field.
FIDDLE()
class ExtensionExternVarModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() DeclRef<Decl> originalDecl;
};

// An 'ActualGlobal' is a global that is output as a normal global in CPU code.
// Globals in HLSL/Slang are constant state passed into kernel execution
FIDDLE()
class ActualGlobalModifier : public Modifier
{
    FIDDLE(...)
};

/// A modifier that indicates an `InheritanceDecl` should be ignored during name lookup (and related
/// checks).
FIDDLE()
class IgnoreForLookupModifier : public Modifier
{
    FIDDLE(...)
};

// A modifier that marks something as an operation that
// has a one-to-one translation to the IR, and thus
// has no direct definition in the high-level language.
//
FIDDLE()
class IntrinsicOpModifier : public Modifier
{
    FIDDLE(...)
    // Token that names the intrinsic op.
    Token opToken;

    // The IR opcode for the intrinsic operation.
    //
    FIDDLE() uint32_t op = 0;
};

// A modifier that marks something as an intrinsic function,
// for some subset of targets.
FIDDLE()
class TargetIntrinsicModifier : public Modifier
{
    FIDDLE(...)
    // Token that names the target that the operation
    // is an intrisic for.
    FIDDLE() Token targetToken;

    // A custom definition for the operation, one of either an ident or a
    // string (the concatenation of several string literals)
    Token definitionIdent;
    FIDDLE() String definitionString;
    bool isString;

    // A predicate to be used on an identifier to guard this intrinsic
    Token predicateToken;
    NameLoc scrutinee;
    FIDDLE() DeclRef<Decl> scrutineeDeclRef;
};

// A modifier that marks a declaration as representing a
// specialization that should be preferred on a particular
// target.
FIDDLE()
class SpecializedForTargetModifier : public Modifier
{
    FIDDLE(...)
    // Token that names the target that the operation
    // has been specialized for.
    FIDDLE() Token targetToken;
};

// A modifier to tag something as an intrinsic that requires
// a certain GLSL extension to be enabled when used
FIDDLE()
class RequiredGLSLExtensionModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() Token extensionNameToken;
};

// A modifier to tag something as an intrinsic that requires
// a certain GLSL version to be enabled when used
FIDDLE()
class RequiredGLSLVersionModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() Token versionNumberToken;
};


// A modifier to tag something as an intrinsic that requires
// a certain SPIRV version to be enabled when used. Specified as "major.minor"
FIDDLE()
class RequiredSPIRVVersionModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() SemanticVersion version;
};

// A modifier to tag something as an intrinsic that requires
// a certain WGSL extension to be enabled when used
FIDDLE()
class RequiredWGSLExtensionModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() Token extensionNameToken;
};

// A modifier to tag something as an intrinsic that requires
// a certain CUDA SM version to be enabled when used. Specified as "major.minor"
FIDDLE()
class RequiredCUDASMVersionModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() SemanticVersion version;
};

FIDDLE()
class InOutModifier : public OutModifier
{
    FIDDLE(...)
};


// `__ref` modifier for by-reference parameter passing
FIDDLE()
class RefModifier : public Modifier
{
    FIDDLE(...)
};

// `__ref` modifier for by-reference parameter passing
FIDDLE()
class ConstRefModifier : public Modifier
{
    FIDDLE(...)
};

// This is a special sentinel modifier that gets added
// to the list when we have multiple variable declarations
// all sharing the same modifiers:
//
//     static uniform int a : FOO, *b : register(x0);
//
// In this case both `a` and `b` share the syntax
// for part of their modifier list, but then have
// their own modifiers as well:
//
//     a: SemanticModifier("FOO") --> SharedModifiers --> StaticModifier --> UniformModifier
//                                 /
//     b: RegisterModifier("x0")  /
//
FIDDLE()
class SharedModifiers : public Modifier
{
    FIDDLE(...)
};

// AST nodes to represent the begin/end of a `layout` modifier group
FIDDLE(abstract)
class GLSLLayoutModifierGroupMarker : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLLayoutModifierGroupBegin : public GLSLLayoutModifierGroupMarker
{
    FIDDLE(...)
};

FIDDLE()
class GLSLLayoutModifierGroupEnd : public GLSLLayoutModifierGroupMarker
{
    FIDDLE(...)
};

FIDDLE()
class GLSLUnparsedLayoutModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLBufferDataLayoutModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLStd140Modifier : public GLSLBufferDataLayoutModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLStd430Modifier : public GLSLBufferDataLayoutModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLScalarModifier : public GLSLBufferDataLayoutModifier
{
    FIDDLE(...)
};


// A catch-all for single-keyword modifiers
FIDDLE()
class SimpleModifier : public Modifier
{
    FIDDLE(...)
};


// Indicates that this is a variable declaration that corresponds to
// a parameter block declaration in the source program.
FIDDLE()
class ImplicitParameterGroupVariableModifier : public Modifier
{
    FIDDLE(...)
};


// Indicates that this is a type that corresponds to the element
// type of a parameter block declaration in the source program.
FIDDLE()
class ImplicitParameterGroupElementTypeModifier : public Modifier
{
    FIDDLE(...)
};


// An HLSL semantic
FIDDLE(abstract)
class HLSLSemantic : public Modifier
{
    FIDDLE(...)
    FIDDLE() Token name;
};

// An HLSL semantic that affects layout
FIDDLE()
class HLSLLayoutSemantic : public HLSLSemantic
{
    FIDDLE(...)
    FIDDLE() Token registerName;
    FIDDLE() Token componentMask;
};

// An HLSL `register` semantic
FIDDLE()
class HLSLRegisterSemantic : public HLSLLayoutSemantic
{
    FIDDLE(...)
    FIDDLE() Token spaceName;
};

// TODO(tfoley): `packoffset`
FIDDLE()
class HLSLPackOffsetSemantic : public HLSLLayoutSemantic
{
    FIDDLE(...)
    FIDDLE() int uniformOffset = 0;
};


// An HLSL semantic that just associated a declaration with a semantic name
FIDDLE()
class HLSLSimpleSemantic : public HLSLSemantic
{
    FIDDLE(...)
};

// A semantic applied to a field of a ray-payload type, to control access
FIDDLE()
class RayPayloadAccessSemantic : public HLSLSemantic
{
    FIDDLE(...)
    FIDDLE() List<Token> stageNameTokens;
};

FIDDLE()
class RayPayloadReadSemantic : public RayPayloadAccessSemantic
{
    FIDDLE(...)
};

FIDDLE()
class RayPayloadWriteSemantic : public RayPayloadAccessSemantic
{
    FIDDLE(...)
};


// GLSL

// Directives that came in via the preprocessor, but
// that we need to keep around for later steps
FIDDLE()
class GLSLPreprocessorDirective : public Modifier
{
    FIDDLE(...)
};


// A GLSL `#version` directive
FIDDLE()
class GLSLVersionDirective : public GLSLPreprocessorDirective
{
    FIDDLE(...)
    // Token giving the version number to use
    FIDDLE() Token versionNumberToken;

    // Optional token giving the sub-profile to be used
    FIDDLE() Token glslProfileToken;
};

// A GLSL `#extension` directive
FIDDLE()
class GLSLExtensionDirective : public GLSLPreprocessorDirective
{
    FIDDLE(...)
    // Token giving the version number to use
    FIDDLE() Token extensionNameToken;

    // Optional token giving the sub-profile to be used
    FIDDLE() Token dispositionToken;
};

FIDDLE()
class ParameterGroupReflectionName : public Modifier
{
    FIDDLE(...)
    FIDDLE() NameLoc nameAndLoc;
};

// A modifier that indicates a built-in base type (e.g., `float`)
FIDDLE()
class BuiltinTypeModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() BaseType tag;
};

// A modifier that indicates a built-in type that isn't a base type (e.g., `vector`)
//
// TODO(tfoley): This deserves a better name than "magic"
FIDDLE()
class MagicTypeModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() SyntaxClass<NodeBase> magicNodeType;

    /// Modifier has a name so call this magicModifier to disambiguate
    FIDDLE() String magicName;
    FIDDLE() uint32_t tag = uint32_t(0);
};

// A modifier that indicates a built-in associated type requirement (e.g., `Differential`)
FIDDLE()
class BuiltinRequirementModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() BuiltinRequirementKind kind;
};


// A modifier applied to declarations of builtin types to indicate how they
// should be lowered to the IR.
//
// TODO: This should really subsume `BuiltinTypeModifier` and
// `MagicTypeModifier` so that we don't have to apply all of them.
FIDDLE()
class IntrinsicTypeModifier : public Modifier
{
    FIDDLE(...)
    // The IR opcode to use when constructing a type
    FIDDLE() uint32_t irOp;

    Token opToken;

    // Additional literal opreands to provide when creating instances.
    // (e.g., for a texture type this passes in shape/mutability info)
    FIDDLE() List<uint32_t> irOperands;
};

// Modifiers that affect the storage layout for matrices
FIDDLE(abstract)
class MatrixLayoutModifier : public Modifier
{
    FIDDLE(...)
};


// Modifiers that specify row- and column-major layout, respectively
FIDDLE(abstract)
class RowMajorLayoutModifier : public MatrixLayoutModifier
{
    FIDDLE(...)
};

FIDDLE(abstract)
class ColumnMajorLayoutModifier : public MatrixLayoutModifier
{
    FIDDLE(...)
};

// The HLSL flavor of those modifiers
FIDDLE()
class HLSLRowMajorLayoutModifier : public RowMajorLayoutModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLColumnMajorLayoutModifier : public ColumnMajorLayoutModifier
{
    FIDDLE(...)
};


// The GLSL flavor of those modifiers
//
// Note(tfoley): The GLSL versions of these modifiers are "backwards"
// in the sense that when a GLSL programmer requests row-major layout,
// we actually interpret that as requesting column-major. This makes
// sense because we interpret matrix conventions backwards from how
// GLSL specifies them.
FIDDLE()
class GLSLRowMajorLayoutModifier : public ColumnMajorLayoutModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLColumnMajorLayoutModifier : public RowMajorLayoutModifier
{
    FIDDLE(...)
};


// More HLSL Keyword

FIDDLE(abstract)
class InterpolationModeModifier : public Modifier
{
    FIDDLE(...)
};

// HLSL `nointerpolation` modifier
FIDDLE()
class HLSLNoInterpolationModifier : public InterpolationModeModifier
{
    FIDDLE(...)
};


// HLSL `noperspective` modifier
FIDDLE()
class HLSLNoPerspectiveModifier : public InterpolationModeModifier
{
    FIDDLE(...)
};


// HLSL `linear` modifier
FIDDLE()
class HLSLLinearModifier : public InterpolationModeModifier
{
    FIDDLE(...)
};


// HLSL `sample` modifier
FIDDLE()
class HLSLSampleModifier : public InterpolationModeModifier
{
    FIDDLE(...)
};


// HLSL `centroid` modifier
FIDDLE()
class HLSLCentroidModifier : public InterpolationModeModifier
{
    FIDDLE(...)
};

/// Slang-defined `pervertex` modifier
FIDDLE()
class PerVertexModifier : public InterpolationModeModifier
{
    FIDDLE(...)
};


// HLSL `precise` modifier
FIDDLE()
class PreciseModifier : public Modifier
{
    FIDDLE(...)
};


// HLSL `shared` modifier (which is used by the effect system,
// and shouldn't be confused with `groupshared`)
FIDDLE()
class HLSLEffectSharedModifier : public Modifier
{
    FIDDLE(...)
};


// HLSL `groupshared` modifier
FIDDLE()
class HLSLGroupSharedModifier : public Modifier
{
    FIDDLE(...)
};


// HLSL `static` modifier (probably doesn't need to be
// treated as HLSL-specific)
FIDDLE()
class HLSLStaticModifier : public Modifier
{
    FIDDLE(...)
};


// HLSL `uniform` modifier (distinct meaning from GLSL
// use of the keyword)
FIDDLE()
class HLSLUniformModifier : public Modifier
{
    FIDDLE(...)
};


// HLSL `volatile` modifier (ignored)
FIDDLE()
class HLSLVolatileModifier : public Modifier
{
    FIDDLE(...)
};


FIDDLE()
class AttributeTargetModifier : public Modifier
{
    FIDDLE(...)
    // A class to which the declared attribute type is applicable
    FIDDLE() SyntaxClass<NodeBase> syntaxClass;
};


// Base class for checked and unchecked `[name(arg0, ...)]` style attribute.
FIDDLE(abstract)
class AttributeBase : public Modifier
{
    FIDDLE(...)
    FIDDLE() AttributeDecl* attributeDecl = nullptr;

    // The original identifier token representing the last part of the qualified name.
    Token originalIdentifierToken;

    FIDDLE() List<Expr*> args;
};

// A `[name(...)]` attribute that hasn't undergone any semantic analysis.
// After analysis, this will be transformed into a more specific case.
FIDDLE()
class UncheckedAttribute : public AttributeBase
{
    FIDDLE(...)
    SLANG_UNREFLECTED
    Scope* scope = nullptr;
};

// A GLSL layout qualifier whose value has not yet been resolved or validated.
FIDDLE()
class UncheckedGLSLLayoutAttribute : public AttributeBase
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

// GLSL `binding` layout qualifier, does not include `set`.
FIDDLE()
class UncheckedGLSLBindingLayoutAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

// GLSL `set` layout qualifier, does not include `binding`.
FIDDLE()
class UncheckedGLSLSetLayoutAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

// GLSL `offset` layout qualifier.
FIDDLE()
class UncheckedGLSLOffsetLayoutAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLInputAttachmentIndexLayoutAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLLocationLayoutAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLIndexLayoutAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLConstantIdAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLRayPayloadAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLRayPayloadInAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLHitObjectAttributesAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLCallablePayloadAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class UncheckedGLSLCallablePayloadInAttribute : public UncheckedGLSLLayoutAttribute
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

// A `[name(arg0, ...)]` style attribute that has been validated.
FIDDLE()
class Attribute : public AttributeBase
{
    FIDDLE(...)
    FIDDLE() List<Val*> intArgVals;
};

FIDDLE()
class UserDefinedAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class AttributeUsageAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() SyntaxClass<NodeBase> targetSyntaxClass;
};

FIDDLE()
class NonDynamicUniformAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class RequireCapabilityAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() CapabilitySet capabilitySet;
};


// An `[unroll]` or `[unroll(count)]` attribute
FIDDLE()
class UnrollAttribute : public Attribute
{
    FIDDLE(...)
};

// An `[unroll]` or `[unroll(count)]` attribute
FIDDLE()
class ForceUnrollAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int32_t maxIterations = 0;
};

// An `[maxiters(count)]`
FIDDLE()
class MaxItersAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() IntVal* value = 0;
};

// An inferred max iteration count on a loop.
FIDDLE()
class InferredMaxItersAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() DeclRef<Decl> inductionVar;
    FIDDLE() int32_t value = 0;
};

FIDDLE()
class LoopAttribute : public Attribute
{
    FIDDLE(...)
};
// `[loop]`

FIDDLE()
class FastOptAttribute : public Attribute
{
    FIDDLE(...)
};
// `[fastopt]`

FIDDLE()
class AllowUAVConditionAttribute : public Attribute
{
    FIDDLE(...)
};
// `[allow_uav_condition]`

FIDDLE()
class BranchAttribute : public Attribute
{
    FIDDLE(...)
};
// `[branch]`

FIDDLE()
class FlattenAttribute : public Attribute
{
    FIDDLE(...)
};
// `[flatten]`

FIDDLE()
class ForceCaseAttribute : public Attribute
{
    FIDDLE(...)
};
// `[forcecase]`

FIDDLE()
class CallAttribute : public Attribute
{
    FIDDLE(...)
};
// `[call]`

FIDDLE()
class UnscopedEnumAttribute : public Attribute
{
    FIDDLE(...)
};

// Marks a enum to have `flags` semantics, where each enum case is a bitfield.
FIDDLE()
class FlagsAttribute : public Attribute
{
    FIDDLE(...)
};

// [[vk_push_constant]] [[push_constant]]
FIDDLE()
class PushConstantAttribute : public Attribute
{
    FIDDLE(...)
};

// [[vk_specialization_constant]] [[specialization_constant]]
FIDDLE()
class SpecializationConstantAttribute : public Attribute
{
    FIDDLE(...)
};

// [[vk_constant_id]]
FIDDLE()
class VkConstantIdAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int location;
};

// [[vk_shader_record]] [[shader_record]]
FIDDLE()
class ShaderRecordAttribute : public Attribute
{
    FIDDLE(...)
};


// [[vk_binding]]
FIDDLE()
class GLSLBindingAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int32_t binding = 0;
    FIDDLE() int32_t set = 0;
};

FIDDLE()
class VkAliasedPointerAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class VkRestrictPointerAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class GLSLOffsetLayoutAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int64_t offset;
};

// Implicitly added offset qualifier when no offset is specified.
FIDDLE()
class GLSLImplicitOffsetLayoutAttribute : public AttributeBase
{
    FIDDLE(...)
    SLANG_UNREFLECTED
};

FIDDLE()
class GLSLSimpleIntegerLayoutAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int32_t value = 0;
};

/// [[vk_input_attachment_index]]
FIDDLE()
class GLSLInputAttachmentIndexLayoutAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() IntegerLiteralValue location;
};

// [[vk_location]]
FIDDLE()
class GLSLLocationAttribute : public GLSLSimpleIntegerLayoutAttribute
{
    FIDDLE(...)
};


// [[vk_index]]
FIDDLE()
class GLSLIndexAttribute : public GLSLSimpleIntegerLayoutAttribute
{
    FIDDLE(...)
};

// [[vk_offset]]
FIDDLE()
class VkStructOffsetAttribute : public GLSLSimpleIntegerLayoutAttribute
{
    FIDDLE(...)
};

// [[vk_spirv_instruction]]
FIDDLE()
class SPIRVInstructionOpAttribute : public Attribute
{
    FIDDLE(...)
};

// [[spv_target_env_1_3]]
FIDDLE()
class SPIRVTargetEnv13Attribute : public Attribute
{
    FIDDLE(...)
};

// [[disable_array_flattening]]
FIDDLE()
class DisableArrayFlatteningAttribute : public Attribute
{
    FIDDLE(...)
};

// A GLSL layout(local_size_x = 64, ... attribute)
FIDDLE()
class GLSLLayoutLocalSizeAttribute : public Attribute
{
    FIDDLE(...)
    // The number of threads to use along each axis
    //
    // TODO: These should be accessors that use the
    // ordinary `args` list, rather than side data.
    FIDDLE() IntVal* extents[3];

    FIDDLE() bool axisIsSpecConstId[3];

    // References to specialization constants, for defining the number of
    // threads with them. If set, the corresponding axis is set to nullptr
    // above.
    FIDDLE() DeclRef<VarDeclBase> specConstExtents[3];
};

FIDDLE()
class GLSLLayoutDerivativeGroupQuadAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class GLSLLayoutDerivativeGroupLinearAttribute : public Attribute
{
    FIDDLE(...)
};

// TODO: for attributes that take arguments, the syntax node
// classes should provide accessors for the values of those arguments.

FIDDLE()
class MaxTessFactorAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class OutputControlPointsAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class OutputTopologyAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class PartitioningAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class PatchConstantFuncAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() FuncDecl* patchConstantFuncDecl = nullptr;
};

FIDDLE()
class DomainAttribute : public Attribute
{
    FIDDLE(...)
};


FIDDLE()
class EarlyDepthStencilAttribute : public Attribute
{
    FIDDLE(...)
};
// `[earlydepthstencil]`

// An HLSL `[numthreads(x,y,z)]` attribute
FIDDLE()
class NumThreadsAttribute : public Attribute
{
    FIDDLE(...)
    // The number of threads to use along each axis
    //
    // TODO: These should be accessors that use the
    // ordinary `args` list, rather than side data.
    FIDDLE() IntVal* extents[3];

    // References to specialization constants, for defining the number of
    // threads with them. If set, the corresponding axis is set to nullptr
    // above.
    FIDDLE() DeclRef<VarDeclBase> specConstExtents[3];
};

FIDDLE()
class WaveSizeAttribute : public Attribute
{
    FIDDLE(...)
    // "numLanes" must be a compile time constant integer
    // value of an allowed wave size, which is one of the
    // followings: 4, 8, 16, 32, 64 or 128.
    //
    FIDDLE() IntVal* numLanes;
};

FIDDLE()
class MaxVertexCountAttribute : public Attribute
{
    FIDDLE(...)
    // The number of max vertex count for geometry shader
    //
    // TODO: This should be an accessor that uses the
    // ordinary `args` list, rather than side data.
    FIDDLE() int32_t value;
};

FIDDLE()
class InstanceAttribute : public Attribute
{
    FIDDLE(...)
    // The number of instances to run for geometry shader
    //
    // TODO: This should be an accessor that uses the
    // ordinary `args` list, rather than side data.
    FIDDLE() int32_t value;
};

// A `[shader("stageName")]`/`[shader("capability")]` attribute which
// marks an entry point for compiling. This attribute also specifies
// the 'capabilities' implicitly supported by an entry point
FIDDLE()
class EntryPointAttribute : public Attribute
{
    FIDDLE(...)
    // The resolved capailities for our entry point.
    FIDDLE() CapabilitySet capabilitySet;
};

// A `[__vulkanRayPayload(location)]` attribute, which is used in the
// core module implementation to indicate that a variable
// actually represents the input/output interface for a Vulkan
// ray tracing shader to pass per-ray payload information.
FIDDLE()
class VulkanRayPayloadAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int location;
};
FIDDLE()
class VulkanRayPayloadInAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int location;
};

// A `[__vulkanCallablePayload(location)]` attribute, which is used in the
// core module implementation to indicate that a variable
// actually represents the input/output interface for a Vulkan
// ray tracing shader to pass payload information to/from a callee.
FIDDLE()
class VulkanCallablePayloadAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int location;
};
FIDDLE()
class VulkanCallablePayloadInAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int location;
};

// A `[__vulkanHitAttributes]` attribute, which is used in the
// core module implementation to indicate that a variable
// actually represents the output interface for a Vulkan
// intersection shader to pass hit attribute information.
FIDDLE()
class VulkanHitAttributesAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[__vulkanHitObjectAttributes(location)]` attribute, which is used in the
// core module implementation to indicate that a variable
// actually represents the attributes on a HitObject as part of
// Shader ExecutionReordering
FIDDLE()
class VulkanHitObjectAttributesAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int location;
};

// A `[mutating]` attribute, which indicates that a member
// function is allowed to modify things through its `this`
// argument.
//
FIDDLE()
class MutatingAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[nonmutating]` attribute, which indicates that a
// `set` accessor does not need to modify anything through
// its `this` parameter.
//
FIDDLE()
class NonmutatingAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[constref]` attribute, which indicates that the `this` parameter of
// a member function should be passed by const reference.
//
FIDDLE()
class ConstRefAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[ref]` attribute, which indicates that the `this` parameter of
// a member function should be passed by reference.
//
FIDDLE()
class RefAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[__readNone]` attribute, which indicates that a function
// computes its results strictly based on argument values, without
// reading or writing through any pointer arguments, or any other
// state that could be observed by a caller.
//
FIDDLE()
class ReadNoneAttribute : public Attribute
{
    FIDDLE(...)
};


// A `[__GLSLRequireShaderInputParameter]` attribute to annotate
// functions that require a shader input as parameter
//
FIDDLE()
class GLSLRequireShaderInputParameterAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() uint32_t parameterNumber;
};

// HLSL modifiers for geometry shader input topology
FIDDLE()
class HLSLGeometryShaderInputPrimitiveTypeModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLPointModifier : public HLSLGeometryShaderInputPrimitiveTypeModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLLineModifier : public HLSLGeometryShaderInputPrimitiveTypeModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLTriangleModifier : public HLSLGeometryShaderInputPrimitiveTypeModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLLineAdjModifier : public HLSLGeometryShaderInputPrimitiveTypeModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLTriangleAdjModifier : public HLSLGeometryShaderInputPrimitiveTypeModifier
{
    FIDDLE(...)
};

// Mesh shader paramters

FIDDLE()
class HLSLMeshShaderOutputModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLVerticesModifier : public HLSLMeshShaderOutputModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLIndicesModifier : public HLSLMeshShaderOutputModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLPrimitivesModifier : public HLSLMeshShaderOutputModifier
{
    FIDDLE(...)
};

FIDDLE()
class HLSLPayloadModifier : public Modifier
{
    FIDDLE(...)
};

// A modifier to indicate that a constructor/initializer can be used
// to perform implicit type conversion, and to specify the cost of
// the conversion, if applied.
FIDDLE()
class ImplicitConversionModifier : public Modifier
{
    FIDDLE(...)
    // The conversion cost, used to rank conversions
    FIDDLE() ConversionCost cost = kConversionCost_None;

    // A builtin identifier for identifying conversions that need special treatment.
    FIDDLE() BuiltinConversionKind builtinConversionKind = kBuiltinConversion_Unknown;
};

FIDDLE()
class FormatAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() ImageFormat format;
};

FIDDLE()
class AllowAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() DiagnosticInfo const* diagnostic = nullptr;
};


// A `[__extern]` attribute, which indicates that a function/type is defined externally
//
FIDDLE()
class ExternAttribute : public Attribute
{
    FIDDLE(...)
};


// An `[__unsafeForceInlineExternal]` attribute indicates that the callee should be inlined
// into call sites after initial IR generation (that is, as early as possible).
//
FIDDLE()
class UnsafeForceInlineEarlyAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[ForceInline]` attribute indicates that the callee should be inlined
// by the Slang compiler.
//
FIDDLE()
class ForceInlineAttribute : public Attribute
{
    FIDDLE(...)
};


/// An attribute that marks a type declaration as either allowing or
/// disallowing the type to be inherited from in other modules.
FIDDLE(abstract)
class InheritanceControlAttribute : public Attribute
{
    FIDDLE(...)
};

/// An attribute that marks a type declaration as allowing the type to be inherited from in other
/// modules.
FIDDLE()
class OpenAttribute : public InheritanceControlAttribute
{
    FIDDLE(...)
};

/// An attribute that marks a type declaration as disallowing the type to be inherited from in other
/// modules.
FIDDLE()
class SealedAttribute : public InheritanceControlAttribute
{
    FIDDLE(...)
};

/// An attribute that marks a decl as a compiler built-in object.
FIDDLE()
class BuiltinAttribute : public Attribute
{
    FIDDLE(...)
};

/// An attribute that marks a decl as a compiler built-in object for the autodiff system.
FIDDLE()
class AutoDiffBuiltinAttribute : public Attribute
{
    FIDDLE(...)
};

/// An attribute that defines the size of `AnyValue` type to represent a polymoprhic value that
/// conforms to the decorated interface type.
FIDDLE()
class AnyValueSizeAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int32_t size;
};

/// This is a stop-gap solution to break overload ambiguity in the core module.
/// When there is a function overload ambiguity, the compiler will pick the one with higher rank
/// specified by this attribute. An overload without this attribute will have a rank of 0.
/// In the future, we should enhance our type system to take into account the "specialized"-ness
/// of an overload, such that `T overload1<T:IDerived>()` is more specialized than `T
/// overload2<T:IBase>()` and preferred during overload resolution.
FIDDLE()
class OverloadRankAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() int32_t rank;
};

/// An attribute that marks an interface for specialization use only. Any operation that triggers
/// dynamic dispatch through the interface is a compile-time error.
FIDDLE()
class SpecializeAttribute : public Attribute
{
    FIDDLE(...)
};

/// An attribute that marks a type, function or variable as differentiable.
FIDDLE()
class DifferentiableAttribute : public Attribute
{
    FIDDLE(...)
    // TODO(tfoley): Why is there this duplication here?
    List<KeyValuePair<Type*, SubtypeWitness*>> m_typeToIDifferentiableWitnessMappings;

    void addType(Type* declRef, SubtypeWitness* witness)
    {
        getMapTypeToIDifferentiableWitness();
        if (m_mapToIDifferentiableWitness.addIfNotExists(declRef, witness))
        {
            m_typeToIDifferentiableWitnessMappings.add(
                KeyValuePair<Type*, SubtypeWitness*>(declRef, witness));
        }
    }

    /// Mapping from types to subtype witnesses for conformance to IDifferentiable.
    const OrderedDictionary<Type*, SubtypeWitness*>& getMapTypeToIDifferentiableWitness();

    SLANG_UNREFLECTED ValSet m_typeRegistrationWorkingSet;

private:
    OrderedDictionary<Type*, SubtypeWitness*> m_mapToIDifferentiableWitness;
};

FIDDLE()
class DllImportAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() String modulePath;

    FIDDLE() String functionName;
};

FIDDLE()
class DllExportAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class TorchEntryPointAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class CudaDeviceExportAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class CudaKernelAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class CudaHostAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class AutoPyBindCudaAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class PyExportAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() String name;
};

FIDDLE()
class PreferRecomputeAttribute : public Attribute
{
    FIDDLE(...)

    enum SideEffectBehavior
    {
        Warn = 0,
        Allow = 1
    };

    FIDDLE() SideEffectBehavior sideEffectBehavior;
};

FIDDLE()
class PreferCheckpointAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class DerivativeMemberAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() DeclRefExpr* memberDeclRef;
};

/// An attribute that marks an interface type as a COM interface declaration.
FIDDLE()
class ComInterfaceAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() String guid;
};

/// A `[__requiresNVAPI]` attribute indicates that the declaration being modifed
/// requires NVAPI operations for its implementation on D3D.
FIDDLE()
class RequiresNVAPIAttribute : public Attribute
{
    FIDDLE(...)
};

/// A `[RequirePrelude(target, "string")]` attribute indicates that the declaration being modifed
/// requires a textual prelude to be injected in the resulting target code.
FIDDLE()
class RequirePreludeAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() CapabilitySet capabilitySet;
    FIDDLE() String prelude;
};

/// A `[__AlwaysFoldIntoUseSite]` attribute indicates that the calls into the modified
/// function should always be folded into use sites during source emit.
FIDDLE()
class AlwaysFoldIntoUseSiteAttribute : public Attribute
{
    FIDDLE(...)
};

// A `[TreatAsDifferentiableAttribute]` attribute indicates that a function or an interface
// should be treated as differentiable in IR validation step.
//
FIDDLE()
class TreatAsDifferentiableAttribute : public DifferentiableAttribute
{
    FIDDLE(...)
};

/// The `[ForwardDifferentiable]` attribute indicates that a function can be forward-differentiated.
FIDDLE()
class ForwardDifferentiableAttribute : public DifferentiableAttribute
{
    FIDDLE(...)
};

FIDDLE()
class UserDefinedDerivativeAttribute : public DifferentiableAttribute
{
    FIDDLE(...)
    FIDDLE() Expr* funcExpr;
};

/// The `[ForwardDerivative(function)]` attribute specifies a custom function that should
/// be used as the derivative for the decorated function.
FIDDLE()
class ForwardDerivativeAttribute : public UserDefinedDerivativeAttribute
{
    FIDDLE(...)
};

FIDDLE()
class DerivativeOfAttribute : public DifferentiableAttribute
{
    FIDDLE(...)
    FIDDLE() Expr* funcExpr;

    FIDDLE()
    Expr* backDeclRef; // DeclRef to this derivative function when initiated from primalFunction.
};

/// The `[ForwardDerivativeOf(primalFunction)]` attribute marks the decorated function as custom
/// derivative implementation for `primalFunction`.
/// ForwardDerivativeOfAttribute inherits from DifferentiableAttribute because a derivative
/// function itself is considered differentiable.
FIDDLE()
class ForwardDerivativeOfAttribute : public DerivativeOfAttribute
{
    FIDDLE(...)
};

/// The `[BackwardDifferentiable]` attribute indicates that a function can be
/// backward-differentiated.
FIDDLE()
class BackwardDifferentiableAttribute : public DifferentiableAttribute
{
    FIDDLE(...)
    FIDDLE() int maxOrder = 0;
};

/// The `[BackwardDerivative(function)]` attribute specifies a custom function that should
/// be used as the backward-derivative for the decorated function.
FIDDLE()
class BackwardDerivativeAttribute : public UserDefinedDerivativeAttribute
{
    FIDDLE(...)
};

/// The `[BackwardDerivativeOf(primalFunction)]` attribute marks the decorated function as custom
/// backward-derivative implementation for `primalFunction`.
FIDDLE()
class BackwardDerivativeOfAttribute : public DerivativeOfAttribute
{
    FIDDLE(...)
};

/// The `[PrimalSubstitute(function)]` attribute specifies a custom function that should
/// be used as the primal function substitute when differentiating code that calls the primal
/// function.
FIDDLE()
class PrimalSubstituteAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() Expr* funcExpr;
};

/// The `[PrimalSubstituteOf(primalFunction)]` attribute marks the decorated function as
/// the substitute primal function in a forward or backward derivative function.
FIDDLE()
class PrimalSubstituteOfAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() Expr* funcExpr;
    FIDDLE()
    Expr* backDeclRef; // DeclRef to this derivative function when initiated from primalFunction.
};

/// The `[NoDiffThis]` attribute is used to specify that the `this` parameter should not be
/// included for differentiation.
FIDDLE()
class NoDiffThisAttribute : public Attribute
{
    FIDDLE(...)
};

/// Indicates that the modified declaration is one of the "magic" declarations
/// that NVAPI uses to communicate extended operations. When NVAPI is being included
/// via the prelude for downstream compilation, declarations with this modifier
/// will not be emitted, instead allowing the versions from the prelude to be used.
FIDDLE()
class NVAPIMagicModifier : public Modifier
{
    FIDDLE(...)
};

/// A modifier that attaches to a `ModuleDecl` to indicate the register/space binding
/// that NVAPI wants to use, as indicated by, e.g., the `NV_SHADER_EXTN_SLOT` and
/// `NV_SHADER_EXTN_REGISTER_SPACE` preprocessor definitions.
FIDDLE()
class NVAPISlotModifier : public Modifier
{
    FIDDLE(...)
    /// The name of the register that is to be used (e.g., `"u3"`)
    ///
    /// This value will come from the `NV_SHADER_EXTN_SLOT` macro, if set.
    ///
    /// The `registerName` field must always be filled in when adding
    /// an `NVAPISlotModifier` to a module; if no register name is defined,
    /// then the modifier should not be added.
    ///
    FIDDLE() String registerName;

    /// The name of the register space to be used (e.g., `space1`)
    ///
    /// This value will come from the `NV_SHADER_EXTN_REGISTER_SPACE` macro,
    /// if set.
    ///
    /// It is valid for a user to specify a register name but not a space name,
    /// and in that case `spaceName` will be set to `"space0"`.
    FIDDLE() String spaceName;
};

/// A `[noinline]` attribute represents a request by the application that,
/// to the extent possible, a function should not be inlined into call sites.
///
/// Note that due to various limitations of different targets, it is entirely
/// possible for such functions to be inlined or specialized to call sites.
///
FIDDLE()
class NoInlineAttribute : public Attribute
{
    FIDDLE(...)
};

/// A `[noRefInline]` attribute represents a request to not force inline a
/// function specifically due to a refType parameter.
FIDDLE()
class NoRefInlineAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class DerivativeGroupQuadAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class DerivativeGroupLinearAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class MaximallyReconvergesAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class QuadDerivativesAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class RequireFullQuadsAttribute : public Attribute
{
    FIDDLE(...)
};

/// A `[payload]` attribute indicates that a `struct` type will be used as
/// a ray payload for `TraceRay()` calls, and thus also as input/output
/// for shaders in the ray tracing pipeline that might be invoked for
/// such a ray.
///
FIDDLE()
class PayloadAttribute : public Attribute
{
    FIDDLE(...)
};

/// A `[raypayload]` attribute indicates that a `struct` type will be used as
/// a ray payload for `TraceRay()` calls, and thus also as input/output
/// for shaders in the ray tracing pipeline that might be invoked for
/// such a ray.
///
FIDDLE()
class RayPayloadAttribute : public Attribute
{
    FIDDLE(...)
};

/// A `[deprecated("message")]` attribute indicates the target is
/// deprecated.
/// A compiler warning including the message will be raised if the
/// deprecated value is used.
///
FIDDLE()
class DeprecatedAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() String message;
};

FIDDLE()
class NonCopyableTypeAttribute : public Attribute
{
    FIDDLE(...)
};

FIDDLE()
class NoSideEffectAttribute : public Attribute
{
    FIDDLE(...)
};

/// A `[KnownBuiltin("name")]` attribute allows the compiler to
/// identify this declaration during compilation, despite obfuscation or
/// linkage removing optimizations
///
FIDDLE()
class KnownBuiltinAttribute : public Attribute
{
    FIDDLE(...)
    FIDDLE() String name;
};

/// A modifier that applies to types rather than declarations.
///
/// In most cases, the Slang compiler assumes that a modifier should
/// inhere to a declaration. Given input like:
///
/// mod1 mod2 int myVar = ...;
///
/// The default assumption is that `mod1` and `mod2` apply to `myVar`
/// and *not* to the `int` type.
///
/// In order to allow modifiers to inhere to the type instead, we introduce
/// a base class for modifiers that really don't want to belong to the declaration,
/// and instead want to belong to the type (or rather the type *specifier*
/// from a parsing standpoint).
///
FIDDLE()
class TypeModifier : public Modifier
{
    FIDDLE(...)
};

/// A kind of syntax element which appears as a modifier in the syntax, but
/// we represent as a function over type expressions
FIDDLE()
class WrappingTypeModifier : public TypeModifier
{
    FIDDLE(...)
};

/// A modifier that applies to a type and implies information about the
/// underlying format of a resource that uses that type as its element type.
///
FIDDLE()
class ResourceElementFormatModifier : public TypeModifier
{
    FIDDLE(...)
};

/// HLSL `unorm` modifier
FIDDLE()
class UNormModifier : public ResourceElementFormatModifier
{
    FIDDLE(...)
};

/// HLSL `snorm` modifier
FIDDLE()
class SNormModifier : public ResourceElementFormatModifier
{
    FIDDLE(...)
};

FIDDLE()
class NoDiffModifier : public TypeModifier
{
    FIDDLE(...)
};

FIDDLE()
class GloballyCoherentModifier : public SimpleModifier
{
    FIDDLE(...)
};

// Some GLSL-specific modifiers
FIDDLE()
class GLSLBufferModifier : public WrappingTypeModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLWriteOnlyModifier : public SimpleModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLReadOnlyModifier : public SimpleModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLVolatileModifier : public SimpleModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLRestrictModifier : public SimpleModifier
{
    FIDDLE(...)
};

FIDDLE()
class GLSLPatchModifier : public SimpleModifier
{
    FIDDLE(...)
};

//
FIDDLE()
class BitFieldModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() IntegerLiteralValue width;

    // Fields filled during semantic analysis
    FIDDLE() IntegerLiteralValue offset = 0;
    FIDDLE() DeclRef<VarDecl> backingDeclRef;
};

FIDDLE()
class DynamicUniformModifier : public Modifier
{
    FIDDLE(...)
};

FIDDLE()
class MemoryQualifierSetModifier : public Modifier
{
    FIDDLE(...)
    FIDDLE() List<Modifier*> memoryModifiers;

    FIDDLE() uint32_t memoryQualifiers = 0;

public:
    struct Flags
    {
        enum MemoryQualifiersBit
        {
            kNone = 0b0,
            kCoherent = 0b1,
            kReadOnly = 0b10,
            kWriteOnly = 0b100,
            kVolatile = 0b1000,
            kRestrict = 0b10000,
            kRasterizerOrdered = 0b100000,
        };
    };

    void addQualifier(Modifier* mod, Flags::MemoryQualifiersBit type)
    {
        memoryModifiers.add(mod);
        memoryQualifiers |= type;
    }
    uint32_t getMemoryQualifierBit() { return memoryQualifiers; }
    List<Modifier*> getModifiers() { return memoryModifiers; }
};

} // namespace Slang
