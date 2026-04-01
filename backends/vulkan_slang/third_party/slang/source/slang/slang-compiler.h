#ifndef SLANG_COMPILER_H_INCLUDED
#define SLANG_COMPILER_H_INCLUDED

#include "../compiler-core/slang-artifact-representation-impl.h"
#include "../compiler-core/slang-command-line-args.h"
#include "../compiler-core/slang-downstream-compiler-util.h"
#include "../compiler-core/slang-downstream-compiler.h"
#include "../compiler-core/slang-include-system.h"
#include "../compiler-core/slang-name.h"
#include "../compiler-core/slang-source-embed-util.h"
#include "../compiler-core/slang-spirv-core-grammar.h"
#include "../core/slang-basic.h"
#include "../core/slang-command-options.h"
#include "../core/slang-crypto.h"
#include "../core/slang-file-system.h"
#include "../core/slang-shared-library.h"
#include "../core/slang-std-writers.h"
#include "slang-capability.h"
#include "slang-com-ptr.h"
#include "slang-compiler-options.h"
#include "slang-content-assist-info.h"
#include "slang-diagnostics.h"
#include "slang-hlsl-to-vulkan-layout-options.h"
#include "slang-preprocessor.h"
#include "slang-profile.h"
#include "slang-serialize-ir-types.h"
#include "slang-syntax.h"
#include "slang.h"

#include <chrono>

namespace Slang
{
struct PathInfo;
struct IncludeHandler;
struct SharedSemanticsContext;
struct ModuleChunkRef;

class ProgramLayout;
class PtrType;
class TargetProgram;
class TargetRequest;
class TypeLayout;
class Artifact;

enum class CompilerMode
{
    ProduceLibrary,
    ProduceShader,
    GenerateChoice
};

enum class StageTarget
{
    Unknown,
    VertexShader,
    HullShader,
    DomainShader,
    GeometryShader,
    FragmentShader,
    ComputeShader,
};

enum class CodeGenTarget : SlangCompileTargetIntegral
{
    Unknown = SLANG_TARGET_UNKNOWN,
    None = SLANG_TARGET_NONE,
    GLSL = SLANG_GLSL,
    HLSL = SLANG_HLSL,
    SPIRV = SLANG_SPIRV,
    SPIRVAssembly = SLANG_SPIRV_ASM,
    DXBytecode = SLANG_DXBC,
    DXBytecodeAssembly = SLANG_DXBC_ASM,
    DXIL = SLANG_DXIL,
    DXILAssembly = SLANG_DXIL_ASM,
    CSource = SLANG_C_SOURCE,
    CPPSource = SLANG_CPP_SOURCE,
    PyTorchCppBinding = SLANG_CPP_PYTORCH_BINDING,
    HostCPPSource = SLANG_HOST_CPP_SOURCE,
    HostExecutable = SLANG_HOST_EXECUTABLE,
    HostSharedLibrary = SLANG_HOST_SHARED_LIBRARY,
    ShaderSharedLibrary = SLANG_SHADER_SHARED_LIBRARY,
    ShaderHostCallable = SLANG_SHADER_HOST_CALLABLE,
    CUDASource = SLANG_CUDA_SOURCE,
    PTX = SLANG_PTX,
    CUDAObjectCode = SLANG_CUDA_OBJECT_CODE,
    ObjectCode = SLANG_OBJECT_CODE,
    HostHostCallable = SLANG_HOST_HOST_CALLABLE,
    Metal = SLANG_METAL,
    MetalLib = SLANG_METAL_LIB,
    MetalLibAssembly = SLANG_METAL_LIB_ASM,
    WGSL = SLANG_WGSL,
    WGSLSPIRVAssembly = SLANG_WGSL_SPIRV_ASM,
    WGSLSPIRV = SLANG_WGSL_SPIRV,
    HostVM = SLANG_HOST_VM,
    CountOf = SLANG_TARGET_COUNT_OF,
};

bool isHeterogeneousTarget(CodeGenTarget target);

void printDiagnosticArg(StringBuilder& sb, CodeGenTarget val);

enum class ContainerFormat : SlangContainerFormatIntegral
{
    None = SLANG_CONTAINER_FORMAT_NONE,
    SlangModule = SLANG_CONTAINER_FORMAT_SLANG_MODULE,
};

enum class LineDirectiveMode : SlangLineDirectiveModeIntegral
{
    Default = SLANG_LINE_DIRECTIVE_MODE_DEFAULT,
    None = SLANG_LINE_DIRECTIVE_MODE_NONE,
    Standard = SLANG_LINE_DIRECTIVE_MODE_STANDARD,
    GLSL = SLANG_LINE_DIRECTIVE_MODE_GLSL,
    SourceMap = SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP,
};

enum class ResultFormat
{
    None,
    Text,
    Binary,
};

// When storing the layout for a matrix-type
// value, we need to know whether it has been
// laid out with row-major or column-major
// storage.
//
enum MatrixLayoutMode : SlangMatrixLayoutModeIntegral
{
    kMatrixLayoutMode_RowMajor = SLANG_MATRIX_LAYOUT_ROW_MAJOR,
    kMatrixLayoutMode_ColumnMajor = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
};

enum class DebugInfoLevel : SlangDebugInfoLevelIntegral
{
    None = SLANG_DEBUG_INFO_LEVEL_NONE,
    Minimal = SLANG_DEBUG_INFO_LEVEL_MINIMAL,
    Standard = SLANG_DEBUG_INFO_LEVEL_STANDARD,
    Maximal = SLANG_DEBUG_INFO_LEVEL_MAXIMAL,
};

enum class DebugInfoFormat : SlangDebugInfoFormatIntegral
{
    Default = SLANG_DEBUG_INFO_FORMAT_DEFAULT,
    C7 = SLANG_DEBUG_INFO_FORMAT_C7,
    Pdb = SLANG_DEBUG_INFO_FORMAT_PDB,

    Stabs = SLANG_DEBUG_INFO_FORMAT_STABS,
    Coff = SLANG_DEBUG_INFO_FORMAT_COFF,
    Dwarf = SLANG_DEBUG_INFO_FORMAT_DWARF,

    CountOf = SLANG_DEBUG_INFO_FORMAT_COUNT_OF,
};

enum class OptimizationLevel : SlangOptimizationLevelIntegral
{
    None = SLANG_OPTIMIZATION_LEVEL_NONE,
    Default = SLANG_OPTIMIZATION_LEVEL_DEFAULT,
    High = SLANG_OPTIMIZATION_LEVEL_HIGH,
    Maximal = SLANG_OPTIMIZATION_LEVEL_MAXIMAL,
};

struct CodeGenContext;
class EndToEndCompileRequest;
class FrontEndCompileRequest;
class Linkage;
class Module;
class TranslationUnitRequest;

/// Information collected about global or entry-point shader parameters
struct ShaderParamInfo
{
    DeclRef<VarDeclBase> paramDeclRef;
    Int firstSpecializationParamIndex = 0;
    Int specializationParamCount = 0;
};

/// A request for the front-end to find and validate an entry-point function
struct FrontEndEntryPointRequest : RefObject
{
public:
    /// Create a request for an entry point.
    FrontEndEntryPointRequest(
        FrontEndCompileRequest* compileRequest,
        int translationUnitIndex,
        Name* name,
        Profile profile);

    /// Get the parent front-end compile request.
    FrontEndCompileRequest* getCompileRequest() { return m_compileRequest; }

    /// Get the translation unit that contains the entry point.
    TranslationUnitRequest* getTranslationUnit();

    /// Get the name of the entry point to find.
    Name* getName() { return m_name; }

    /// Get the stage that the entry point is to be compiled for
    Stage getStage() { return m_profile.getStage(); }

    /// Get the profile that the entry point is to be compiled for
    Profile getProfile() { return m_profile; }

    /// Get the index to the translation unit
    int getTranslationUnitIndex() const { return m_translationUnitIndex; }

private:
    // The parent compile request
    FrontEndCompileRequest* m_compileRequest;

    // The index of the translation unit that will hold the entry point
    int m_translationUnitIndex;

    // The name of the entry point function to look for
    Name* m_name;

    // The profile to compile for (including stage)
    Profile m_profile;
};

/// Tracks an ordered list of modules that something depends on.
/// TODO: Shader caching currently relies on this being in well defined order.
struct ModuleDependencyList
{
public:
    /// Get the list of modules that are depended on.
    List<Module*> const& getModuleList() { return m_moduleList; }

    /// Add a module and everything it depends on to the list.
    void addDependency(Module* module);

    /// Add a module to the list, but not the modules it depends on.
    void addLeafDependency(Module* module);

private:
    void _addDependency(Module* module);

    List<Module*> m_moduleList;
    HashSet<Module*> m_moduleSet;
};

/// Tracks an unordered list of source files that something depends on
/// TODO: Shader caching currently relies on this being in well defined order.
struct FileDependencyList
{
public:
    /// Get the list of files that are depended on.
    List<SourceFile*> const& getFileList() { return m_fileList; }

    /// Add a file to the list, if it is not already present
    void addDependency(SourceFile* sourceFile);

    /// Add all of the paths that `module` depends on to the list
    void addDependency(Module* module);

    void clear()
    {
        m_fileList.clear();
        m_fileSet.clear();
    }

private:
    // TODO: We are using a `HashSet` here to deduplicate
    // the paths so that we don't return the same path
    // multiple times from `getFilePathList`, but because
    // order isn't important, we could potentially do better
    // in terms of memory (at some cost in performance) by
    // just sorting the `m_fileList` every once in
    // a while and then deduplicating.

    List<SourceFile*> m_fileList;
    HashSet<SourceFile*> m_fileSet;
};


class EntryPoint;

class ComponentType;
class ComponentTypeVisitor;

/// Base class for "component types" that represent the pieces a final
/// shader program gets linked together from.
///
class ComponentType : public RefObject,
                      public slang::IComponentType,
                      public slang::IModulePrecompileService_Experimental
{
public:
    //
    // ISlangUnknown interface
    //

    SLANG_REF_OBJECT_IUNKNOWN_ALL;
    ISlangUnknown* getInterface(Guid const& guid);

    //
    // slang::IComponentType interface
    //

    SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() SLANG_OVERRIDE;
    SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex, slang::IBlob** outDiagnostics) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE;

    IArtifact* getTargetArtifact(SlangInt targetIndex, slang::IBlob** outDiagnostics);

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics = nullptr) SLANG_OVERRIDE;

    SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) SLANG_OVERRIDE;

    SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, slang::IComponentType** outEntryPoint) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL link(
        slang::IComponentType** outLinkedComponentType,
        ISlangBlob** outDiagnostics) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE;

    /// ComponentType is the only class inheriting from IComponentType that provides a
    /// meaningful implementation for this function. All others should forward these and
    /// implement `buildHash`.
    SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outHash) SLANG_OVERRIDE;

    SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        slang::IComponentType** outLinkedComponentType,
        uint32_t count,
        slang::CompilerOptionEntry* entries,
        ISlangBlob** outDiagnostics) override;


    //
    // slang::IModulePrecompileService interface
    //
    SLANG_NO_THROW SlangResult SLANG_MCALL
    precompileForTarget(SlangCompileTarget target, slang::IBlob** outDiagnostics) SLANG_OVERRIDE;

    SLANG_NO_THROW SlangResult SLANG_MCALL getPrecompiledTargetCode(
        SlangCompileTarget target,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) SLANG_OVERRIDE;

    SLANG_NO_THROW SlangInt SLANG_MCALL getModuleDependencyCount() SLANG_OVERRIDE;

    SLANG_NO_THROW SlangResult SLANG_MCALL getModuleDependency(
        SlangInt dependencyIndex,
        slang::IModule** outModule,
        slang::IBlob** outDiagnostics = nullptr) SLANG_OVERRIDE;

    CompilerOptionSet& getOptionSet() { return m_optionSet; }

    /// Get the linkage (aka "session" in the public API) for this component type.
    Linkage* getLinkage() { return m_linkage; }

    /// Get the target-specific version of this program for the given `target`.
    ///
    /// The `target` must be a target on the `Linkage` that was used to create this program.
    TargetProgram* getTargetProgram(TargetRequest* target);

    /// Update the hash builder with the dependencies for this component type.
    virtual void buildHash(DigestBuilder<SHA1>& builder) = 0;

    /// Get the number of entry points linked into this component type.
    virtual Index getEntryPointCount() = 0;

    /// Get one of the entry points linked into this component type.
    virtual RefPtr<EntryPoint> getEntryPoint(Index index) = 0;

    /// Get the mangled name of one of the entry points linked into this component type.
    virtual String getEntryPointMangledName(Index index) = 0;

    /// Get the name override of one of the entry points linked into this component type.
    virtual String getEntryPointNameOverride(Index index) = 0;

    /// Get the number of global shader parameters linked into this component type.
    virtual Index getShaderParamCount() = 0;

    /// Get one of the global shader parametesr linked into this component type.
    virtual ShaderParamInfo getShaderParam(Index index) = 0;

    /// Get the specialization parameter at `index`.
    virtual SpecializationParam const& getSpecializationParam(Index index) = 0;

    /// Get the number of "requirements" that this component type has.
    ///
    /// A requirement represents another component type that this component
    /// needs in order to function correctly. For example, the dependency
    /// of one module on another module that it `import`s is represented
    /// as a requirement, as is the dependency of an entry point on the
    /// module that defines it.
    ///
    virtual Index getRequirementCount() = 0;

    /// Get the requirement at `index`.
    virtual RefPtr<ComponentType> getRequirement(Index index) = 0;

    /// Parse a type from a string, in the context of this component type.
    ///
    /// Any names in the string will be resolved using the modules
    /// referenced by the program.
    ///
    /// On an error, returns null and reports diagnostic messages
    /// to the provided `sink`.
    ///
    /// TODO: This function shouldn't be on the base class, since
    /// it only really makes sense on `Module`.
    ///
    Type* getTypeFromString(String const& typeStr, DiagnosticSink* sink);

    Expr* findDeclFromString(String const& name, DiagnosticSink* sink);

    Expr* findDeclFromStringInType(
        Type* type,
        String const& name,
        LookupMask mask,
        DiagnosticSink* sink);

    bool isSubType(Type* subType, Type* superType);

    Dictionary<String, IntVal*>& getMangledNameToIntValMap();
    ConstantIntVal* tryFoldIntVal(IntVal* intVal);

    /// Get a list of modules that this component type depends on.
    ///
    virtual List<Module*> const& getModuleDependencies() = 0;

    /// Get the full list of source files this component type depends on.
    ///
    virtual List<SourceFile*> const& getFileDependencies() = 0;

    /// Callback for use with `enumerateIRModules`
    typedef void (*EnumerateIRModulesCallback)(IRModule* irModule, void* userData);

    /// Invoke `callback` on all the IR modules that are (transitively) linked into this component
    /// type.
    void enumerateIRModules(EnumerateIRModulesCallback callback, void* userData);

    /// Invoke `callback` on all the IR modules that are (transitively) linked into this component
    /// type.
    template<typename F>
    void enumerateIRModules(F const& callback)
    {
        struct Helper
        {
            static void helper(IRModule* irModule, void* userData) { (*(F*)userData)(irModule); }
        };
        enumerateIRModules(&Helper::helper, (void*)&callback);
    }

    /// Callback for use with `enumerateModules`
    typedef void (*EnumerateModulesCallback)(Module* module, void* userData);

    /// Invoke `callback` on all the modules that are (transitively) linked into this component
    /// type.
    void enumerateModules(EnumerateModulesCallback callback, void* userData);

    /// Invoke `callback` on all the modules that are (transitively) linked into this component
    /// type.
    template<typename F>
    void enumerateModules(F const& callback)
    {
        struct Helper
        {
            static void helper(Module* module, void* userData) { (*(F*)userData)(module); }
        };
        enumerateModules(&Helper::helper, (void*)&callback);
    }

    /// Side-band information generated when specializing this component type.
    ///
    /// Difference subclasses of `ComponentType` are expected to create their
    /// own subclass of `SpecializationInfo` as the output of `_validateSpecializationArgs`.
    /// Later, whenever we want to use a specialized component type we will
    /// also have the `SpecializationInfo` available and will expect it to
    /// have the correct (subclass-specific) type.
    ///
    class SpecializationInfo : public RefObject
    {
    };

    /// Validate the given specialization `args` and compute any side-band specialization info.
    ///
    /// Any errors will be reported to `sink`, which can thus be used to test
    /// if the operation was successful.
    ///
    /// A null return value is allowed, since not all subclasses require
    /// custom side-band specialization information.
    ///
    /// This function is an implementation detail of `specialize()`.
    ///
    virtual RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) = 0;

    /// Validate the given specialization `args` and compute any side-band specialization info.
    ///
    /// Any errors will be reported to `sink`, which can thus be used to test
    /// if the operation was successful.
    ///
    /// A null return value is allowed, since not all subclasses require
    /// custom side-band specialization information.
    ///
    /// This function is an implementation detail of `specialize()`.
    ///
    RefPtr<SpecializationInfo> _validateSpecializationArgs(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink)
    {
        if (argCount == 0)
            return nullptr;
        return _validateSpecializationArgsImpl(args, argCount, sink);
    }

    /// Specialize this component type given `specializationArgs`
    ///
    /// Any diagnostics will be reported to `sink`, which can be used
    /// to determine if the operation was successful. It is allowed
    /// for this operation to have a non-null return even when an
    /// error is ecnountered.
    ///
    RefPtr<ComponentType> specialize(
        SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        DiagnosticSink* sink);

    /// Invoke `visitor` on this component type, using the appropriate dynamic type.
    ///
    /// This function implements the "visitor pattern" for `ComponentType`.
    ///
    /// If the `specializationInfo` argument is non-null, it must be specialization
    /// information generated for this specific component type by `_validateSpecializationArgs`.
    /// In that case, appropriately-typed specialization information will be passed
    /// when invoking the `visitor`.
    ///
    virtual void acceptVisitor(
        ComponentTypeVisitor* visitor,
        SpecializationInfo* specializationInfo) = 0;

    /// Create a scope suitable for looking up names or parsing specialization arguments.
    ///
    /// This facility is only needed to support legacy APIs for string-based lookup
    /// and parsing via Slang reflection, and is not recommended for future APIs to use.
    ///
    Scope* _getOrCreateScopeForLegacyLookup(ASTBuilder* astBuilder);

protected:
    ComponentType(Linkage* linkage);

protected:
    Linkage* m_linkage;

    CompilerOptionSet m_optionSet;

    // Cache of target-specific programs for each target.
    Dictionary<TargetRequest*, RefPtr<TargetProgram>> m_targetPrograms;

    // Any types looked up dynamically using `getTypeFromString`
    //
    // TODO: Remove this. Type lookup should only be supported on `Module`s.
    //
    Dictionary<String, Type*> m_types;

    // Any decls looked up dynamically using `findDeclFromString`.
    Dictionary<String, Expr*> m_decls;

    Scope* m_lookupScope = nullptr;
    std::unique_ptr<Dictionary<String, IntVal*>> m_mapMangledNameToIntVal;

    Dictionary<Int, ComPtr<IArtifact>> m_targetArtifacts;
};

/// A component type built up from other component types.
class CompositeComponentType : public ComponentType
{
public:
    static RefPtr<ComponentType> create(
        Linkage* linkage,
        List<RefPtr<ComponentType>> const& childComponents);

    virtual void buildHash(DigestBuilder<SHA1>& builder) SLANG_OVERRIDE;

    List<RefPtr<ComponentType>> const& getChildComponents() { return m_childComponents; };
    Index getChildComponentCount() { return m_childComponents.getCount(); }
    RefPtr<ComponentType> getChildComponent(Index index) { return m_childComponents[index]; }

    Index getEntryPointCount() SLANG_OVERRIDE;
    RefPtr<EntryPoint> getEntryPoint(Index index) SLANG_OVERRIDE;
    String getEntryPointMangledName(Index index) SLANG_OVERRIDE;
    String getEntryPointNameOverride(Index index) SLANG_OVERRIDE;

    Index getShaderParamCount() SLANG_OVERRIDE;
    ShaderParamInfo getShaderParam(Index index) SLANG_OVERRIDE;

    SLANG_NO_THROW Index SLANG_MCALL getSpecializationParamCount() SLANG_OVERRIDE;
    SpecializationParam const& getSpecializationParam(Index index) SLANG_OVERRIDE;

    Index getRequirementCount() SLANG_OVERRIDE;
    RefPtr<ComponentType> getRequirement(Index index) SLANG_OVERRIDE;

    List<Module*> const& getModuleDependencies() SLANG_OVERRIDE;
    List<SourceFile*> const& getFileDependencies() SLANG_OVERRIDE;

    class CompositeSpecializationInfo : public SpecializationInfo
    {
    public:
        List<RefPtr<SpecializationInfo>> childInfos;
    };

protected:
    void acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
        SLANG_OVERRIDE;


    RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) SLANG_OVERRIDE;

public:
    CompositeComponentType(Linkage* linkage, List<RefPtr<ComponentType>> const& childComponents);

private:
    List<RefPtr<ComponentType>> m_childComponents;

    // The following arrays hold the concatenated entry points, parameters,
    // etc. from the child components. This approach allows for reasonably
    // fast (constant time) access through operations like `getShaderParam`,
    // but means that the memory usage of a composite is proportional to
    // the sum of the memory usage of the children, rather than being fixed
    // by the number of children (as it would be if we just stored
    // `m_childComponents`).
    //
    // TODO: We could conceivably build some O(numChildren) arrays that
    // support binary-search to provide logarithmic-time access to entry
    // points, parameters, etc. while giving a better overall memory usage.
    //
    List<EntryPoint*> m_entryPoints;
    List<String> m_entryPointMangledNames;
    List<String> m_entryPointNameOverrides;
    List<ShaderParamInfo> m_shaderParams;
    List<SpecializationParam> m_specializationParams;
    List<ComponentType*> m_requirements;

    ModuleDependencyList m_moduleDependencyList;
    FileDependencyList m_fileDependencyList;
};

/// A component type created by specializing another component type.
class SpecializedComponentType : public ComponentType
{
public:
    SpecializedComponentType(
        ComponentType* base,
        SpecializationInfo* specializationInfo,
        List<SpecializationArg> const& specializationArgs,
        DiagnosticSink* sink);

    virtual void buildHash(DigestBuilder<SHA1>& builer) SLANG_OVERRIDE;

    /// Get the base (unspecialized) component type that is being specialized.
    RefPtr<ComponentType> getBaseComponentType() { return m_base; }

    RefPtr<SpecializationInfo> getSpecializationInfo() { return m_specializationInfo; }

    /// Get the number of arguments supplied for existential type parameters.
    ///
    /// Note that the number of arguments may not match the number of parameters.
    /// In particular, an unspecialized entry point may have many parameters, but zero arguments.
    Index getSpecializationArgCount() { return m_specializationArgs.getCount(); }

    /// Get the existential type argument (type and witness table) at `index`.
    SpecializationArg const& getSpecializationArg(Index index)
    {
        return m_specializationArgs[index];
    }

    /// Get an array of all existential type arguments.
    SpecializationArg const* getSpecializationArgs() { return m_specializationArgs.getBuffer(); }

    Index getEntryPointCount() SLANG_OVERRIDE { return m_base->getEntryPointCount(); }
    RefPtr<EntryPoint> getEntryPoint(Index index) SLANG_OVERRIDE
    {
        return m_base->getEntryPoint(index);
    }
    String getEntryPointMangledName(Index index) SLANG_OVERRIDE;
    String getEntryPointNameOverride(Index index) SLANG_OVERRIDE;

    Index getShaderParamCount() SLANG_OVERRIDE { return m_base->getShaderParamCount(); }
    ShaderParamInfo getShaderParam(Index index) SLANG_OVERRIDE
    {
        return m_base->getShaderParam(index);
    }

    SLANG_NO_THROW Index SLANG_MCALL getSpecializationParamCount() SLANG_OVERRIDE { return 0; }
    SpecializationParam const& getSpecializationParam(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        static SpecializationParam dummy;
        return dummy;
    }

    Index getRequirementCount() SLANG_OVERRIDE;
    RefPtr<ComponentType> getRequirement(Index index) SLANG_OVERRIDE;

    List<Module*> const& getModuleDependencies() SLANG_OVERRIDE { return m_moduleDependencies; }
    List<SourceFile*> const& getFileDependencies() SLANG_OVERRIDE { return m_fileDependencies; }

    RefPtr<IRModule> getIRModule() { return m_irModule; }

    void acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
        SLANG_OVERRIDE;

protected:
    RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) SLANG_OVERRIDE
    {
        SLANG_UNUSED(args);
        SLANG_UNUSED(argCount);
        SLANG_UNUSED(sink);
        return nullptr;
    }

private:
    RefPtr<ComponentType> m_base;
    RefPtr<SpecializationInfo> m_specializationInfo;
    SpecializationArgs m_specializationArgs;
    RefPtr<IRModule> m_irModule;

    List<String> m_entryPointMangledNames;
    List<String> m_entryPointNameOverrides;

    List<Module*> m_moduleDependencies;
    List<SourceFile*> m_fileDependencies;
    List<RefPtr<ComponentType>> m_requirements;
};

class RenamedEntryPointComponentType : public ComponentType
{
public:
    using Super = ComponentType;

    RenamedEntryPointComponentType(ComponentType* base, String newName);

    ComponentType* getBase() { return m_base.Ptr(); }

    // Forward `IComponentType` methods

    SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() SLANG_OVERRIDE
    {
        return Super::getSession();
    }

    SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex, slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getLayout(targetIndex, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointCode(entryPointIndex, targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::specialize(
            specializationArgs,
            specializationArgCount,
            outSpecializedComponentType,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, slang::IComponentType** outEntryPoint) SLANG_OVERRIDE
    {
        return Super::renameEntryPoint(newName, outEntryPoint);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    link(slang::IComponentType** outLinkedComponentType, ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::link(outLinkedComponentType, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointHostCallable(
            entryPointIndex,
            targetIndex,
            outSharedLibrary,
            outDiagnostics);
    }

    List<Module*> const& getModuleDependencies() SLANG_OVERRIDE
    {
        return m_base->getModuleDependencies();
    }
    List<SourceFile*> const& getFileDependencies() SLANG_OVERRIDE
    {
        return m_base->getFileDependencies();
    }

    SLANG_NO_THROW Index SLANG_MCALL getSpecializationParamCount() SLANG_OVERRIDE
    {
        return m_base->getSpecializationParamCount();
    }

    SpecializationParam const& getSpecializationParam(Index index) SLANG_OVERRIDE
    {
        return m_base->getSpecializationParam(index);
    }

    Index getRequirementCount() SLANG_OVERRIDE { return m_base->getRequirementCount(); }
    RefPtr<ComponentType> getRequirement(Index index) SLANG_OVERRIDE
    {
        return m_base->getRequirement(index);
    }
    Index getEntryPointCount() SLANG_OVERRIDE { return m_base->getEntryPointCount(); }
    RefPtr<EntryPoint> getEntryPoint(Index index) SLANG_OVERRIDE
    {
        return m_base->getEntryPoint(index);
    }
    String getEntryPointMangledName(Index index) SLANG_OVERRIDE
    {
        return m_base->getEntryPointMangledName(index);
    }
    String getEntryPointNameOverride(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        SLANG_ASSERT(index == 0);
        return m_entryPointNameOverride;
    }

    Index getShaderParamCount() SLANG_OVERRIDE { return m_base->getShaderParamCount(); }
    ShaderParamInfo getShaderParam(Index index) SLANG_OVERRIDE
    {
        return m_base->getShaderParam(index);
    }

    void acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
        SLANG_OVERRIDE;

    virtual void buildHash(DigestBuilder<SHA1>& builder) SLANG_OVERRIDE;

private:
    RefPtr<ComponentType> m_base;
    String m_entryPointNameOverride;

protected:
    RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) SLANG_OVERRIDE
    {
        return m_base->_validateSpecializationArgsImpl(args, argCount, sink);
    }
};

/// Describes an entry point for the purposes of layout and code generation.
///
/// This class also tracks any generic arguments to the entry point,
/// in the case that it is a specialization of a generic entry point.
///
/// There is also a provision for creating a "dummy" entry point for
/// the purposes of pass-through compilation modes. Only the
/// `getName()` and `getProfile()` methods should be expected to
/// return useful data on pass-through entry points.
///
class EntryPoint : public ComponentType, public slang::IEntryPoint
{
    typedef ComponentType Super;

public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL

    ISlangUnknown* getInterface(const Guid& guid);


    // Forward `IComponentType` methods

    SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() SLANG_OVERRIDE
    {
        return Super::getSession();
    }

    SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex, slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getLayout(targetIndex, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointCode(entryPointIndex, targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetCode(targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointMetadata(
            entryPointIndex,
            targetIndex,
            outMetadata,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetMetadata(targetIndex, outMetadata, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) SLANG_OVERRIDE
    {
        return Super::getResultAsFileSystem(entryPointIndex, targetIndex, outFileSystem);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::specialize(
            specializationArgs,
            specializationArgCount,
            outSpecializedComponentType,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, slang::IComponentType** outEntryPoint) SLANG_OVERRIDE
    {
        return Super::renameEntryPoint(newName, outEntryPoint);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    link(slang::IComponentType** outLinkedComponentType, ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::link(outLinkedComponentType, outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        slang::IComponentType** outLinkedComponentType,
        uint32_t count,
        slang::CompilerOptionEntry* entries,
        ISlangBlob** outDiagnostics) override
    {
        return Super::linkWithOptions(outLinkedComponentType, count, entries, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointHostCallable(
            entryPointIndex,
            targetIndex,
            outSharedLibrary,
            outDiagnostics);
    }

    SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outHash) SLANG_OVERRIDE
    {
        return Super::getEntryPointHash(entryPointIndex, targetIndex, outHash);
    }

    virtual void buildHash(DigestBuilder<SHA1>& builder) SLANG_OVERRIDE;

    /// Create an entry point that refers to the given function.
    static RefPtr<EntryPoint> create(
        Linkage* linkage,
        DeclRef<FuncDecl> funcDeclRef,
        Profile profile);

    /// Get the function decl-ref, including any generic arguments.
    DeclRef<FuncDecl> getFuncDeclRef() { return m_funcDeclRef; }

    /// Get the function declaration (without generic arguments).
    FuncDecl* getFuncDecl() { return m_funcDeclRef.getDecl(); }

    /// Get the name of the entry point
    Name* getName() { return m_name; }

    /// Get the profile associated with the entry point
    ///
    /// Note: only the stage part of the profile is expected
    /// to contain useful data, but certain legacy code paths
    /// allow for "shader model" information to come via this path.
    ///
    Profile getProfile() { return m_profile; }

    /// Get the stage that the entry point is for.
    Stage getStage() { return m_profile.getStage(); }

    /// Get the module that contains the entry point.
    Module* getModule();

    /// Get a list of modules that this entry point depends on.
    ///
    /// This will include the module that defines the entry point (see `getModule()`),
    /// but may also include modules that are required by its generic type arguments.
    ///
    List<Module*> const& getModuleDependencies()
        SLANG_OVERRIDE; // { return getModule()->getModuleDependencies(); }
    List<SourceFile*> const& getFileDependencies()
        SLANG_OVERRIDE; // { return getModule()->getFileDependencies(); }

    /// Create a dummy `EntryPoint` that is only usable for pass-through compilation.
    static RefPtr<EntryPoint> createDummyForPassThrough(
        Linkage* linkage,
        Name* name,
        Profile profile);

    /// Create a dummy `EntryPoint` that stands in for a serialized entry point
    static RefPtr<EntryPoint> createDummyForDeserialize(
        Linkage* linkage,
        Name* name,
        Profile profile,
        String mangledName);

    /// Get the number of existential type parameters for the entry point.
    SLANG_NO_THROW Index SLANG_MCALL getSpecializationParamCount() SLANG_OVERRIDE;

    /// Get the existential type parameter at `index`.
    SpecializationParam const& getSpecializationParam(Index index) SLANG_OVERRIDE;

    Index getRequirementCount() SLANG_OVERRIDE;
    RefPtr<ComponentType> getRequirement(Index index) SLANG_OVERRIDE;

    SpecializationParams const& getExistentialSpecializationParams()
    {
        return m_existentialSpecializationParams;
    }

    Index getGenericSpecializationParamCount() { return m_genericSpecializationParams.getCount(); }
    Index getExistentialSpecializationParamCount()
    {
        return m_existentialSpecializationParams.getCount();
    }

    /// Get an array of all entry-point shader parameters.
    List<ShaderParamInfo> const& getShaderParams() { return m_shaderParams; }

    Index getEntryPointCount() SLANG_OVERRIDE { return 1; };
    RefPtr<EntryPoint> getEntryPoint(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return this;
    }
    String getEntryPointMangledName(Index index) SLANG_OVERRIDE;
    String getEntryPointNameOverride(Index index) SLANG_OVERRIDE;

    Index getShaderParamCount() SLANG_OVERRIDE { return 0; }
    ShaderParamInfo getShaderParam(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return ShaderParamInfo();
    }

    class EntryPointSpecializationInfo : public SpecializationInfo
    {
    public:
        DeclRef<FuncDecl> specializedFuncDeclRef;
        List<ExpandedSpecializationArg> existentialSpecializationArgs;
    };

    SLANG_NO_THROW slang::FunctionReflection* SLANG_MCALL getFunctionReflection() SLANG_OVERRIDE
    {
        return (slang::FunctionReflection*)m_funcDeclRef.declRefBase;
    }

protected:
    void acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
        SLANG_OVERRIDE;

    RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) SLANG_OVERRIDE;

private:
    EntryPoint(Linkage* linkage, Name* name, Profile profile, DeclRef<FuncDecl> funcDeclRef);

    void _collectGenericSpecializationParamsRec(Decl* decl);
    void _collectShaderParams();

    // The name of the entry point function (e.g., `main`)
    //
    Name* m_name = nullptr;

    // The declaration of the entry-point function itself.
    //
    DeclRef<FuncDecl> m_funcDeclRef;

    /// The mangled name of the entry point function
    String m_mangledName;

    SpecializationParams m_genericSpecializationParams;
    SpecializationParams m_existentialSpecializationParams;

    /// Information about entry-point parameters
    List<ShaderParamInfo> m_shaderParams;

    // The profile that the entry point will be compiled for
    // (this is a combination of the target stage, and also
    // a feature level that sets capabilities)
    //
    // Note: the profile-version part of this should probably
    // be moving towards deprecation, in favor of the version
    // information (e.g., "Shader Model 5.1") always coming
    // from the target, while the stage part is all that is
    // intrinsic to the entry point.
    //
    Profile m_profile;
};

class TypeConformance : public ComponentType, public slang::ITypeConformance
{
    typedef ComponentType Super;

public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL

    ISlangUnknown* getInterface(const Guid& guid);

    TypeConformance(
        Linkage* linkage,
        SubtypeWitness* witness,
        Int confomrmanceIdOverride,
        DiagnosticSink* sink);

    // Forward `IComponentType` methods

    SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() SLANG_OVERRIDE
    {
        return Super::getSession();
    }

    SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex, slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getLayout(targetIndex, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointCode(entryPointIndex, targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetCode(targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointMetadata(
            entryPointIndex,
            targetIndex,
            outMetadata,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetMetadata(targetIndex, outMetadata, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) SLANG_OVERRIDE
    {
        return Super::getResultAsFileSystem(entryPointIndex, targetIndex, outFileSystem);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::specialize(
            specializationArgs,
            specializationArgCount,
            outSpecializedComponentType,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, slang::IComponentType** outEntryPoint) SLANG_OVERRIDE
    {
        return Super::renameEntryPoint(newName, outEntryPoint);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    link(slang::IComponentType** outLinkedComponentType, ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::link(outLinkedComponentType, outDiagnostics);
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        slang::IComponentType** outLinkedComponentType,
        uint32_t count,
        slang::CompilerOptionEntry* entries,
        ISlangBlob** outDiagnostics) override
    {
        return Super::linkWithOptions(outLinkedComponentType, count, entries, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointHostCallable(
            entryPointIndex,
            targetIndex,
            outSharedLibrary,
            outDiagnostics);
    }

    SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outHash) SLANG_OVERRIDE
    {
        return Super::getEntryPointHash(entryPointIndex, targetIndex, outHash);
    }

    virtual void buildHash(DigestBuilder<SHA1>& builder) SLANG_OVERRIDE;

    List<Module*> const& getModuleDependencies() SLANG_OVERRIDE;
    List<SourceFile*> const& getFileDependencies() SLANG_OVERRIDE;

    SLANG_NO_THROW Index SLANG_MCALL getSpecializationParamCount() SLANG_OVERRIDE { return 0; }

    /// Get the existential type parameter at `index`.
    SpecializationParam const& getSpecializationParam(Index /*index*/) SLANG_OVERRIDE
    {
        static SpecializationParam emptyParam;
        return emptyParam;
    }

    Index getRequirementCount() SLANG_OVERRIDE;
    RefPtr<ComponentType> getRequirement(Index index) SLANG_OVERRIDE;
    Index getEntryPointCount() SLANG_OVERRIDE { return 0; };
    RefPtr<EntryPoint> getEntryPoint(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return nullptr;
    }
    String getEntryPointMangledName(Index /*index*/) SLANG_OVERRIDE { return ""; }
    String getEntryPointNameOverride(Index /*index*/) SLANG_OVERRIDE { return ""; }

    Index getShaderParamCount() SLANG_OVERRIDE { return 0; }
    ShaderParamInfo getShaderParam(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return ShaderParamInfo();
    }

    SubtypeWitness* getSubtypeWitness() { return m_subtypeWitness; }
    IRModule* getIRModule() { return m_irModule.Ptr(); }

protected:
    void acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
        SLANG_OVERRIDE;

    RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) SLANG_OVERRIDE;

private:
    SubtypeWitness* m_subtypeWitness;
    ModuleDependencyList m_moduleDependencyList;
    FileDependencyList m_fileDependencyList;
    List<RefPtr<Module>> m_requirements;
    HashSet<Module*> m_requirementSet;
    RefPtr<IRModule> m_irModule;
    Int m_conformanceIdOverride;
    void addDepedencyFromWitness(SubtypeWitness* witness);
};

enum class PassThroughMode : SlangPassThroughIntegral
{
    None = SLANG_PASS_THROUGH_NONE,                  ///< don't pass through: use Slang compiler
    Fxc = SLANG_PASS_THROUGH_FXC,                    ///< pass through HLSL to `D3DCompile` API
    Dxc = SLANG_PASS_THROUGH_DXC,                    ///< pass through HLSL to `IDxcCompiler` API
    Glslang = SLANG_PASS_THROUGH_GLSLANG,            ///< pass through GLSL to `glslang` library
    SpirvDis = SLANG_PASS_THROUGH_SPIRV_DIS,         ///< pass through spirv-dis
    Clang = SLANG_PASS_THROUGH_CLANG,                ///< Pass through clang compiler
    VisualStudio = SLANG_PASS_THROUGH_VISUAL_STUDIO, ///< Visual studio compiler
    Gcc = SLANG_PASS_THROUGH_GCC,                    ///< Gcc compiler
    GenericCCpp = SLANG_PASS_THROUGH_GENERIC_C_CPP,  ///< Generic C/C++ compiler
    NVRTC = SLANG_PASS_THROUGH_NVRTC,                ///< NVRTC CUDA compiler
    LLVM = SLANG_PASS_THROUGH_LLVM,                  ///< LLVM 'compiler'
    SpirvOpt = SLANG_PASS_THROUGH_SPIRV_OPT,         ///< pass thorugh spirv to spirv-opt
    MetalC = SLANG_PASS_THROUGH_METAL,
    Tint = SLANG_PASS_THROUGH_TINT,            ///< pass through spirv to Tint API
    SpirvLink = SLANG_PASS_THROUGH_SPIRV_LINK, ///< pass through spirv to spirv-link
    CountOf = SLANG_PASS_THROUGH_COUNT_OF,
};
void printDiagnosticArg(StringBuilder& sb, PassThroughMode val);

class SourceFile;

/// A module of code that has been compiled through the front-end
///
/// A module comprises all the code from one translation unit (which
/// may span multiple Slang source files), and provides access
/// to both the AST and IR representations of that code.
///
class Module : public ComponentType, public slang::IModule
{
    typedef ComponentType Super;

public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL

    ISlangUnknown* getInterface(const Guid& guid);


    // Forward `IComponentType` methods

    SLANG_NO_THROW slang::ISession* SLANG_MCALL getSession() SLANG_OVERRIDE
    {
        return Super::getSession();
    }

    SLANG_NO_THROW slang::ProgramLayout* SLANG_MCALL
    getLayout(SlangInt targetIndex, slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getLayout(targetIndex, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCode(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointCode(entryPointIndex, targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetCode(
        SlangInt targetIndex,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetCode(targetIndex, outCode, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getResultAsFileSystem(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        ISlangMutableFileSystem** outFileSystem) SLANG_OVERRIDE
    {
        return Super::getResultAsFileSystem(entryPointIndex, targetIndex, outFileSystem);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL specialize(
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        slang::IComponentType** outSpecializedComponentType,
        ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::specialize(
            specializationArgs,
            specializationArgCount,
            outSpecializedComponentType,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    renameEntryPoint(const char* newName, slang::IComponentType** outEntryPoint) SLANG_OVERRIDE
    {
        return Super::renameEntryPoint(newName, outEntryPoint);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    link(slang::IComponentType** outLinkedComponentType, ISlangBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::link(outLinkedComponentType, outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointHostCallable(
            entryPointIndex,
            targetIndex,
            outSharedLibrary,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    findEntryPointByName(char const* name, slang::IEntryPoint** outEntryPoint) SLANG_OVERRIDE
    {
        if (outEntryPoint == nullptr)
        {
            return SLANG_E_INVALID_ARG;
        }
        SLANG_AST_BUILDER_RAII(m_astBuilder);
        ComPtr<slang::IEntryPoint> entryPoint(findEntryPointByName(UnownedStringSlice(name)));
        if ((!entryPoint))
            return SLANG_FAIL;

        *outEntryPoint = entryPoint.detach();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL findAndCheckEntryPoint(
        char const* name,
        SlangStage stage,
        slang::IEntryPoint** outEntryPoint,
        ISlangBlob** outDiagnostics) override
    {
        if (outEntryPoint == nullptr)
        {
            return SLANG_E_INVALID_ARG;
        }
        ComPtr<slang::IEntryPoint> entryPoint(
            findAndCheckEntryPoint(UnownedStringSlice(name), stage, outDiagnostics));
        if ((!entryPoint))
            return SLANG_FAIL;

        *outEntryPoint = entryPoint.detach();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDefinedEntryPointCount() override
    {
        return (SlangInt32)m_entryPoints.getCount();
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getDefinedEntryPoint(SlangInt32 index, slang::IEntryPoint** outEntryPoint) override
    {
        if (index < 0 || index >= m_entryPoints.getCount())
            return SLANG_E_INVALID_ARG;

        if (outEntryPoint == nullptr)
        {
            return SLANG_E_INVALID_ARG;
        }

        ComPtr<slang::IEntryPoint> entryPoint(m_entryPoints[index].Ptr());
        *outEntryPoint = entryPoint.detach();
        return SLANG_OK;
    }

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL linkWithOptions(
        slang::IComponentType** outLinkedComponentType,
        uint32_t count,
        slang::CompilerOptionEntry* entries,
        ISlangBlob** outDiagnostics) override
    {
        return Super::linkWithOptions(outLinkedComponentType, count, entries, outDiagnostics);
    }
    //

    SLANG_NO_THROW void SLANG_MCALL getEntryPointHash(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IBlob** outHash) SLANG_OVERRIDE
    {
        return Super::getEntryPointHash(entryPointIndex, targetIndex, outHash);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointMetadata(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getEntryPointMetadata(
            entryPointIndex,
            targetIndex,
            outMetadata,
            outDiagnostics);
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getTargetMetadata(
        SlangInt targetIndex,
        slang::IMetadata** outMetadata,
        slang::IBlob** outDiagnostics) SLANG_OVERRIDE
    {
        return Super::getTargetMetadata(targetIndex, outMetadata, outDiagnostics);
    }

    /// Get a serialized representation of the checked module.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    serialize(ISlangBlob** outSerializedBlob) override;

    /// Write the serialized representation of this module to a file.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL writeToFile(char const* fileName) override;

    /// Get the name of the module.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getName() override;

    /// Get the path of the module.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getFilePath() override;

    /// Get the unique identity of the module.
    virtual SLANG_NO_THROW const char* SLANG_MCALL getUniqueIdentity() override;

    /// Get the number of dependency files that this module depends on.
    /// This includes both the explicit source files, as well as any
    /// additional files that were transitively referenced (e.g., via
    /// a `#include` directive).
    virtual SLANG_NO_THROW SlangInt32 SLANG_MCALL getDependencyFileCount() override;

    /// Get the path to a file this module depends on.
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(SlangInt32 index) override;


    // IModulePrecompileService_Experimental
    /// Precompile TU to target language
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    precompileForTarget(SlangCompileTarget target, slang::IBlob** outDiagnostics) override;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getPrecompiledTargetCode(
        SlangCompileTarget target,
        slang::IBlob** outCode,
        slang::IBlob** outDiagnostics = nullptr) override;

    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getModuleDependencyCount() SLANG_OVERRIDE;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getModuleDependency(
        SlangInt dependencyIndex,
        slang::IModule** outModule,
        slang::IBlob** outDiagnostics = nullptr) SLANG_OVERRIDE;

    virtual void buildHash(DigestBuilder<SHA1>& builder) SLANG_OVERRIDE;

    virtual SLANG_NO_THROW slang::DeclReflection* SLANG_MCALL getModuleReflection() SLANG_OVERRIDE;

    void setDigest(SHA1::Digest const& digest) { m_digest = digest; }
    SHA1::Digest computeDigest();

    /// Create a module (initially empty).
    Module(Linkage* linkage, ASTBuilder* astBuilder = nullptr);

    /// Get the AST for the module (if it has been parsed)
    ModuleDecl* getModuleDecl() { return m_moduleDecl; }

    /// The the IR for the module (if it has been generated)
    IRModule* getIRModule() { return m_irModule; }

    /// Get the list of other modules this module depends on
    List<Module*> const& getModuleDependencyList()
    {
        return m_moduleDependencyList.getModuleList();
    }

    /// Get the list of files this module depends on
    List<SourceFile*> const& getFileDependencyList() { return m_fileDependencyList.getFileList(); }

    /// Register a module that this module depends on
    void addModuleDependency(Module* module);

    /// Register a source file that this module depends on
    void addFileDependency(SourceFile* sourceFile);

    void clearFileDependency() { m_fileDependencyList.clear(); }
    /// Set the AST for this module.
    ///
    /// This should only be called once, during creation of the module.
    ///
    void setModuleDecl(ModuleDecl* moduleDecl); // { m_moduleDecl = moduleDecl; }

    void setName(String name);
    void setName(Name* name) { m_name = name; }
    Name* getNameObj() { return m_name; }

    void setPathInfo(PathInfo pathInfo) { m_pathInfo = pathInfo; }

    /// Set the IR for this module.
    ///
    /// This should only be called once, during creation of the module.
    ///
    void setIRModule(IRModule* irModule) { m_irModule = irModule; }

    Index getEntryPointCount() SLANG_OVERRIDE { return 0; }
    RefPtr<EntryPoint> getEntryPoint(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return nullptr;
    }
    String getEntryPointMangledName(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return String();
    }
    String getEntryPointNameOverride(Index index) SLANG_OVERRIDE
    {
        SLANG_UNUSED(index);
        return String();
    }

    Index getShaderParamCount() SLANG_OVERRIDE { return m_shaderParams.getCount(); }
    ShaderParamInfo getShaderParam(Index index) SLANG_OVERRIDE { return m_shaderParams[index]; }

    SLANG_NO_THROW Index SLANG_MCALL getSpecializationParamCount() SLANG_OVERRIDE
    {
        return m_specializationParams.getCount();
    }
    SpecializationParam const& getSpecializationParam(Index index) SLANG_OVERRIDE
    {
        return m_specializationParams[index];
    }

    Index getRequirementCount() SLANG_OVERRIDE;
    RefPtr<ComponentType> getRequirement(Index index) SLANG_OVERRIDE;

    List<Module*> const& getModuleDependencies() SLANG_OVERRIDE
    {
        return m_moduleDependencyList.getModuleList();
    }
    List<SourceFile*> const& getFileDependencies() SLANG_OVERRIDE
    {
        return m_fileDependencyList.getFileList();
    }

    /// Given a mangled name finds the exported NodeBase associated with this module.
    /// If not found returns nullptr.
    NodeBase* findExportFromMangledName(const UnownedStringSlice& slice);

    /// Get the ASTBuilder
    ASTBuilder* getASTBuilder() { return m_astBuilder; }

    /// Collect information on the shader parameters of the module.
    ///
    /// This method should only be called once, after the core
    /// structured of the module (its AST and IR) have been created,
    /// and before any of the `ComponentType` APIs are used.
    ///
    /// TODO: We might eventually consider a non-stateful approach
    /// to constructing a `Module`.
    ///
    void _collectShaderParams();

    void _discoverEntryPoints(DiagnosticSink* sink, const List<RefPtr<TargetRequest>>& targets);
    void _discoverEntryPointsImpl(
        ContainerDecl* containerDecl,
        DiagnosticSink* sink,
        const List<RefPtr<TargetRequest>>& targets);


    class ModuleSpecializationInfo : public SpecializationInfo
    {
    public:
        struct GenericArgInfo
        {
            Decl* paramDecl = nullptr;
            Val* argVal = nullptr;
        };

        List<GenericArgInfo> genericArgs;
        List<ExpandedSpecializationArg> existentialArgs;
    };

    RefPtr<EntryPoint> findEntryPointByName(UnownedStringSlice const& name);
    RefPtr<EntryPoint> findAndCheckEntryPoint(
        UnownedStringSlice const& name,
        SlangStage stage,
        ISlangBlob** outDiagnostics);

    List<RefPtr<EntryPoint>>& getEntryPoints() { return m_entryPoints; }
    void _addEntryPoint(EntryPoint* entryPoint);
    void _processFindDeclsExportSymbolsRec(Decl* decl);

    // Gets the files that has been included into the module.
    Dictionary<SourceFile*, FileDecl*>& getIncludedSourceFileMap()
    {
        return m_mapSourceFileToFileDecl;
    }

protected:
    void acceptVisitor(ComponentTypeVisitor* visitor, SpecializationInfo* specializationInfo)
        SLANG_OVERRIDE;

    RefPtr<SpecializationInfo> _validateSpecializationArgsImpl(
        SpecializationArg const* args,
        Index argCount,
        DiagnosticSink* sink) SLANG_OVERRIDE;

private:
    Name* m_name = nullptr;
    PathInfo m_pathInfo;

    // The AST for the module
    ModuleDecl* m_moduleDecl = nullptr;

    // The IR for the module
    RefPtr<IRModule> m_irModule = nullptr;

    List<ShaderParamInfo> m_shaderParams;
    SpecializationParams m_specializationParams;

    List<Module*> m_requirements;

    // A digest that uniquely identifies the contents of the module.
    SHA1::Digest m_digest;

    // List of modules this module depends on
    ModuleDependencyList m_moduleDependencyList;

    // List of source files this module depends on
    FileDependencyList m_fileDependencyList;

    // Entry points that were defined in this module
    //
    // Note: the entry point defined in the module are *not*
    // part of the memory image/layout of the module when
    // it is considered as an IComponentType. This can be
    // a bit confusing, but if all the entry points in the
    // module were automatically linked into the component
    // type, we'd need a way to access just the global
    // scope of the module without the entry points, in
    // case we wanted to link a single entry point against
    // the global scope. The `Module` type provides exactly
    // that "module without its entry points" unit of
    // granularity for linking.
    //
    // This list only exists for lookup purposes, so that
    // the user can find an existing entry-point function
    // that was defined as part of the module.
    //
    List<RefPtr<EntryPoint>> m_entryPoints;

    // The builder that owns all of the AST nodes from parsing the source of
    // this module.
    RefPtr<ASTBuilder> m_astBuilder;

    // Holds map of exported mangled names to symbols. m_mangledExportPool maps names to indices,
    // and m_mangledExportSymbols holds the NodeBase* values for each index.
    StringSlicePool m_mangledExportPool;
    List<NodeBase*> m_mangledExportSymbols;

    // Source files that have been pulled into the module with `__include`.
    Dictionary<SourceFile*, FileDecl*> m_mapSourceFileToFileDecl;

public:
    SLANG_NO_THROW SlangResult SLANG_MCALL disassemble(slang::IBlob** outDisassembledBlob) override
    {
        if (!outDisassembledBlob)
            return SLANG_E_INVALID_ARG;
        String disassembly;
        this->getIRModule()->getModuleInst()->dump(disassembly);
        auto blob = StringUtil::createStringBlob(disassembly);
        *outDisassembledBlob = blob.detach();
        return SLANG_OK;
    }
};
typedef Module LoadedModule;

/// A request for the front-end to compile a translation unit.
class TranslationUnitRequest : public RefObject
{
public:
    TranslationUnitRequest(FrontEndCompileRequest* compileRequest);
    TranslationUnitRequest(FrontEndCompileRequest* compileRequest, Module* m);

    // The parent compile request
    FrontEndCompileRequest* compileRequest = nullptr;

    // The language in which the source file(s)
    // are assumed to be written
    SourceLanguage sourceLanguage = SourceLanguage::Unknown;

    /// Makes any source artifact available as a SourceFile.
    /// If successful any of the source artifacts will be represented by the same index
    /// of sourceArtifacts
    SlangResult requireSourceFiles();

    /// Get the source files.
    /// Since lazily evaluated requires calling requireSourceFiles to know it's in sync
    /// with sourceArtifacts.
    List<SourceFile*> const& getSourceFiles();

    /// Get the source artifacts associated
    const List<ComPtr<IArtifact>>& getSourceArtifacts() const { return m_sourceArtifacts; }

    /// Clear all of the source
    void clearSource()
    {
        m_sourceArtifacts.clear();
        m_sourceFiles.clear();
    }

    /// Add a source artifact
    void addSourceArtifact(IArtifact* sourceArtifact);

    /// Add both the artifact and the sourceFile.
    void addSource(IArtifact* sourceArtifact, SourceFile* sourceFile);

    // The entry points associated with this translation unit
    List<RefPtr<EntryPoint>> const& getEntryPoints() { return module->getEntryPoints(); }

    void _addEntryPoint(EntryPoint* entryPoint) { module->_addEntryPoint(entryPoint); }

    // Preprocessor definitions to use for this translation unit only
    // (whereas the ones on `compileRequest` will be shared)
    Dictionary<String, String> preprocessorDefinitions;

    /// The name that will be used for the module this translation unit produces.
    Name* moduleName = nullptr;

    /// Result of compiling this translation unit (a module)
    RefPtr<Module> module;

    bool isChecked = false;

    Module* getModule() { return module; }
    ModuleDecl* getModuleDecl() { return module->getModuleDecl(); }

    Session* getSession();
    NamePool* getNamePool();
    SourceManager* getSourceManager();

    Scope* getLanguageScope();

    Dictionary<String, String> getCombinedPreprocessorDefinitions();

    void setModuleName(Name* name)
    {
        moduleName = name;
        if (module)
            module->setName(name);
    }

protected:
    void _addSourceFile(SourceFile* sourceFile);
    /* Given an artifact, find a PathInfo.
    If no PathInfo can be found will return an unknown PathInfo */
    PathInfo _findSourcePathInfo(IArtifact* artifact);

    List<ComPtr<IArtifact>> m_sourceArtifacts;
    // The source file(s) that will be compiled to form this translation unit
    //
    // Usually, for HLSL or GLSL there will be only one file.
    // NOTE! This member is generated lazily from m_sourceArtifacts
    // it is *necessary* to call requireSourceFiles to ensure it's in sync.
    List<SourceFile*> m_sourceFiles;
};

enum class FloatingPointMode : SlangFloatingPointModeIntegral
{
    Default = SLANG_FLOATING_POINT_MODE_DEFAULT,
    Fast = SLANG_FLOATING_POINT_MODE_FAST,
    Precise = SLANG_FLOATING_POINT_MODE_PRECISE,
};

enum class WriterChannel : SlangWriterChannelIntegral
{
    Diagnostic = SLANG_WRITER_CHANNEL_DIAGNOSTIC,
    StdOutput = SLANG_WRITER_CHANNEL_STD_OUTPUT,
    StdError = SLANG_WRITER_CHANNEL_STD_ERROR,
    CountOf = SLANG_WRITER_CHANNEL_COUNT_OF,
};

enum class WriterMode : SlangWriterModeIntegral
{
    Text = SLANG_WRITER_MODE_TEXT,
    Binary = SLANG_WRITER_MODE_BINARY,
};

class TargetRequest;

/// Are we generating code for a D3D API?
bool isD3DTarget(TargetRequest* targetReq);

// Are we generating code for Metal?
bool isMetalTarget(TargetRequest* targetReq);

/// Are we generating code for a Khronos API (OpenGL or Vulkan)?
bool isKhronosTarget(TargetRequest* targetReq);
bool isKhronosTarget(CodeGenTarget target);

/// Are we generating code for a CUDA API (CUDA / OptiX)?
bool isCUDATarget(TargetRequest* targetReq);

// Are we generating code for a CPU target
bool isCPUTarget(TargetRequest* targetReq);

/// Are we generating code for the WebGPU API?
bool isWGPUTarget(TargetRequest* targetReq);
bool isWGPUTarget(CodeGenTarget target);

/// A request to generate output in some target format.
class TargetRequest : public RefObject
{
public:
    TargetRequest(Linkage* linkage, CodeGenTarget format);

    TargetRequest(const TargetRequest& other);

    Linkage* getLinkage() { return linkage; }

    Session* getSession();

    CodeGenTarget getTarget()
    {
        return optionSet.getEnumOption<CodeGenTarget>(CompilerOptionName::Target);
    }

    // TypeLayouts created on the fly by reflection API
    struct TypeLayoutKey
    {
        Type* type;
        slang::LayoutRules rules;
        HashCode getHashCode() const
        {
            Hasher hasher;
            hasher.hashValue(type);
            hasher.hashValue(rules);
            return hasher.getResult();
        }
        bool operator==(TypeLayoutKey other) const
        {
            return type == other.type && rules == other.rules;
        }
    };
    Dictionary<TypeLayoutKey, RefPtr<TypeLayout>> typeLayouts;

    Dictionary<TypeLayoutKey, RefPtr<TypeLayout>>& getTypeLayouts() { return typeLayouts; }

    TypeLayout* getTypeLayout(Type* type, slang::LayoutRules rules);

    CompilerOptionSet& getOptionSet() { return optionSet; }

    CapabilitySet getTargetCaps();

    void setTargetCaps(CapabilitySet capSet);

    HLSLToVulkanLayoutOptions* getHLSLToVulkanLayoutOptions();

private:
    Linkage* linkage = nullptr;
    CompilerOptionSet optionSet;
    CapabilitySet cookedCapabilities;
    RefPtr<HLSLToVulkanLayoutOptions> hlslToVulkanOptions;
};

/// Given a target request returns which (if any) intermediate source language is required
/// to produce it.
///
/// If no intermediate source language is required, will return SourceLanguage::Unknown
SourceLanguage getIntermediateSourceLanguageForTarget(TargetProgram* req);

/// Are resource types "bindless" (implemented as ordinary data) on the given `target`?
bool areResourceTypesBindlessOnTarget(TargetRequest* target);

// Compute the "effective" profile to use when outputting the given entry point
// for the chosen code-generation target.
//
// The stage of the effective profile will always come from the entry point, while
// the profile version (aka "shader model") will be computed as follows:
//
// - If the entry point and target belong to the same profile family, then take
//   the latest version between the two (e.g., if the entry point specified `ps_5_1`
//   and the target specifies `sm_5_0` then use `sm_5_1` as the version).
//
// - If the entry point and target disagree on the profile family, always use the
//   profile family and version from the target.
//
Profile getEffectiveProfile(EntryPoint* entryPoint, TargetRequest* target);


/// Given a target returns the required downstream compiler
PassThroughMode getDownstreamCompilerRequiredForTarget(CodeGenTarget target);
/// Given a target returns a downstream compiler the prelude should be taken from.
SourceLanguage getDefaultSourceLanguageForDownstreamCompiler(PassThroughMode compiler);

/// Get the build tag string
const char* getBuildTagString();

struct TypeCheckingCache;

struct ContainerTypeKey
{
    slang::TypeReflection* elementType;
    slang::ContainerType containerType;
    bool operator==(ContainerTypeKey other) const
    {
        return elementType == other.elementType && containerType == other.containerType;
    }
    Slang::HashCode getHashCode() const
    {
        return Slang::combineHash(
            Slang::getHashCode(elementType),
            Slang::getHashCode(containerType));
    }
};

/// A dictionary of modules to be considered when resolving `import`s,
/// beyond those that would normally be found through a `Linkage`.
///
/// Checking of an `import` declaration will bottleneck through
/// `Linkage::findOrImportModule`, which would usually just check for
/// any module that had been previously loaded into the same `Linkage`
/// (e.g., by a call to `Linkage::loadModule()`).
///
/// In the case where compilation is being done through an
/// explicit `FrontEndCompileRequest` or `EndToEndCompileRequest`,
/// the modules being compiled by that request do not get added to
/// the surrounding `Linkage`.
///
/// There is a corner case when an explicit compile request has
/// multiple `TranslationUnitRequest`s, because the user (reasonably)
/// expects that if they compile `A.slang` and `B.slang` as two
/// distinct translation units in the same compile request, then
/// an `import B` inside of `A.slang` should resolve to reference
/// the code of `B.slang`. But because neither `A` nor `B` gets
/// added to the `Linkage`, and the `Linkage` is what usually
/// determines what is or isn't loaded, that intuition will
/// be wrong, without a bit of help.
///
/// The `LoadedModuleDictionary` is thus filled in by a
/// `FrontEndCompileRequest` to collect the modules it is compiling,
/// so that they can cross-reference one another (albeit with
/// a current implementation restriction that modules in the
/// request can only `import` those earlier in the request...).
///
/// The dictionary then gets passed around between nearly all of
/// the operations that deal with loading modules, to make sure
/// that they can detect a previously loaded module.
///
typedef Dictionary<Name*, Module*> LoadedModuleDictionary;

enum ModuleBlobType
{
    Source,
    IR
};

/// A context for loading and re-using code modules.
class Linkage : public RefObject, public slang::ISession
{
public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL

    CompilerOptionSet m_optionSet;

    ISlangUnknown* getInterface(const Guid& guid);

    SLANG_NO_THROW slang::IGlobalSession* SLANG_MCALL getGlobalSession() override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL
    loadModule(const char* moduleName, slang::IBlob** outDiagnostics = nullptr) override;
    slang::IModule* loadModuleFromBlob(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        ModuleBlobType blobType,
        slang::IBlob** outDiagnostics = nullptr);
    SLANG_NO_THROW slang::IModule* SLANG_MCALL loadModuleFromIRBlob(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL loadModuleFromSource(
        const char* moduleName,
        const char* path,
        slang::IBlob* source,
        slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::IModule* SLANG_MCALL loadModuleFromSourceString(
        const char* moduleName,
        const char* path,
        const char* string,
        slang::IBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL createCompositeComponentType(
        slang::IComponentType* const* componentTypes,
        SlangInt componentTypeCount,
        slang::IComponentType** outCompositeComponentType,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL specializeType(
        slang::TypeReflection* type,
        slang::SpecializationArg const* specializationArgs,
        SlangInt specializationArgCount,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeLayoutReflection* SLANG_MCALL getTypeLayout(
        slang::TypeReflection* type,
        SlangInt targetIndex = 0,
        slang::LayoutRules rules = slang::LayoutRules::Default,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL getContainerType(
        slang::TypeReflection* elementType,
        slang::ContainerType containerType,
        ISlangBlob** outDiagnostics = nullptr) override;
    SLANG_NO_THROW slang::TypeReflection* SLANG_MCALL getDynamicType() override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    getTypeRTTIMangledName(slang::TypeReflection* type, ISlangBlob** outNameBlob) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessMangledName(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        ISlangBlob** outNameBlob) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL getTypeConformanceWitnessSequentialID(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        uint32_t* outId) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL createTypeConformanceComponentType(
        slang::TypeReflection* type,
        slang::TypeReflection* interfaceType,
        slang::ITypeConformance** outConformance,
        SlangInt conformanceIdOverride,
        ISlangBlob** outDiagnostics) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createCompileRequest(SlangCompileRequest** outCompileRequest) override;
    virtual SLANG_NO_THROW SlangInt SLANG_MCALL getLoadedModuleCount() override;
    virtual SLANG_NO_THROW slang::IModule* SLANG_MCALL getLoadedModule(SlangInt index) override;
    virtual SLANG_NO_THROW bool SLANG_MCALL
    isBinaryModuleUpToDate(const char* modulePath, slang::IBlob* binaryModuleBlob) override;

    // Updates the supplied builder with linkage-related information, which includes preprocessor
    // defines, the compiler version, and other compiler options. This is then merged with the hash
    // produced for the program to produce a key that can be used with the shader cache.
    void buildHash(DigestBuilder<SHA1>& builder, SlangInt targetIndex = -1);

    void addTarget(slang::TargetDesc const& desc);
    SlangResult addSearchPath(char const* path);
    SlangResult addPreprocessorDefine(char const* name, char const* value);
    SlangResult setMatrixLayoutMode(SlangMatrixLayoutMode mode);
    /// Create an initially-empty linkage
    Linkage(Session* session, ASTBuilder* astBuilder, Linkage* builtinLinkage);

    /// Dtor
    ~Linkage();

    bool isInLanguageServer()
    {
        return contentAssistInfo.checkingMode != ContentAssistCheckingMode::None;
    }

    /// Get the parent session for this linkage
    Session* getSessionImpl() { return m_session; }

    // Information on the targets we are being asked to
    // generate code for.
    List<RefPtr<TargetRequest>> targets;

    // Directories to search for `#include` files or `import`ed modules
    SearchDirectoryList& getSearchDirectories();

    // Source manager to help track files loaded
    SourceManager m_defaultSourceManager;
    SourceManager* m_sourceManager = nullptr;
    RefPtr<CommandLineContext> m_cmdLineContext;

    // Name pool for looking up names
    NamePool namePool;

    NamePool* getNamePool() { return &namePool; }

    ASTBuilder* getASTBuilder() { return m_astBuilder; }

    RefPtr<ASTBuilder> m_astBuilder;

    // Cache for container types.
    Dictionary<ContainerTypeKey, Type*> m_containerTypes;

    // cache used by type checking, implemented in check.cpp
    TypeCheckingCache* getTypeCheckingCache();
    void destroyTypeCheckingCache();

    RefPtr<RefObject> m_typeCheckingCache = nullptr;

    // Modules that have been dynamically loaded via `import`
    //
    // This is a list of unique modules loaded, in the order they were encountered.
    List<RefPtr<LoadedModule>> loadedModulesList;

    // Map from the path (or uniqueIdentity if available) of a module file to its definition
    Dictionary<String, RefPtr<LoadedModule>> mapPathToLoadedModule;

    // Map from the logical name of a module to its definition
    Dictionary<Name*, RefPtr<LoadedModule>> mapNameToLoadedModules;

    // Map from the mangled name of RTTI objects to sequential IDs
    // used by `switch`-based dynamic dispatch.
    Dictionary<String, uint32_t> mapMangledNameToRTTIObjectIndex;

    // Counters for allocating sequential IDs to witness tables conforming to each interface type.
    Dictionary<String, uint32_t> mapInterfaceMangledNameToSequentialIDCounters;

    SearchDirectoryList searchDirectoryCache;

    // The resulting specialized IR module for each entry point request
    List<RefPtr<IRModule>> compiledModules;

    ContentAssistInfo contentAssistInfo;

    /// File system implementation to use when loading files from disk.
    ///
    /// If this member is `null`, a default implementation that tries
    /// to use the native OS filesystem will be used instead.
    ///
    ComPtr<ISlangFileSystem> m_fileSystem;

    /// The extended file system implementation. Will be set to a default implementation
    /// if fileSystem is nullptr. Otherwise it will either be fileSystem's interface,
    /// or a wrapped impl that makes fileSystem operate as fileSystemExt
    ComPtr<ISlangFileSystemExt> m_fileSystemExt;

    /// Get the currenly set file system
    ISlangFileSystemExt* getFileSystemExt() { return m_fileSystemExt; }

    /// Load a file into memory using the configured file system.
    ///
    /// @param path The path to attempt to load from
    /// @param outBlob A destination pointer to receive the loaded blob
    /// @returns A `SlangResult` to indicate success or failure.
    ///
    SlangResult loadFile(String const& path, PathInfo& outPathInfo, ISlangBlob** outBlob);

    Expr* parseTermString(String str, Scope* scope);

    Type* specializeType(
        Type* unspecializedType,
        Int argCount,
        Type* const* args,
        DiagnosticSink* sink);

    /// Add a new target and return its index.
    UInt addTarget(CodeGenTarget target);

    /// "Bottleneck" routine for loading a module.
    ///
    /// All attempts to load a module, whether through
    /// Slang API calls, `import` operations, or other
    /// means, should bottleneck through `loadModuleImpl`,
    /// or one of the specialized cases `loadSourceModuleImpl`
    /// and `loadBinaryModuleImpl`.
    ///
    RefPtr<Module> loadModuleImpl(
        Name* name,
        const PathInfo& filePathInfo,
        ISlangBlob* fileContentsBlob,
        SourceLoc const& loc,
        DiagnosticSink* sink,
        const LoadedModuleDictionary* additionalLoadedModules,
        ModuleBlobType blobType);

    RefPtr<Module> loadSourceModuleImpl(
        Name* name,
        const PathInfo& filePathInfo,
        ISlangBlob* fileContentsBlob,
        SourceLoc const& loc,
        DiagnosticSink* sink,
        const LoadedModuleDictionary* additionalLoadedModules);

    RefPtr<Module> loadBinaryModuleImpl(
        Name* name,
        const PathInfo& filePathInfo,
        ISlangBlob* fileContentsBlob,
        SourceLoc const& loc,
        DiagnosticSink* sink);

    /// Either finds a previously-loaded module matching what
    /// was serialized into `moduleChunk`, or else attempts
    /// to load the serialized module.
    ///
    /// If a previously-loaded module is found that matches the
    /// name or path information in `moduleChunk`, then that
    /// previously-loaded module is returned.
    ///
    /// Othwerise, attempts to load a module from `moduleChunk`
    /// and, if successful, returns the freshly loaded module.
    ///
    /// Otherwise, return null.
    ///
    RefPtr<Module> findOrLoadSerializedModuleForModuleLibrary(
        ModuleChunkRef moduleChunk,
        DiagnosticSink* sink);

    RefPtr<Module> loadSerializedModule(
        Name* moduleName,
        const PathInfo& moduleFilePathInfo,
        ModuleChunkRef moduleChunk,
        SourceLoc const& requestingLoc,
        DiagnosticSink* sink);

    SlangResult loadSerializedModuleContents(
        Module* module,
        const PathInfo& moduleFilePathInfo,
        ModuleChunkRef moduleChunk,
        DiagnosticSink* sink);

    SourceFile* loadSourceFile(String pathFrom, String path);

    void loadParsedModule(
        RefPtr<FrontEndCompileRequest> compileRequest,
        RefPtr<TranslationUnitRequest> translationUnit,
        Name* name,
        PathInfo const& pathInfo);

    bool isBinaryModuleUpToDate(String fromPath, RiffContainer* container);
    bool isBinaryModuleUpToDate(String fromPath, ModuleChunkRef moduleChunk);

    RefPtr<Module> findOrImportModule(
        Name* name,
        SourceLoc const& loc,
        DiagnosticSink* sink,
        const LoadedModuleDictionary* loadedModules = nullptr);

    SourceFile* findFile(Name* name, SourceLoc loc, IncludeSystem& outIncludeSystem);
    struct IncludeResult
    {
        FileDecl* fileDecl;
        bool isNew;
    };
    IncludeResult findAndIncludeFile(
        Module* module,
        TranslationUnitRequest* translationUnit,
        Name* name,
        SourceLoc const& loc,
        DiagnosticSink* sink);

    SourceManager* getSourceManager() { return m_sourceManager; }

    /// Override the source manager for the linkage.
    ///
    /// This is only used to install a temporary override when
    /// parsing stuff from strings (where we don't want to retain
    /// full source files for the parsed result).
    ///
    /// TODO: We should remove the need for this hack.
    ///
    void setSourceManager(SourceManager* sourceManager) { m_sourceManager = sourceManager; }

    void setRequireCacheFileSystem(bool requireCacheFileSystem);

    void setFileSystem(ISlangFileSystem* fileSystem);

    DeclRef<Decl> specializeGeneric(
        DeclRef<Decl> declRef,
        List<Expr*> argExprs,
        DiagnosticSink* sink);

    DeclRef<Decl> specializeWithArgTypes(
        Expr* funcExpr,
        List<Type*> argTypes,
        DiagnosticSink* sink);

    bool isSpecialized(DeclRef<Decl> declRef);

    DiagnosticSink::Flags diagnosticSinkFlags = 0;

    bool m_requireCacheFileSystem = false;

    // Modules that have been read in with the -r option
    List<ComPtr<IArtifact>> m_libModules;

    void _stopRetainingParentSession() { m_retainedSession = nullptr; }

    // Get shared semantics information for reflection purposes.
    SharedSemanticsContext* getSemanticsForReflection();

private:
    /// The global Slang library session that this linkage is a child of
    Session* m_session = nullptr;

    RefPtr<Session> m_retainedSession;

    /// Tracks state of modules currently being loaded.
    ///
    /// This information is used to diagnose cases where
    /// a user tries to recursively import the same module
    /// (possibly along a transitive chain of `import`s).
    ///
    struct ModuleBeingImportedRAII
    {
    public:
        ModuleBeingImportedRAII(
            Linkage* linkage,
            Module* module,
            Name* name,
            SourceLoc const& importLoc)
            : linkage(linkage), module(module), name(name), importLoc(importLoc)
        {
            next = linkage->m_modulesBeingImported;
            linkage->m_modulesBeingImported = this;
        }

        ~ModuleBeingImportedRAII() { linkage->m_modulesBeingImported = next; }

        Linkage* linkage;
        Module* module;
        Name* name;
        SourceLoc importLoc;
        ModuleBeingImportedRAII* next;
    };

    // Any modules currently being imported will be listed here
    ModuleBeingImportedRAII* m_modulesBeingImported = nullptr;

    /// Is the given module in the middle of being imported?
    bool isBeingImported(Module* module);

    /// Diagnose that an error occured in the process of importing a module
    void _diagnoseErrorInImportedModule(DiagnosticSink* sink);

    List<Type*> m_specializedTypes;

    RefPtr<SharedSemanticsContext> m_semanticsForReflection;
};

/// Shared functionality between front- and back-end compile requests.
///
/// This is the base class for both `FrontEndCompileRequest` and
/// `BackEndCompileRequest`, and allows a small number of parts of
/// the compiler to be easily invocable from either front-end or
/// back-end work.
///
class CompileRequestBase : public RefObject
{
    // TODO: We really shouldn't need this type in the long run.
    // The few places that rely on it should be refactored to just
    // depend on the underlying information (a linkage and a diagnostic
    // sink) directly.
    //
    // The flags to control dumping and validation of IR should be
    // moved to some kind of shared settings/options `struct` that
    // both front-end and back-end requests can store.

public:
    Session* getSession();
    Linkage* getLinkage() { return m_linkage; }
    DiagnosticSink* getSink() { return m_sink; }
    SourceManager* getSourceManager() { return getLinkage()->getSourceManager(); }
    NamePool* getNamePool() { return getLinkage()->getNamePool(); }
    ISlangFileSystemExt* getFileSystemExt() { return getLinkage()->getFileSystemExt(); }
    SlangResult loadFile(String const& path, PathInfo& outPathInfo, ISlangBlob** outBlob)
    {
        return getLinkage()->loadFile(path, outPathInfo, outBlob);
    }

protected:
    CompileRequestBase(Linkage* linkage, DiagnosticSink* sink);

private:
    Linkage* m_linkage = nullptr;
    DiagnosticSink* m_sink = nullptr;
};

/// A request to compile source code to an AST + IR.
class FrontEndCompileRequest : public CompileRequestBase
{
public:
    /// Note that writers can be parsed as nullptr to disable output,
    /// and individual channels set to null to disable them
    FrontEndCompileRequest(Linkage* linkage, StdWriters* writers, DiagnosticSink* sink);

    int addEntryPoint(int translationUnitIndex, String const& name, Profile entryPointProfile);

    // Translation units we are being asked to compile
    List<RefPtr<TranslationUnitRequest>> translationUnits;

    // Additional modules that needs to be made visible to `import` while checking.
    const LoadedModuleDictionary* additionalLoadedModules = nullptr;

    RefPtr<TranslationUnitRequest> getTranslationUnit(UInt index)
    {
        return translationUnits[index];
    }

    // If true then generateIR will serialize out IR, and serialize back in again. Making
    // serialization a bottleneck or firewall between the front end and the backend
    bool useSerialIRBottleneck = false;

    // If true will serialize and de-serialize with debug information
    bool verifyDebugSerialization = false;

    CompilerOptionSet optionSet;

    List<RefPtr<FrontEndEntryPointRequest>> m_entryPointReqs;

    List<RefPtr<FrontEndEntryPointRequest>> const& getEntryPointReqs() { return m_entryPointReqs; }
    UInt getEntryPointReqCount() { return m_entryPointReqs.getCount(); }
    FrontEndEntryPointRequest* getEntryPointReq(UInt index) { return m_entryPointReqs[index]; }

    void parseTranslationUnit(TranslationUnitRequest* translationUnit);

    // Perform primary semantic checking on all
    // of the translation units in the program
    void checkAllTranslationUnits();

    void checkEntryPoints();

    void generateIR();

    SlangResult executeActionsInner();

    /// Add a translation unit to be compiled.
    ///
    /// @param language The source language that the translation unit will use (e.g.,
    /// `SourceLanguage::Slang`
    /// @param moduleName The name that will be used for the module compile from the translation
    /// unit.
    ///
    /// If moduleName is passed as nullptr a module name is generated.
    /// If all translation units in a compile request use automatically generated
    /// module names, then they are guaranteed not to conflict with one another.
    ///
    /// @return The zero-based index of the translation unit in this compile request.
    int addTranslationUnit(SourceLanguage language, Name* moduleName);

    int addTranslationUnit(TranslationUnitRequest* translationUnit);

    void addTranslationUnitSourceArtifact(int translationUnitIndex, IArtifact* sourceArtifact);

    void addTranslationUnitSourceBlob(
        int translationUnitIndex,
        String const& path,
        ISlangBlob* sourceBlob);

    void addTranslationUnitSourceFile(int translationUnitIndex, String const& path);

    /// Get a component type that represents the global scope of the compile request.
    ComponentType* getGlobalComponentType() { return m_globalComponentType; }

    /// Get a component type that represents the global scope of the compile request, plus the
    /// requested entry points.
    ComponentType* getGlobalAndEntryPointsComponentType()
    {
        return m_globalAndEntryPointsComponentType;
    }

    List<RefPtr<ComponentType>> const& getUnspecializedEntryPoints()
    {
        return m_unspecializedEntryPoints;
    }

    /// Does the code we are compiling represent part of the Slang core module?
    bool m_isCoreModuleCode = false;

    Name* m_defaultModuleName = nullptr;

    /// The irDumpOptions
    IRDumpOptions m_irDumpOptions;

    /// An "extra" entry point that was added via a library reference
    struct ExtraEntryPointInfo
    {
        Name* name;
        Profile profile;
        String mangledName;
    };

    /// A list of "extra" entry points added via a library reference
    List<ExtraEntryPointInfo> m_extraEntryPoints;

private:
    /// A component type that includes only the global scopes of the translation unit(s) that were
    /// compiled.
    RefPtr<ComponentType> m_globalComponentType;

    /// A component type that extends the global scopes with all of the entry points that were
    /// specified.
    RefPtr<ComponentType> m_globalAndEntryPointsComponentType;

    List<RefPtr<ComponentType>> m_unspecializedEntryPoints;

    RefPtr<StdWriters> m_writers;
};

/// A visitor for use with `ComponentType`s, allowing dispatch over the concrete subclasses.
class ComponentTypeVisitor
{
public:
    // The following methods should be overriden in a concrete subclass
    // to customize how it acts on each of the concrete types of component.
    //
    // In cases where the application wants to simply "recurse" on a
    // composite, specialized, or legacy component type it can use
    // the `visitChildren` methods below.
    //
    virtual void visitEntryPoint(
        EntryPoint* entryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) = 0;
    virtual void visitModule(
        Module* module,
        Module::ModuleSpecializationInfo* specializationInfo) = 0;
    virtual void visitComposite(
        CompositeComponentType* composite,
        CompositeComponentType::CompositeSpecializationInfo* specializationInfo) = 0;
    virtual void visitSpecialized(SpecializedComponentType* specialized) = 0;
    virtual void visitTypeConformance(TypeConformance* conformance) = 0;
    virtual void visitRenamedEntryPoint(
        RenamedEntryPointComponentType* renamedEntryPoint,
        EntryPoint::EntryPointSpecializationInfo* specializationInfo) = 0;

protected:
    // These helpers can be used to recurse into the logical children of a
    // component type, and are useful for the common case where a visitor
    // only cares about a few leaf cases.
    //
    void visitChildren(
        CompositeComponentType* composite,
        CompositeComponentType::CompositeSpecializationInfo* specializationInfo);
    void visitChildren(SpecializedComponentType* specialized);
};

/// A `TargetProgram` represents a `ComponentType` specialized for a particular `TargetRequest`
///
/// TODO: This should probably be renamed to `TargetComponentType`.
///
/// By binding a component type to a specific target, a `TargetProgram` allows
/// for things like layout to be computed, that fundamentally depend on
/// the choice of target.
///
/// A `TargetProgram` handles request for compiled kernel code for
/// entry point functions. In practice, kernel code can only be
/// correctly generated when the underlying `ComponentType` is "fully linked"
/// (has no remaining unsatisfied requirements).
///
class TargetProgram : public RefObject
{
public:
    TargetProgram(ComponentType* componentType, TargetRequest* targetReq);

    /// Get the underlying program
    ComponentType* getProgram() { return m_program; }

    /// Get the underlying target
    TargetRequest* getTargetReq() { return m_targetReq; }

    /// Get the layout for the program on the target.
    ///
    /// If this is the first time the layout has been
    /// requested, report any errors that arise during
    /// layout to the given `sink`.
    ///
    ProgramLayout* getOrCreateLayout(DiagnosticSink* sink);

    /// Get the layout for the program on the target.
    ///
    /// This routine assumes that `getOrCreateLayout`
    /// has already been called previously.
    ///
    ProgramLayout* getExistingLayout()
    {
        SLANG_ASSERT(m_layout);
        return m_layout;
    }

    /// Get the compiled code for an entry point on the target.
    ///
    /// If this is the first time that code generation has
    /// been requested, report any errors that arise during
    /// code generation to the given `sink`.
    ///
    IArtifact* getOrCreateEntryPointResult(Int entryPointIndex, DiagnosticSink* sink);
    IArtifact* getOrCreateWholeProgramResult(DiagnosticSink* sink);

    IArtifact* getExistingWholeProgramResult() { return m_wholeProgramResult; }
    /// Get the compiled code for an entry point on the target.
    ///
    /// This routine assumes that `getOrCreateEntryPointResult`
    /// has already been called previously.
    ///
    IArtifact* getExistingEntryPointResult(Int entryPointIndex)
    {
        return m_entryPointResults[entryPointIndex];
    }

    IArtifact* _createWholeProgramResult(
        DiagnosticSink* sink,
        EndToEndCompileRequest* endToEndReq = nullptr);

    /// Internal helper for `getOrCreateEntryPointResult`.
    ///
    /// This is used so that command-line and API-based
    /// requests for code can bottleneck through the same place.
    ///
    /// Shouldn't be called directly by most code.
    ///
    IArtifact* _createEntryPointResult(
        Int entryPointIndex,
        DiagnosticSink* sink,
        EndToEndCompileRequest* endToEndReq = nullptr);

    RefPtr<IRModule> getOrCreateIRModuleForLayout(DiagnosticSink* sink);

    RefPtr<IRModule> getExistingIRModuleForLayout() { return m_irModuleForLayout; }

    CompilerOptionSet& getOptionSet() { return m_optionSet; }

    HLSLToVulkanLayoutOptions* getHLSLToVulkanLayoutOptions()
    {
        return m_targetReq->getHLSLToVulkanLayoutOptions();
    }

    bool shouldEmitSPIRVDirectly()
    {
        return isKhronosTarget(m_targetReq) && getOptionSet().shouldEmitSPIRVDirectly();
    }

private:
    RefPtr<IRModule> createIRModuleForLayout(DiagnosticSink* sink);

    // The program being compiled or laid out
    ComponentType* m_program;

    // The target that code/layout will be generated for
    TargetRequest* m_targetReq;

    // The computed layout, if it has been generated yet
    RefPtr<ProgramLayout> m_layout;

    CompilerOptionSet m_optionSet;

    // Generated compile results for each entry point
    // in the parent `Program` (indexing matches
    // the order they are given in the `Program`)
    ComPtr<IArtifact> m_wholeProgramResult;
    List<ComPtr<IArtifact>> m_entryPointResults;

    RefPtr<IRModule> m_irModuleForLayout;
};

/// A back-end-specific object to track optional feaures/capabilities/extensions
/// that are discovered to be used by a program/kernel as part of code generation.
class ExtensionTracker : public RefObject
{
    // TODO: The existence of this type is evidence of a design/architecture problem.
    //
    // A better formulation of things requires a few key changes:
    //
    // 1. All optional capabilities need to be enumerated as part of the `CapabilitySet`
    //  system, so that they can be reasoned about uniformly across different targets
    //  and different layers of the compiler.
    //
    // 2. The front-end should be responsible for either or both of:
    //
    //   * Checking that `public` or otherwise externally-visible items (declarations/definitions)
    //     explicitly declare the capabilities they require, and that they only ever
    //     make use of items that are comatible with those required capabilities.
    //
    //   * Inferring the capabilities required by items that are not externally visible,
    //     and attaching those capabilities explicit as a modifier or other synthesized AST node.
    //
    // 3. The capabilities required by a given `ComponentType` and its entry points should be
    // explicitly know-able, and they should be something we can compare to the capabilities
    // of a code generation target *before* back-end code generation is started. We should be
    // able to issue error messages around lacking capabilities in a way the user can understand,
    // in terms of the high-level-language entities.

public:
};

/// A context for code generation in the compiler back-end
struct CodeGenContext
{
public:
    typedef List<Index> EntryPointIndices;

    struct Shared
    {
    public:
        Shared(
            TargetProgram* targetProgram,
            EntryPointIndices const& entryPointIndices,
            DiagnosticSink* sink,
            EndToEndCompileRequest* endToEndReq)
            : targetProgram(targetProgram)
            , entryPointIndices(entryPointIndices)
            , sink(sink)
            , endToEndReq(endToEndReq)
        {
        }

        //            Shared(
        //                TargetProgram*              targetProgram,
        //                EndToEndCompileRequest*     endToEndReq);

        TargetProgram* targetProgram = nullptr;
        EntryPointIndices entryPointIndices;
        DiagnosticSink* sink = nullptr;
        EndToEndCompileRequest* endToEndReq = nullptr;
    };

    CodeGenContext(Shared* shared)
        : m_shared(shared)
        , m_targetFormat(shared->targetProgram->getTargetReq()->getTarget())
        , m_targetProfile(shared->targetProgram->getOptionSet().getProfile())
    {
    }

    CodeGenContext(
        CodeGenContext* base,
        CodeGenTarget targetFormat,
        ExtensionTracker* extensionTracker = nullptr)
        : m_shared(base->m_shared)
        , m_targetFormat(targetFormat)
        , m_extensionTracker(extensionTracker)
    {
    }

    /// Get the diagnostic sink
    DiagnosticSink* getSink() { return m_shared->sink; }

    TargetProgram* getTargetProgram() { return m_shared->targetProgram; }

    EntryPointIndices const& getEntryPointIndices() { return m_shared->entryPointIndices; }

    CodeGenTarget getTargetFormat() { return m_targetFormat; }

    ExtensionTracker* getExtensionTracker() { return m_extensionTracker; }

    TargetRequest* getTargetReq() { return getTargetProgram()->getTargetReq(); }

    CapabilitySet getTargetCaps() { return getTargetReq()->getTargetCaps(); }

    CodeGenTarget getFinalTargetFormat() { return getTargetReq()->getTarget(); }

    ComponentType* getProgram() { return getTargetProgram()->getProgram(); }

    Linkage* getLinkage() { return getProgram()->getLinkage(); }

    Session* getSession() { return getLinkage()->getSessionImpl(); }

    /// Get the source manager
    SourceManager* getSourceManager() { return getLinkage()->getSourceManager(); }

    ISlangFileSystemExt* getFileSystemExt() { return getLinkage()->getFileSystemExt(); }

    EndToEndCompileRequest* isEndToEndCompile() { return m_shared->endToEndReq; }

    EndToEndCompileRequest* isPassThroughEnabled();

    Count getEntryPointCount() { return getEntryPointIndices().getCount(); }

    EntryPoint* getEntryPoint(Index index) { return getProgram()->getEntryPoint(index); }

    Index getSingleEntryPointIndex()
    {
        SLANG_ASSERT(getEntryPointCount() == 1);
        return getEntryPointIndices()[0];
    }

    //

    IRDumpOptions getIRDumpOptions();

    bool shouldValidateIR();
    bool shouldDumpIR();
    bool shouldReportCheckpointIntermediates();

    bool shouldTrackLiveness();

    bool shouldDumpIntermediates();
    String getIntermediateDumpPrefix();

    bool getUseUnknownImageFormatAsDefault();

    bool isSpecializationDisabled();

    bool shouldSkipSPIRVValidation();

    SlangResult requireTranslationUnitSourceFiles();

    //

    SlangResult emitEntryPoints(ComPtr<IArtifact>& outArtifact);

    SlangResult emitPrecompiledDownstreamIR(ComPtr<IArtifact>& outArtifact);

    void maybeDumpIntermediate(IArtifact* artifact);

    // Used to cause instructions available in precompiled blobs to be
    // removed between IR linking and target source generation.
    bool removeAvailableInDownstreamIR = false;

    // Determines if program level compilation like getTargetCode() or getEntryPointCode()
    // should return a fully linked downstream program or just the glue SPIR-V/DXIL that
    // imports and uses the precompiled SPIR-V/DXIL from constituent modules.
    // This is a no-op if modules are not precompiled.
    bool shouldSkipDownstreamLinking();

protected:
    CodeGenTarget m_targetFormat = CodeGenTarget::Unknown;
    Profile m_targetProfile;
    ExtensionTracker* m_extensionTracker = nullptr;

    /// Will output assembly as well as the artifact if appropriate for the artifact type for
    /// assembly output and conversion is possible
    void _dumpIntermediateMaybeWithAssembly(IArtifact* artifact);

    void _dumpIntermediate(IArtifact* artifact);
    void _dumpIntermediate(const ArtifactDesc& desc, void const* data, size_t size);

    /* Emits entry point source taking into account if a pass-through or not. Uses 'targetFormat' to
    determine the target (not targetReq) */
    SlangResult emitEntryPointsSource(ComPtr<IArtifact>& outArtifact);

    SlangResult emitEntryPointsSourceFromIR(ComPtr<IArtifact>& outArtifact);

    SlangResult emitWithDownstreamForEntryPoints(ComPtr<IArtifact>& outArtifact);

    /* Determines a suitable filename to identify the input for a given entry point being compiled.
    If the end-to-end compile is a pass-through case, will attempt to find the (unique) source file
    pathname for the translation unit containing the entry point at `entryPointIndex.
    If the compilation is not in a pass-through case, then always returns `"slang-generated"`.
    @param endToEndReq The end-to-end compile request which might be using pass-through compilation
    @param entryPointIndex The index of the entry point to compute a filename for.
    @return the appropriate source filename */
    String calcSourcePathForEntryPoints();

    TranslationUnitRequest* findPassThroughTranslationUnit(Int entryPointIndex);


    SlangResult _emitEntryPoints(ComPtr<IArtifact>& outArtifact);

private:
    Shared* m_shared = nullptr;
};

/// A compile request that spans the front and back ends of the compiler
///
/// This is what the command-line `slangc` uses, as well as the legacy
/// C API. It ties together the functionality of `Linkage`,
/// `FrontEndCompileRequest`, and `BackEndCompileRequest`, plus a small
/// number of additional features that primarily make sense for
/// command-line usage.
///
class EndToEndCompileRequest : public RefObject, public slang::ICompileRequest
{
public:
    SLANG_CLASS_GUID(0xce6d2383, 0xee1b, 0x4fd7, {0xa0, 0xf, 0xb8, 0xb6, 0x33, 0x12, 0x95, 0xc8})

    // ISlangUnknown
    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject)
        SLANG_OVERRIDE;
    SLANG_REF_OBJECT_IUNKNOWN_ADD_REF
    SLANG_REF_OBJECT_IUNKNOWN_RELEASE

    // slang::ICompileRequest
    virtual SLANG_NO_THROW void SLANG_MCALL setFileSystem(ISlangFileSystem* fileSystem)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setCompileFlags(SlangCompileFlags flags) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangCompileFlags SLANG_MCALL getCompileFlags() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setDumpIntermediates(int enable) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setDumpIntermediatePrefix(const char* prefix)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setEnableEffectAnnotations(bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setLineDirectiveMode(SlangLineDirectiveMode mode)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setCodeGenTarget(SlangCompileTarget target)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL addCodeGenTarget(SlangCompileTarget target)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetProfile(int targetIndex, SlangProfileID profile) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setTargetFlags(int targetIndex, SlangTargetFlags flags)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetFloatingPointMode(int targetIndex, SlangFloatingPointMode mode) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetMatrixLayoutMode(int targetIndex, SlangMatrixLayoutMode mode) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetForceGLSLScalarBufferLayout(int targetIndex, bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setTargetForceDXLayout(int targetIndex, bool value)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetGenerateWholeProgram(int targetIndex, bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setTargetEmbedDownstreamIR(int targetIndex, bool value)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setMatrixLayoutMode(SlangMatrixLayoutMode mode)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setDebugInfoLevel(SlangDebugInfoLevel level)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setOptimizationLevel(SlangOptimizationLevel level)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setOutputContainerFormat(SlangContainerFormat format)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setPassThrough(SlangPassThrough passThrough)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setDiagnosticCallback(SlangDiagnosticCallback callback, void const* userData) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setWriter(SlangWriterChannel channel, ISlangWriter* writer) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW ISlangWriter* SLANG_MCALL getWriter(SlangWriterChannel channel)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addSearchPath(const char* searchDir) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    addPreprocessorDefine(const char* key, const char* value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    processCommandLineArguments(char const* const* args, int argCount) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL
    addTranslationUnit(SlangSourceLanguage language, char const* name) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setDefaultModuleName(const char* defaultModuleName)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitPreprocessorDefine(
        int translationUnitIndex,
        const char* key,
        const char* value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    addTranslationUnitSourceFile(int translationUnitIndex, char const* path) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceString(
        int translationUnitIndex,
        char const* path,
        char const* source) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL addLibraryReference(
        const char* basePath,
        const void* libData,
        size_t libDataSize) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceStringSpan(
        int translationUnitIndex,
        char const* path,
        char const* sourceBegin,
        char const* sourceEnd) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceBlob(
        int translationUnitIndex,
        char const* path,
        ISlangBlob* sourceBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL
    addEntryPoint(int translationUnitIndex, char const* name, SlangStage stage) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL addEntryPointEx(
        int translationUnitIndex,
        char const* name,
        SlangStage stage,
        int genericArgCount,
        char const** genericArgs) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    setGlobalGenericArgs(int genericArgCount, char const** genericArgs) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    setTypeNameForGlobalExistentialTypeParam(int slotIndex, char const* typeName) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL setTypeNameForEntryPointExistentialTypeParam(
        int entryPointIndex,
        int slotIndex,
        char const* typeName) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setAllowGLSLInput(bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL compile() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDiagnosticOutput() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getDiagnosticOutputBlob(ISlangBlob** outBlob)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL getDependencyFileCount() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(int index) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW int SLANG_MCALL getTranslationUnitCount() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW char const* SLANG_MCALL getEntryPointSource(int entryPointIndex)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void const* SLANG_MCALL
    getEntryPointCode(int entryPointIndex, size_t* outSize) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointCodeBlob(
        int entryPointIndex,
        int targetIndex,
        ISlangBlob** outBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getTargetCodeBlob(int targetIndex, ISlangBlob** outBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getTargetHostCallable(int targetIndex, ISlangSharedLibrary** outSharedLibrary) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void const* SLANG_MCALL getCompileRequestCode(size_t* outSize)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW ISlangMutableFileSystem* SLANG_MCALL
    getCompileRequestResultAsFileSystem() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getContainerCode(ISlangBlob** outBlob)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadRepro(ISlangFileSystem* fileSystem, const void* data, size_t size) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveRepro(ISlangBlob** outBlob) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL enableReproCapture() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getProgram(slang::IComponentType** outProgram)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getEntryPoint(SlangInt entryPointIndex, slang::IComponentType** outEntryPoint) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getModule(SlangInt translationUnitIndex, slang::IModule** outModule) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getSession(slang::ISession** outSession)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangReflection* SLANG_MCALL getReflection() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setCommandLineCompilerMode() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    addTargetCapability(SlangInt targetIndex, SlangCapabilityID capability) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getProgramWithEntryPoints(slang::IComponentType** outProgram) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL isParameterLocationUsed(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        SlangParameterCategory category,
        SlangUInt spaceIndex,
        SlangUInt registerIndex,
        bool& outUsed) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetLineDirectiveMode(SlangInt targetIndex, SlangLineDirectiveMode mode) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    overrideDiagnosticSeverity(SlangInt messageID, SlangSeverity overrideSeverity) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangDiagnosticFlags SLANG_MCALL getDiagnosticFlags() SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setDiagnosticFlags(SlangDiagnosticFlags flags)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setDebugInfoFormat(SlangDebugInfoFormat format)
        SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setReportDownstreamTime(bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setReportPerfBenchmark(bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setSkipSPIRVValidation(bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetUseMinimumSlangOptimization(int targetIndex, bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW void SLANG_MCALL setIgnoreCapabilityCheck(bool value) SLANG_OVERRIDE;
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getCompileTimeProfile(ISlangProfiler** compileTimeProfile, bool isClear) SLANG_OVERRIDE;

    void setTrackLiveness(bool v);

    EndToEndCompileRequest(Session* session);

    EndToEndCompileRequest(Linkage* linkage);

    ~EndToEndCompileRequest();

    // If enabled will emit IR
    bool m_emitIr = false;

    // What container format are we being asked to generate?
    // If it's set to a format, the container blob will be calculated during compile
    ContainerFormat m_containerFormat = ContainerFormat::None;

    /// Where the container is stored. This is calculated as part of compile if m_containerFormat is
    /// set to a supported format.
    ComPtr<IArtifact> m_containerArtifact;
    /// Holds the container as a file system
    ComPtr<ISlangMutableFileSystem> m_containerFileSystem;

    /// File system used by repro system if a file couldn't be found within the repro (or associated
    /// directory)
    ComPtr<ISlangFileSystem> m_reproFallbackFileSystem =
        ComPtr<ISlangFileSystem>(OSFileSystem::getExtSingleton());

    // Path to output container to
    String m_containerOutputPath;

    // Should we just pass the input to another compiler?
    PassThroughMode m_passThrough = PassThroughMode::None;

    /// If output should be source embedded, define the style of the embedding
    SourceEmbedUtil::Style m_sourceEmbedStyle = SourceEmbedUtil::Style::None;
    /// The language to be used for source embedding
    SourceLanguage m_sourceEmbedLanguage = SourceLanguage::C;
    /// Source embed variable name. Note may be used as a basis for names if multiple items written
    String m_sourceEmbedName;

    /// Source code for the specialization arguments to use for the global specialization parameters
    /// of the program.
    List<String> m_globalSpecializationArgStrings;

    // Are we being driven by the command-line `slangc`, and should act accordingly?
    bool m_isCommandLineCompile = false;

    String m_diagnosticOutput;

    /// A blob holding the diagnostic output
    ComPtr<ISlangBlob> m_diagnosticOutputBlob;

    /// Per-entry-point information not tracked by other compile requests
    class EntryPointInfo : public RefObject
    {
    public:
        /// Source code for the specialization arguments to use for the specialization parameters of
        /// the entry point.
        List<String> specializationArgStrings;
    };
    List<EntryPointInfo> m_entryPoints;

    /// Per-target information only needed for command-line compiles
    class TargetInfo : public RefObject
    {
    public:
        // Requested output paths for each entry point.
        // An empty string indices no output desired for
        // the given entry point.
        Dictionary<Int, String> entryPointOutputPaths;
        String wholeTargetOutputPath;
        CompilerOptionSet targetOptions;
    };
    Dictionary<TargetRequest*, RefPtr<TargetInfo>> m_targetInfos;

    CompilerOptionSet m_optionSetForDefaultTarget;

    CompilerOptionSet& getTargetOptionSet(TargetRequest* req);

    CompilerOptionSet& getTargetOptionSet(Index targetIndex);

    String m_dependencyOutputPath;

    /// Writes the modules in a container to the stream
    SlangResult writeContainerToStream(Stream* stream);

    /// If a container format has been specified produce a container (stored in m_containerBlob)
    SlangResult maybeCreateContainer();
    /// If a container has been constructed and the filename/path has contents will try to write
    /// the container contents to the file
    SlangResult maybeWriteContainer(const String& fileName);

    Linkage* getLinkage() { return m_linkage; }

    int addEntryPoint(
        int translationUnitIndex,
        String const& name,
        Profile profile,
        List<String> const& genericTypeNames);

    void setWriter(WriterChannel chan, ISlangWriter* writer);
    ISlangWriter* getWriter(WriterChannel chan) const
    {
        return m_writers->getWriter(SlangWriterChannel(chan));
    }

    /// The end to end request can be passed as nullptr, if not driven by one
    SlangResult executeActionsInner();
    SlangResult executeActions();

    Session* getSession() { return m_session; }
    DiagnosticSink* getSink() { return &m_sink; }
    NamePool* getNamePool() { return getLinkage()->getNamePool(); }

    FrontEndCompileRequest* getFrontEndReq() { return m_frontEndReq; }

    ComponentType* getUnspecializedGlobalComponentType()
    {
        return getFrontEndReq()->getGlobalComponentType();
    }
    ComponentType* getUnspecializedGlobalAndEntryPointsComponentType()
    {
        return getFrontEndReq()->getGlobalAndEntryPointsComponentType();
    }

    ComponentType* getSpecializedGlobalComponentType() { return m_specializedGlobalComponentType; }
    ComponentType* getSpecializedGlobalAndEntryPointsComponentType()
    {
        return m_specializedGlobalAndEntryPointsComponentType;
    }

    ComponentType* getSpecializedEntryPointComponentType(Index index)
    {
        return m_specializedEntryPoints[index];
    }

    void writeArtifactToStandardOutput(IArtifact* artifact, DiagnosticSink* sink);

    void generateOutput();

    CompilerOptionSet& getOptionSet() { return m_linkage->m_optionSet; }

private:
    String _getWholeProgramPath(TargetRequest* targetReq);
    String _getEntryPointPath(TargetRequest* targetReq, Index entryPointIndex);

    /// Maybe write the artifact to the path (if set), or stdout (if there is no container or path)
    SlangResult _maybeWriteArtifact(const String& path, IArtifact* artifact);
    SlangResult _writeArtifact(const String& path, IArtifact* artifact);

    /// Adds any extra settings to complete a targetRequest
    void _completeTargetRequest(UInt targetIndex);

    ISlangUnknown* getInterface(const Guid& guid);

    void generateOutput(ComponentType* program);
    void generateOutput(TargetProgram* targetProgram);

    void init();

    Session* m_session = nullptr;
    RefPtr<Linkage> m_linkage;
    DiagnosticSink m_sink;
    RefPtr<FrontEndCompileRequest> m_frontEndReq;
    RefPtr<ComponentType> m_specializedGlobalComponentType;
    RefPtr<ComponentType> m_specializedGlobalAndEntryPointsComponentType;
    List<RefPtr<ComponentType>> m_specializedEntryPoints;

    // For output

    RefPtr<StdWriters> m_writers;
};

/* Returns SLANG_OK if pass through support is available */
SlangResult checkExternalCompilerSupport(Session* session, PassThroughMode passThrough);
/* Report an error appearing from external compiler to the diagnostic sink error to the diagnostic
sink.
@param compilerName The name of the compiler the error came for (or nullptr if not known)
@param res Result associated with the error. The error code will be reported. (Can take HRESULT -
and will expand to string if known)
@param diagnostic The diagnostic string associated with the compile failure
@param sink The diagnostic sink to report to */
void reportExternalCompileError(
    const char* compilerName,
    SlangResult res,
    const UnownedStringSlice& diagnostic,
    DiagnosticSink* sink);

//

// Information about BaseType that's useful for checking literals
struct BaseTypeInfo
{
    typedef uint8_t Flags;
    struct Flag
    {
        enum Enum : Flags
        {
            Signed = 0x1,
            FloatingPoint = 0x2,
            Integer = 0x4,
        };
    };

    SLANG_FORCE_INLINE static const BaseTypeInfo& getInfo(BaseType baseType)
    {
        return s_info[Index(baseType)];
    }

    static UnownedStringSlice asText(BaseType baseType);

    uint8_t sizeInBytes; ///< Size of type in bytes
    Flags flags;
    uint8_t baseType;

    static bool check();

private:
    static const BaseTypeInfo s_info[Index(BaseType::CountOf)];
};

class CodeGenTransitionMap
{
public:
    struct Pair
    {
        typedef Pair ThisType;
        SLANG_FORCE_INLINE bool operator==(const ThisType& rhs) const
        {
            return source == rhs.source && target == rhs.target;
        }
        SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

        SLANG_FORCE_INLINE HashCode getHashCode() const
        {
            return combineHash(HashCode(source), HashCode(target));
        }

        CodeGenTarget source;
        CodeGenTarget target;
    };

    void removeTransition(CodeGenTarget source, CodeGenTarget target)
    {
        m_map.remove(Pair{source, target});
    }
    void addTransition(CodeGenTarget source, CodeGenTarget target, PassThroughMode compiler)
    {
        SLANG_ASSERT(source != target);
        m_map.set(Pair{source, target}, compiler);
    }
    bool hasTransition(CodeGenTarget source, CodeGenTarget target) const
    {
        return m_map.containsKey(Pair{source, target});
    }
    PassThroughMode getTransition(CodeGenTarget source, CodeGenTarget target) const
    {
        const Pair pair{source, target};
        auto value = m_map.tryGetValue(pair);
        return value ? *value : PassThroughMode::None;
    }

protected:
    Dictionary<Pair, PassThroughMode> m_map;
};

class Session : public RefObject, public slang::IGlobalSession
{
public:
    SLANG_COM_INTERFACE(
        0xd6b767eb,
        0xd786,
        0x4343,
        {0x2a, 0x8c, 0x6d, 0xa0, 0x3d, 0x5a, 0xb4, 0x4a})

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject)
        SLANG_OVERRIDE;
    SLANG_REF_OBJECT_IUNKNOWN_ADD_REF
    SLANG_REF_OBJECT_IUNKNOWN_RELEASE

    // slang::IGlobalSession
    SLANG_NO_THROW SlangResult SLANG_MCALL
    createSession(slang::SessionDesc const& desc, slang::ISession** outSession) override;
    SLANG_NO_THROW SlangProfileID SLANG_MCALL findProfile(char const* name) override;
    SLANG_NO_THROW void SLANG_MCALL
    setDownstreamCompilerPath(SlangPassThrough passThrough, char const* path) override;
    SLANG_NO_THROW void SLANG_MCALL
    setDownstreamCompilerPrelude(SlangPassThrough inPassThrough, char const* prelude) override;
    SLANG_NO_THROW void SLANG_MCALL
    getDownstreamCompilerPrelude(SlangPassThrough inPassThrough, ISlangBlob** outPrelude) override;
    SLANG_NO_THROW const char* SLANG_MCALL getBuildTagString() override;
    SLANG_NO_THROW SlangResult SLANG_MCALL setDefaultDownstreamCompiler(
        SlangSourceLanguage sourceLanguage,
        SlangPassThrough defaultCompiler) override;
    SLANG_NO_THROW SlangPassThrough SLANG_MCALL
    getDefaultDownstreamCompiler(SlangSourceLanguage sourceLanguage) override;

    SLANG_NO_THROW void SLANG_MCALL
    setLanguagePrelude(SlangSourceLanguage inSourceLanguage, char const* prelude) override;
    SLANG_NO_THROW void SLANG_MCALL
    getLanguagePrelude(SlangSourceLanguage inSourceLanguage, ISlangBlob** outPrelude) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL
    createCompileRequest(slang::ICompileRequest** outCompileRequest) override;

    SLANG_NO_THROW void SLANG_MCALL
    addBuiltins(char const* sourcePath, char const* sourceString) override;
    SLANG_NO_THROW void SLANG_MCALL
    setSharedLibraryLoader(ISlangSharedLibraryLoader* loader) override;
    SLANG_NO_THROW ISlangSharedLibraryLoader* SLANG_MCALL getSharedLibraryLoader() override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    checkCompileTargetSupport(SlangCompileTarget target) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    checkPassThroughSupport(SlangPassThrough passThrough) override;

    void writeCoreModuleDoc(String config);
    SLANG_NO_THROW SlangResult SLANG_MCALL
    compileCoreModule(slang::CompileCoreModuleFlags flags) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    loadCoreModule(const void* coreModule, size_t coreModuleSizeInBytes) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL
    saveCoreModule(SlangArchiveType archiveType, ISlangBlob** outBlob) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL compileBuiltinModule(
        slang::BuiltinModuleName moduleName,
        slang::CompileCoreModuleFlags flags) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL loadBuiltinModule(
        slang::BuiltinModuleName moduleName,
        const void* coreModule,
        size_t coreModuleSizeInBytes) override;
    SLANG_NO_THROW SlangResult SLANG_MCALL saveBuiltinModule(
        slang::BuiltinModuleName moduleName,
        SlangArchiveType archiveType,
        ISlangBlob** outBlob) override;

    SLANG_NO_THROW SlangCapabilityID SLANG_MCALL findCapability(char const* name) override;

    SLANG_NO_THROW void SLANG_MCALL setDownstreamCompilerForTransition(
        SlangCompileTarget source,
        SlangCompileTarget target,
        SlangPassThrough compiler) override;
    SLANG_NO_THROW SlangPassThrough SLANG_MCALL getDownstreamCompilerForTransition(
        SlangCompileTarget source,
        SlangCompileTarget target) override;
    SLANG_NO_THROW void SLANG_MCALL
    getCompilerElapsedTime(double* outTotalTime, double* outDownstreamTime) override
    {
        *outDownstreamTime = m_downstreamCompileTime;
        *outTotalTime = m_totalCompileTime;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL setSPIRVCoreGrammar(char const* jsonPath) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL parseCommandLineArguments(
        int argc,
        const char* const* argv,
        slang::SessionDesc* outSessionDesc,
        ISlangUnknown** outAllocation) override;

    SLANG_NO_THROW SlangResult SLANG_MCALL
    getSessionDescDigest(slang::SessionDesc* sessionDesc, ISlangBlob** outBlob) override;

    /// Get the downstream compiler for a transition
    IDownstreamCompiler* getDownstreamCompiler(CodeGenTarget source, CodeGenTarget target);

    // This needs to be atomic not because of contention between threads as `Session` is
    // *not* multithreaded, but can be used exclusively on one thread at a time.
    // The need for atomic is purely for visibility. If the session is used on a different
    // thread we need to be sure any changes to m_epochId are visible to this thread.
    std::atomic<Index> m_epochId = 1;

    Scope* baseLanguageScope = nullptr;
    Scope* coreLanguageScope = nullptr;
    Scope* hlslLanguageScope = nullptr;
    Scope* slangLanguageScope = nullptr;
    Scope* glslLanguageScope = nullptr;
    Name* glslModuleName = nullptr;

    ModuleDecl* baseModuleDecl = nullptr;
    List<RefPtr<Module>> coreModules;

    SourceManager builtinSourceManager;

    SourceManager* getBuiltinSourceManager() { return &builtinSourceManager; }

    // Name pool stuff for unique-ing identifiers

    RootNamePool rootNamePool;
    NamePool namePool;

    RootNamePool* getRootNamePool() { return &rootNamePool; }
    NamePool* getNamePool() { return &namePool; }
    Name* getNameObj(String name) { return namePool.getName(name); }
    Name* tryGetNameObj(String name) { return namePool.tryGetName(name); }
    //

    /// This AST Builder should only be used for creating AST nodes that are global across requests
    /// not doing so could lead to memory being consumed but not used.
    ASTBuilder* getGlobalASTBuilder() { return globalAstBuilder; }
    void finalizeSharedASTBuilder();

    RefPtr<ASTBuilder> globalAstBuilder;

    // Generated code for core module, etc.
    String coreModulePath;

    ComPtr<ISlangBlob> coreLibraryCode;
    // ComPtr<ISlangBlob> slangLibraryCode;
    ComPtr<ISlangBlob> hlslLibraryCode;
    ComPtr<ISlangBlob> glslLibraryCode;
    ComPtr<ISlangBlob> autodiffLibraryCode;

    String getCoreModulePath();

    ComPtr<ISlangBlob> getCoreLibraryCode();
    ComPtr<ISlangBlob> getHLSLLibraryCode();
    ComPtr<ISlangBlob> getAutodiffLibraryCode();
    ComPtr<ISlangBlob> getGLSLLibraryCode();

    RefPtr<SharedASTBuilder> m_sharedASTBuilder;

    SPIRVCoreGrammarInfo& getSPIRVCoreGrammarInfo()
    {
        if (!spirvCoreGrammarInfo)
            setSPIRVCoreGrammar(nullptr);
        SLANG_ASSERT(spirvCoreGrammarInfo);
        return *spirvCoreGrammarInfo;
    }
    RefPtr<SPIRVCoreGrammarInfo> spirvCoreGrammarInfo;

    //

    void _setSharedLibraryLoader(ISlangSharedLibraryLoader* loader);

    /// Will try to load the library by specified name (using the set loader), if not one already
    /// available.
    IDownstreamCompiler* getOrLoadDownstreamCompiler(PassThroughMode type, DiagnosticSink* sink);
    /// Will unload the specified shared library if it's currently loaded
    void resetDownstreamCompiler(PassThroughMode type);

    /// Get the prelude associated with the language
    const String& getPreludeForLanguage(SourceLanguage language)
    {
        return m_languagePreludes[int(language)];
    }

    /// Get the built in linkage -> handy to get the core module from
    Linkage* getBuiltinLinkage() const { return m_builtinLinkage; }

    Module* getBuiltinModule(slang::BuiltinModuleName builtinModuleName);

    Name* getCompletionRequestTokenName() const { return m_completionTokenName; }

    void init();

    void addBuiltinSource(
        Scope* scope,
        String const& path,
        ISlangBlob* sourceBlob,
        Module*& outModule);
    ~Session();

    void addDownstreamCompileTime(double time) { m_downstreamCompileTime += time; }
    void addTotalCompileTime(double time) { m_totalCompileTime += time; }

    ComPtr<ISlangSharedLibraryLoader>
        m_sharedLibraryLoader; ///< The shared library loader (never null)

    int m_downstreamCompilerInitialized = 0;

    RefPtr<DownstreamCompilerSet>
        m_downstreamCompilerSet; ///< Information about all available downstream compilers.
    ComPtr<IDownstreamCompiler> m_downstreamCompilers[int(
        PassThroughMode::CountOf)]; ///< A downstream compiler for a pass through
    DownstreamCompilerLocatorFunc m_downstreamCompilerLocators[int(PassThroughMode::CountOf)];
    Name* m_completionTokenName = nullptr; ///< The name of a completion request token.

    /// For parsing command line options
    CommandOptions m_commandOptions;

    int m_typeDictionarySize = 0;

    RefPtr<RefObject> m_typeCheckingCache;
    TypeCheckingCache* getTypeCheckingCache();
    std::mutex m_typeCheckingCacheMutex;

private:
    struct BuiltinModuleInfo
    {
        const char* name;
        Scope* languageScope;
    };

    BuiltinModuleInfo getBuiltinModuleInfo(slang::BuiltinModuleName name);

    void _initCodeGenTransitionMap();

    SlangResult _readBuiltinModule(
        ISlangFileSystem* fileSystem,
        Scope* scope,
        String moduleName,
        Module*& outModule);

    SlangResult _loadRequest(EndToEndCompileRequest* request, const void* data, size_t size);

    /// Linkage used for all built-in (core module) code.
    RefPtr<Linkage> m_builtinLinkage;

    String
        m_downstreamCompilerPaths[int(PassThroughMode::CountOf)]; ///< Paths for each pass through
    String m_languagePreludes[int(SourceLanguage::CountOf)]; ///< Prelude for each source language
    PassThroughMode m_defaultDownstreamCompilers[int(SourceLanguage::CountOf)];

    // Describes a conversion from one code gen target (source) to another (target)
    CodeGenTransitionMap m_codeGenTransitionMap;

    double m_downstreamCompileTime = 0.0;
    double m_totalCompileTime = 0.0;
};

const char* getBuiltinModuleNameStr(slang::BuiltinModuleName name);

void checkTranslationUnit(
    TranslationUnitRequest* translationUnit,
    LoadedModuleDictionary& loadedModules);

// Look for a module that matches the given name:
// either one we've loaded already, or one we
// can find vai the search paths available to us.
//
// Needed by import declaration checking.
//
RefPtr<Module> findOrImportModule(
    Linkage* linkage,
    Name* name,
    SourceLoc const& loc,
    DiagnosticSink* sink,
    const LoadedModuleDictionary* additionalLoadedModules);

SlangResult passthroughDownstreamDiagnostics(
    DiagnosticSink* sink,
    IDownstreamCompiler* compiler,
    IArtifact* artifact);

//
// The following functions are utilties to convert between
// matching "external" (public API) and "internal" (implementation)
// types. They are favored over explicit casts because they
// help avoid making incorrect conversions (e.g., when using
// `reinterpret_cast` or C-style casts), and because they
// abstract over the conversion required for each pair of types.
//

SLANG_FORCE_INLINE slang::IGlobalSession* asExternal(Session* session)
{
    return static_cast<slang::IGlobalSession*>(session);
}

SLANG_FORCE_INLINE ComPtr<Session> asInternal(slang::IGlobalSession* session)
{
    Slang::Session* internalSession = nullptr;
    session->queryInterface(SLANG_IID_PPV_ARGS(&internalSession));
    return ComPtr<Session>(INIT_ATTACH, static_cast<Session*>(internalSession));
}

SLANG_FORCE_INLINE slang::ISession* asExternal(Linkage* linkage)
{
    return static_cast<slang::ISession*>(linkage);
}

SLANG_FORCE_INLINE Module* asInternal(slang::IModule* module)
{
    return static_cast<Module*>(module);
}

SLANG_FORCE_INLINE slang::IModule* asExternal(Module* module)
{
    return static_cast<slang::IModule*>(module);
}

ComponentType* asInternal(slang::IComponentType* inComponentType);

SLANG_FORCE_INLINE slang::IComponentType* asExternal(ComponentType* componentType)
{
    return static_cast<slang::IComponentType*>(componentType);
}

SLANG_FORCE_INLINE slang::ProgramLayout* asExternal(ProgramLayout* programLayout)
{
    return (slang::ProgramLayout*)programLayout;
}

SLANG_FORCE_INLINE Type* asInternal(slang::TypeReflection* type)
{
    return reinterpret_cast<Type*>(type);
}

SLANG_FORCE_INLINE slang::TypeReflection* asExternal(Type* type)
{
    return reinterpret_cast<slang::TypeReflection*>(type);
}

SLANG_FORCE_INLINE DeclRef<Decl> asInternal(slang::GenericReflection* generic)
{
    return DeclRef<Decl>(reinterpret_cast<DeclRefBase*>(generic));
}

SLANG_FORCE_INLINE slang::GenericReflection* asExternal(DeclRef<Decl> generic)
{
    return reinterpret_cast<slang::GenericReflection*>(generic.declRefBase);
}

SLANG_FORCE_INLINE TypeLayout* asInternal(slang::TypeLayoutReflection* type)
{
    return reinterpret_cast<TypeLayout*>(type);
}

SLANG_FORCE_INLINE slang::TypeLayoutReflection* asExternal(TypeLayout* type)
{
    return reinterpret_cast<slang::TypeLayoutReflection*>(type);
}

SLANG_FORCE_INLINE SlangCompileRequest* asExternal(EndToEndCompileRequest* request)
{
    return static_cast<SlangCompileRequest*>(request);
}

SLANG_FORCE_INLINE EndToEndCompileRequest* asInternal(SlangCompileRequest* request)
{
    // Converts to the internal type -- does a runtime type check through queryInterfae
    SLANG_ASSERT(request);
    EndToEndCompileRequest* endToEndRequest = nullptr;
    // NOTE! We aren't using to access an interface, so *doesn't* return with a refcount
    request->queryInterface(SLANG_IID_PPV_ARGS(&endToEndRequest));
    SLANG_ASSERT(endToEndRequest);
    return endToEndRequest;
}

SLANG_FORCE_INLINE SlangCompileTarget asExternal(CodeGenTarget target)
{
    return (SlangCompileTarget)target;
}

SLANG_FORCE_INLINE SlangSourceLanguage asExternal(SourceLanguage sourceLanguage)
{
    return (SlangSourceLanguage)sourceLanguage;
}

// helpers for error/warning reporting
enum class DiagnosticCategory
{
    None = 0,
    Capability = 1 << 0,
};
template<typename P, typename... Args>
bool maybeDiagnose(
    DiagnosticSink* sink,
    CompilerOptionSet& optionSet,
    DiagnosticCategory errorType,
    P const& pos,
    DiagnosticInfo const& info,
    Args const&... args)
{
    if ((int)errorType & (int)DiagnosticCategory::Capability &&
        optionSet.getBoolOption(CompilerOptionName::IgnoreCapabilities))
        return false;
    return sink->diagnose(pos, info, args...);
}

template<typename P, typename... Args>
bool maybeDiagnoseWarningOrError(
    DiagnosticSink* sink,
    CompilerOptionSet& optionSet,
    DiagnosticCategory errorType,
    P const& pos,
    DiagnosticInfo const& warningInfo,
    DiagnosticInfo const& errorInfo,
    Args const&... args)
{
    if ((int)errorType & (int)DiagnosticCategory::Capability &&
        optionSet.getBoolOption(CompilerOptionName::RestrictiveCapabilityCheck))
    {
        return maybeDiagnose(sink, optionSet, errorType, pos, errorInfo, args...);
    }
    else
    {
        return maybeDiagnose(sink, optionSet, errorType, pos, warningInfo, args...);
    }
}

} // namespace Slang

#endif
