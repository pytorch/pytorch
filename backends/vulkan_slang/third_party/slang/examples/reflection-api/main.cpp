// main.cpp

// Reflection API Example Program
// ==============================
//
// This file provides the application code for the `reflection-api` example.
// This example uses the Slang reflection API to travserse the structure
// of the parameters of a Slang program and their types.
//
// This program is a companion Slang reflection API documentation:
// https://shader-slang.org/slang/user-guide/compiling.html
//
// Boilerplate
// -----------
//
// The following lines are boilerplate common to set up this example
// to use the infrastructure for example programs in the Slang
// repository.
//

#include "slang-com-ptr.h"
#include "slang.h"
typedef SlangResult Result;

#include "core/slang-basic.h"
#include "examples/example-base/example-base.h"
using Slang::ComPtr;
using Slang::String;
using Slang::List;

static const ExampleResources resourceBase("reflection-api");

// Configuration
// -------------
//
// For simplicity, this example uses a hard-coded list of shader programs
// to compile, each represented as the name of a `.slang` file, along with
// a hard-coded list of targets to compile and reflect the programs for.
//

static const char* kSourceFileNames[] = {
    "raster-simple.slang",
    "compute-simple.slang",
};

static const struct
{
    SlangCompileTarget format;
    const char* profile;
} kTargets[] = {
    {SLANG_DXIL, "sm_6_0"},
    {SLANG_SPIRV, "sm_6_0"},
};
static const int kTargetCount = SLANG_COUNT_OF(kTargets);

// The `ReflectingPrinting` Type
// -------------------------
//
// We wrap most of the code for this example in a `struct`
// type, in order to provide a bit more freedom in order
// of declaration.
//
// When possible, we will follow the order of declarations
// in the accompanying document, to help readers who want
// to following along in the code while reading.
//
struct ReflectingPrinting
{
    // Scoping things in a type allows us to declare functions
    // out of order more easily, but we still have to forward-declare
    // types when they will be used before they are declared.
    //
    struct AccessPath;

    // Output Formatting
    // -----------------
    //
    // This example program outputs reflection information in a format
    // that is (or at least is intended to be) compatible with YAML.
    //
    // We do not want the code to be overly complicated with issues
    // around formatting, so the details of the actual printing logic
    // are largely left until later. However, there are a pair of
    // macros that help to keep things tidy that we need to introduce
    // here, before they are used.
    //
#define WITH_ARRAY() for (int _i = (beginArray(), 1); _i; _i = (endArray(), 0))

#define SCOPED_OBJECT() ScopedObject scopedObject##__COUNTER__(this)

    // Compiling a Program
    // -------------------
    //
    Result compileAndReflectProgram(slang::ISession* session, const char* sourceFileName)
    {
        SCOPED_OBJECT();
        printComment("program");

        key("file name");
        printQuotedString(sourceFileName);
        String sourceFilePath = resourceBase.resolveResource(sourceFileName);

        ComPtr<slang::IBlob> diagnostics;
        Result result = SLANG_OK;

        // ### Loading a Module
        //

        ComPtr<slang::IModule> module;
        module = session->loadModule(sourceFilePath.getBuffer(), diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        if (!module)
            return SLANG_FAIL;

        List<ComPtr<slang::IComponentType>> componentsToLink;

        // ### Variable decls
        //
        key("global constants");
        WITH_ARRAY()
        for (auto decl : module->getModuleReflection()->getChildren())
        {
            if (auto varDecl = decl->asVariable(); varDecl &&
                                                   varDecl->findModifier(slang::Modifier::Const) &&
                                                   varDecl->findModifier(slang::Modifier::Static))
            {
                element();
                printVariable(varDecl);
            }
        }

        // ### Finding Entry Points
        //

        key("defined entry points");
        int definedEntryPointCount = module->getDefinedEntryPointCount();
        WITH_ARRAY()
        for (int i = 0; i < definedEntryPointCount; i++)
        {
            ComPtr<slang::IEntryPoint> entryPoint;
            SLANG_RETURN_ON_FAIL(module->getDefinedEntryPoint(i, entryPoint.writeRef()));

            element();
            SCOPED_OBJECT();
            key("name");
            printQuotedString(entryPoint->getFunctionReflection()->getName());

            componentsToLink.add(ComPtr<slang::IComponentType>(entryPoint.get()));
        }

        // ### Composing and Linking
        //

        ComPtr<slang::IComponentType> composed;
        result = session->createCompositeComponentType(
            (slang::IComponentType**)componentsToLink.getBuffer(),
            componentsToLink.getCount(),
            composed.writeRef(),
            diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        SLANG_RETURN_ON_FAIL(result);

        ComPtr<slang::IComponentType> program;
        result = composed->link(program.writeRef(), diagnostics.writeRef());
        diagnoseIfNeeded(diagnostics);
        SLANG_RETURN_ON_FAIL(result);

        key("layouts");
        WITH_ARRAY()
        for (int targetIndex = 0; targetIndex < kTargetCount; ++targetIndex)
        {
            element();

            // ### Getting the Program Layout
            //
            slang::ProgramLayout* programLayout =
                program->getLayout(targetIndex, diagnostics.writeRef());
            diagnoseIfNeeded(diagnostics);
            if (!programLayout)
            {
                result = SLANG_FAIL;
                continue;
            }

            SLANG_RETURN_ON_FAIL(
                collectEntryPointMetadata(program, targetIndex, definedEntryPointCount));

            _programLayout = programLayout;
            auto targetFormat = kTargets[targetIndex].format;
            printProgramLayout(programLayout, targetFormat);
        }

        return result;
    }
    slang::ProgramLayout* _programLayout = nullptr;

    Result compileAndReflectPrograms(slang::ISession* session)
    {
        Result result = SLANG_OK;

        WITH_ARRAY()
        for (auto fileName : kSourceFileNames)
        {
            element();
            auto programResult = compileAndReflectProgram(session, fileName);
            if (SLANG_FAILED(programResult))
            {
                result = programResult;
            }
        }

        return result;
    }

    // Types and Variables
    // -------------------
    //
    // ### Variables
    //
    void printVariable(slang::VariableReflection* variable)
    {
        SCOPED_OBJECT();

        const char* name = variable->getName();
        slang::TypeReflection* type = variable->getType();

        key("name");
        printQuotedString(name);
        key("type");
        printType(type);

        int64_t value;
        if (SLANG_SUCCEEDED(variable->getDefaultValueInt(&value)))
        {
            key("value");
            printf("%" PRId64, value);
        }
    }

    // ### Types
    //
    void printType(slang::TypeReflection* type)
    {
        SCOPED_OBJECT();

        const char* name = type->getName();
        slang::TypeReflection::Kind kind = type->getKind();

        key("name");
        printQuotedString(name);
        key("kind");
        printTypeKind(kind);

        // There is information that we would like to
        // print for both types and type layouts, so
        // we will factor the common logic into a
        // subroutine so that we can share the code.
        //
        printCommonTypeInfo(type);

        switch (type->getKind())
        {
        default:
            break;

        // #### Structure Types
        //
        case slang::TypeReflection::Kind::Struct:
            {
                key("fields");
                int fieldCount = type->getFieldCount();

                WITH_ARRAY();
                for (int f = 0; f < fieldCount; f++)
                {
                    element();
                    auto field = type->getFieldByIndex(f);

                    printVariable(field);
                }
            }
            break;

        // #### Array Types
        // #### Vector Types
        // #### Matrix Types
        //
        case slang::TypeReflection::Kind::Array:
        case slang::TypeReflection::Kind::Vector:
        case slang::TypeReflection::Kind::Matrix:
            {
                key("element type");
                printType(type->getElementType());
            }
            break;

        // #### Resource Types
        //
        case slang::TypeReflection::Kind::Resource:
            {
                key("result type");
                printType(type->getResourceResultType());
            }
            break;

        // #### Single-Element Container Types
        //
        case slang::TypeReflection::Kind::ConstantBuffer:
        case slang::TypeReflection::Kind::ParameterBlock:
        case slang::TypeReflection::Kind::TextureBuffer:
        case slang::TypeReflection::Kind::ShaderStorageBuffer:
            {
                key("element type");
                printType(type->getElementType());
            }
            break;
        }
    }

    // #### Array Types
    //
    void printPossiblyUnbounded(size_t value)
    {
        if (value == ~size_t(0))
        {
            printf("unbounded");
        }
        else
        {
            printf("%u", unsigned(value));
        }
    }

    void printCommonTypeInfo(slang::TypeReflection* type)
    {
        switch (type->getKind())
        {
        // #### Scalar Types
        //
        case slang::TypeReflection::Kind::Scalar:
            {
                key("scalar type");
                printScalarType(type->getScalarType());
            }
            break;

        // #### Array Types
        //
        case slang::TypeReflection::Kind::Array:
            {
                key("element count");
                printPossiblyUnbounded(type->getElementCount());
            }
            break;

        // #### Vector Types
        //
        case slang::TypeReflection::Kind::Vector:
            {
                key("element count");
                print(type->getElementCount());
            }
            break;

        // #### Matrix Types
        //
        case slang::TypeReflection::Kind::Matrix:
            {
                key("row count");
                print(type->getRowCount());

                key("column count");
                print(type->getColumnCount());
            }
            break;

        // #### Resource Types
        //
        case slang::TypeReflection::Kind::Resource:
            {
                key("shape");
                printResourceShape(type->getResourceShape());

                key("access");
                printResourceAccess(type->getResourceAccess());
            }
            break;

        default:
            break;
        }
    }

    // Layout for Types and Variables
    // ------------------------------
    //
    // ### Variable Layouts
    //
    void printVariableLayout(slang::VariableLayoutReflection* variableLayout, AccessPath accessPath)
    {
        SCOPED_OBJECT();

        key("name");
        printQuotedString(variableLayout->getName());

        printOffsets(variableLayout, accessPath);

        printVaryingParameterInfo(variableLayout);

        ExtendedAccessPath variablePath(accessPath, variableLayout);

        key("type layout");
        printTypeLayout(variableLayout->getTypeLayout(), variablePath);
    }

    // #### Offsets

    void printRelativeOffsets(slang::VariableLayoutReflection* variableLayout)
    {
        key("relative");
        int usedLayoutUnitCount = variableLayout->getCategoryCount();
        WITH_ARRAY();
        for (int i = 0; i < usedLayoutUnitCount; ++i)
        {
            element();

            auto layoutUnit = variableLayout->getCategoryByIndex(i);
            printOffset(variableLayout, layoutUnit);
        }
    }

    void printOffset(
        slang::VariableLayoutReflection* variableLayout,
        slang::ParameterCategory layoutUnit)
    {
        printOffset(
            layoutUnit,
            variableLayout->getOffset(layoutUnit),
            variableLayout->getBindingSpace(layoutUnit));
    }

    void printOffset(slang::ParameterCategory layoutUnit, size_t offset, size_t spaceOffset)
    {
        SCOPED_OBJECT();

        key("value");
        print(offset);
        key("unit");
        printLayoutUnit(layoutUnit);

        // #### Spaces / Sets

        switch (layoutUnit)
        {
        default:
            break;

        case slang::ParameterCategory::ConstantBuffer:
        case slang::ParameterCategory::ShaderResource:
        case slang::ParameterCategory::UnorderedAccess:
        case slang::ParameterCategory::SamplerState:
        case slang::ParameterCategory::DescriptorTableSlot:
            key("space");
            print(spaceOffset);
            break;
        }
    }

    // ### Type Layouts
    //
    void printTypeLayout(slang::TypeLayoutReflection* typeLayout, AccessPath accessPath)
    {
        SCOPED_OBJECT();

        key("name");
        printQuotedString(typeLayout->getName());
        key("kind");
        printTypeKind(typeLayout->getKind());
        printCommonTypeInfo(typeLayout->getType());

        printSizes(typeLayout);

        printKindSpecificInfo(typeLayout, accessPath);
    }

    // #### Size
    //
    void printSizes(slang::TypeLayoutReflection* typeLayout)
    {
        key("size");

        int usedLayoutUnitCount = typeLayout->getCategoryCount();
        WITH_ARRAY()
        for (int i = 0; i < usedLayoutUnitCount; ++i)
        {
            element();

            auto layoutUnit = typeLayout->getCategoryByIndex(i);
            printSize(typeLayout, layoutUnit);
        }

        // #### Alignment and Stride
        if (typeLayout->getSize() != 0)
        {
            key("alignment in bytes");
            print(typeLayout->getAlignment());

            key("stride in bytes");
            print(typeLayout->getStride());
        }
    }

    void printSize(slang::TypeLayoutReflection* typeLayout, slang::ParameterCategory layoutUnit)
    {
        printSize(layoutUnit, typeLayout->getSize(layoutUnit));
    }

    void printSize(slang::ParameterCategory layoutUnit, size_t size)
    {
        SCOPED_OBJECT();

        key("value");
        printPossiblyUnbounded(size);
        key("unit");
        printLayoutUnit(layoutUnit);
    }

    // #### Kind-Specific Information
    //
    void printKindSpecificInfo(slang::TypeLayoutReflection* typeLayout, AccessPath accessPath)
    {
        switch (typeLayout->getKind())
        {
        // #### Structure Type Layouts
        //
        case slang::TypeReflection::Kind::Struct:
            {
                key("fields");

                int fieldCount = typeLayout->getFieldCount();
                WITH_ARRAY()
                for (int f = 0; f < fieldCount; f++)
                {
                    element();

                    auto field = typeLayout->getFieldByIndex(f);
                    printVariableLayout(field, accessPath);
                }
            }
            break;

        // #### Array Type Layouts
        //
        case slang::TypeReflection::Kind::Array:
            {
                key("element type layout");
                printTypeLayout(typeLayout->getElementTypeLayout(), AccessPath());
            }
            break;

        // #### Matrix Type Layouts
        //
        case slang::TypeReflection::Kind::Matrix:
            {
                key("matrix layout mode");
                printMatrixLayoutMode(typeLayout->getMatrixLayoutMode());

                key("element type layout");
                printTypeLayout(typeLayout->getElementTypeLayout(), AccessPath());
            }
            break;

        case slang::TypeReflection::Kind::Vector:
            {
                key("element type layout");
                printTypeLayout(typeLayout->getElementTypeLayout(), AccessPath());
            }
            break;

        // #### Single-Element Containers
        //
        case slang::TypeReflection::Kind::ConstantBuffer:
        case slang::TypeReflection::Kind::ParameterBlock:
        case slang::TypeReflection::Kind::TextureBuffer:
        case slang::TypeReflection::Kind::ShaderStorageBuffer:
            {
                auto containerVarLayout = typeLayout->getContainerVarLayout();
                auto elementVarLayout = typeLayout->getElementVarLayout();

                key("container");
                {
                    SCOPED_OBJECT();
                    printOffsets(containerVarLayout, accessPath);
                }

                AccessPath innerOffsets = accessPath;
                innerOffsets.deepestConstantBufer = innerOffsets.leaf;
                if (containerVarLayout->getTypeLayout()->getSize(
                        slang::ParameterCategory::SubElementRegisterSpace) != 0)
                {
                    innerOffsets.deepestParameterBlock = innerOffsets.leaf;
                }

                key("content");
                {
                    SCOPED_OBJECT();

                    printOffsets(elementVarLayout, innerOffsets);

                    ExtendedAccessPath elementOffsets(innerOffsets, elementVarLayout);

                    key("type layout");
                    printTypeLayout(elementVarLayout->getTypeLayout(), elementOffsets);
                }
            }
            break;

        case slang::TypeReflection::Kind::Resource:
            {
                if ((typeLayout->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK) ==
                    SLANG_STRUCTURED_BUFFER)
                {
                    key("element type layout");
                    printTypeLayout(typeLayout->getElementTypeLayout(), accessPath);
                }
                else
                {
                    key("result type");
                    printType(typeLayout->getResourceResultType());
                }
            }
            break;

        default:
            break;
        }
    }

    // Programs and Scopes
    // -------------------
    //
    void printProgramLayout(slang::ProgramLayout* programLayout, SlangCompileTarget targetFormat)
    {
        SCOPED_OBJECT();

        key("target");
        printTargetFormat(targetFormat);

        AccessPath rootOffsets;
        rootOffsets.valid = true;

        key("global scope");
        {
            SCOPED_OBJECT();
            printScope(programLayout->getGlobalParamsVarLayout(), rootOffsets);
        }

        key("entry points");
        int entryPointCount = programLayout->getEntryPointCount();
        WITH_ARRAY()
        for (int i = 0; i < entryPointCount; ++i)
        {
            element();
            printEntryPointLayout(programLayout->getEntryPointByIndex(i), rootOffsets);
        }
    }

    // ### Global Scope
    //
    void printScope(slang::VariableLayoutReflection* scopeVarLayout, AccessPath accessPath)
    {
        ExtendedAccessPath scopeOffsets(accessPath, scopeVarLayout);

        auto scopeTypeLayout = scopeVarLayout->getTypeLayout();
        switch (scopeTypeLayout->getKind())
        {
        // #### Parameters are Grouped Into a Structure
        //
        case slang::TypeReflection::Kind::Struct:
            {
                key("parameters");

                int paramCount = scopeTypeLayout->getFieldCount();
                for (int i = 0; i < paramCount; i++)
                {
                    element();

                    auto param = scopeTypeLayout->getFieldByIndex(i);

                    printVariableLayout(param, scopeOffsets);
                }
            }
            break;

        // #### Wrapped in a Constant Buffer If Needed
        //
        case slang::TypeReflection::Kind::ConstantBuffer:
            key("automatically-introduced constant buffer");
            {
                SCOPED_OBJECT();
                printOffsets(scopeTypeLayout->getContainerVarLayout(), scopeOffsets);
            }

            printScope(scopeTypeLayout->getElementVarLayout(), scopeOffsets);
            break;

        // #### Wrapped in a Parameter Block If Needed
        //
        case slang::TypeReflection::Kind::ParameterBlock:
            key("automatically-introduced parameter block");
            {
                SCOPED_OBJECT();
                printOffsets(scopeTypeLayout->getContainerVarLayout(), scopeOffsets);
            }

            printScope(scopeTypeLayout->getElementVarLayout(), scopeOffsets);
            break;

        default:
            // Note that this default case is never expected to
            // arise with the current Slang compiler and reflection
            // API, but we include it here as a kind of failsafe.
            //
            key("variable layout");
            printVariableLayout(scopeVarLayout, accessPath);
            break;
        }
    }

    // ### Entry Points
    //
    void printEntryPointLayout(slang::EntryPointReflection* entryPointLayout, AccessPath accessPath)
    {
        SCOPED_OBJECT();

        key("stage");
        printStage(entryPointLayout->getStage());

        printStageSpecificInfo(entryPointLayout);

        printScope(entryPointLayout->getVarLayout(), accessPath);

        auto resultVariableLayout = entryPointLayout->getResultVarLayout();
        if (resultVariableLayout->getTypeLayout()->getKind() != slang::TypeReflection::Kind::None)
        {
            key("result");
            printVariableLayout(resultVariableLayout, accessPath);
        }
    }

    // #### Stage-Specific Information
    //
    void printStageSpecificInfo(slang::EntryPointReflection* entryPointLayout)
    {
        switch (entryPointLayout->getStage())
        {
        default:
            break;

        case SLANG_STAGE_COMPUTE:
            {
                static const int kAxisCount = 3;
                SlangUInt sizes[kAxisCount];
                entryPointLayout->getComputeThreadGroupSize(kAxisCount, sizes);

                key("thread group size");
                SCOPED_OBJECT();
                key("x");
                print(sizes[0]);
                key("y");
                print(sizes[1]);
                key("z");
                print(sizes[2]);
            }
            break;

        case SLANG_STAGE_FRAGMENT:
            key("uses any sample-rate inputs");
            printBool(entryPointLayout->usesAnySampleRateInput());
            break;
        }
    }

    // #### Varying Parameters
    //
    void printVaryingParameterInfo(slang::VariableLayoutReflection* variableLayout)
    {
        if (auto semanticName = variableLayout->getSemanticName())
        {
            key("semantic");
            SCOPED_OBJECT();
            key("name");
            printQuotedString(semanticName);
            key("index");
            print(variableLayout->getSemanticIndex());
        }
    }

    // Calculating Cumulative Offsets
    // ------------------------------
    //
    struct CumulativeOffset
    {
        size_t value = 0;
        size_t space = 0;
    };

    // ### Access Paths

    struct AccessPathNode
    {
        slang::VariableLayoutReflection* variableLayout = nullptr;
        AccessPathNode* outer = nullptr;
    };

    struct AccessPath
    {
        AccessPath() {}

        bool valid = false;
        AccessPathNode* deepestConstantBufer = nullptr;
        AccessPathNode* deepestParameterBlock = nullptr;
        AccessPathNode* leaf = nullptr;
    };

    void printCumulativeOffsets(
        slang::VariableLayoutReflection* variableLayout,
        AccessPath accessPath)
    {
        key("cumulative");

        int usedLayoutUnitCount = variableLayout->getCategoryCount();
        WITH_ARRAY();
        for (int i = 0; i < usedLayoutUnitCount; ++i)
        {
            element();

            auto layoutUnit = variableLayout->getCategoryByIndex(i);
            printCumulativeOffset(variableLayout, layoutUnit, accessPath);
        }
    }

    CumulativeOffset calculateCumulativeOffset(
        slang::VariableLayoutReflection* variableLayout,
        slang::ParameterCategory layoutUnit,
        AccessPath accessPath)
    {
        CumulativeOffset result = calculateCumulativeOffset(layoutUnit, accessPath);
        result.value += variableLayout->getOffset(layoutUnit);
        result.space += variableLayout->getBindingSpace(layoutUnit);
        return result;
    }

    void printCumulativeOffset(
        slang::VariableLayoutReflection* variableLayout,
        slang::ParameterCategory layoutUnit,
        AccessPath accessPath)
    {
        CumulativeOffset cumulativeOffset =
            calculateCumulativeOffset(variableLayout, layoutUnit, accessPath);

        printOffset(layoutUnit, cumulativeOffset.value, cumulativeOffset.space);
    }

    // ### Tracking Access Paths

    struct ExtendedAccessPath : AccessPath
    {
        ExtendedAccessPath(AccessPath const& base, slang::VariableLayoutReflection* variableLayout)
            : AccessPath(base)
        {
            if (!valid)
                return;

            element.variableLayout = variableLayout;
            element.outer = leaf;

            leaf = &element;
        }

        AccessPathNode element;
    };

    // ### Accumulating Offsets Along An Access Path

    CumulativeOffset calculateCumulativeOffset(
        slang::ParameterCategory layoutUnit,
        AccessPath accessPath)
    {
        CumulativeOffset result;
        switch (layoutUnit)
        {
        // #### Layout Units That Don't Require Special Handling
        //
        default:
            for (auto node = accessPath.leaf; node != nullptr; node = node->outer)
            {
                result.value += node->variableLayout->getOffset(layoutUnit);
            }
            break;

        // #### Bytes
        //
        case slang::ParameterCategory::Uniform:
            for (auto node = accessPath.leaf; node != accessPath.deepestConstantBufer;
                 node = node->outer)
            {
                result.value += node->variableLayout->getOffset(layoutUnit);
            }
            break;

        // #### Layout Units That Care About Spaces
        //
        case slang::ParameterCategory::ConstantBuffer:
        case slang::ParameterCategory::ShaderResource:
        case slang::ParameterCategory::UnorderedAccess:
        case slang::ParameterCategory::SamplerState:
        case slang::ParameterCategory::DescriptorTableSlot:
            for (auto node = accessPath.leaf; node != accessPath.deepestParameterBlock;
                 node = node->outer)
            {
                result.value += node->variableLayout->getOffset(layoutUnit);
                result.space += node->variableLayout->getBindingSpace(layoutUnit);
            }
            for (auto node = accessPath.deepestParameterBlock; node != nullptr; node = node->outer)
            {
                result.space += node->variableLayout->getOffset(
                    slang::ParameterCategory::SubElementRegisterSpace);
            }
            break;
        }
        return result;
    }

    // Determining Whether Parameters Are Used
    // ---------------------------------------

    Result collectEntryPointMetadata(
        slang::IComponentType* program,
        int targetIndex,
        int entryPointCount)
    {
        _metadataForEntryPoints.setCount(entryPointCount);
        for (int entryPointIndex = 0; entryPointIndex < entryPointCount; entryPointIndex++)
        {
            ComPtr<slang::IMetadata> entryPointMetadata;
            ComPtr<slang::IBlob> diagnostics;
            SLANG_RETURN_ON_FAIL(program->getEntryPointMetadata(
                entryPointIndex,
                targetIndex,
                entryPointMetadata.writeRef(),
                diagnostics.writeRef()));
            diagnoseIfNeeded(diagnostics);

            _metadataForEntryPoints[entryPointIndex] = entryPointMetadata;
        }
        return SLANG_OK;
    }
    Slang::List<ComPtr<slang::IMetadata>> _metadataForEntryPoints;

    typedef unsigned int StageMask;

    StageMask calculateParameterStageMask(
        slang::ParameterCategory layoutUnit,
        CumulativeOffset offset)
    {
        unsigned mask = 0;
        auto entryPointCount = _metadataForEntryPoints.getCount();
        for (int i = 0; i < entryPointCount; ++i)
        {
            bool isUsed = false;
            _metadataForEntryPoints[i]->isParameterLocationUsed(
                SlangParameterCategory(layoutUnit),
                offset.space,
                offset.value,
                isUsed);
            if (isUsed)
            {
                auto entryPointStage = _programLayout->getEntryPointByIndex(i)->getStage();

                mask |= 1 << unsigned(entryPointStage);
            }
        }
        return mask;
    }

    StageMask calculateStageMask(
        slang::VariableLayoutReflection* variableLayout,
        AccessPath accessPath)
    {
        StageMask mask = 0;

        int usedLayoutUnitCount = variableLayout->getCategoryCount();
        for (int i = 0; i < usedLayoutUnitCount; ++i)
        {
            auto layoutUnit = variableLayout->getCategoryByIndex(i);
            auto offset = calculateCumulativeOffset(variableLayout, layoutUnit, accessPath);

            mask |= calculateParameterStageMask(layoutUnit, offset);
        }

        return mask;
    }

    void printStageUsage(slang::VariableLayoutReflection* variableLayout, AccessPath accessPath)
    {
        StageMask stageMask = calculateStageMask(variableLayout, accessPath);

        key("used by stages");
        WITH_ARRAY()
        for (int i = 0; i < SLANG_STAGE_COUNT; i++)
        {
            if (stageMask & (1 << i))
            {
                element();
                printStage(SlangStage(i));
            }
        }
    }

    void printOffsets(slang::VariableLayoutReflection* variableLayout, AccessPath accessPath)
    {
        key("offset");
        {
            SCOPED_OBJECT();
            printRelativeOffsets(variableLayout);

            if (accessPath.valid)
            {
                printCumulativeOffsets(variableLayout, accessPath);
            }
        }


        if (accessPath.valid)
        {
            printStageUsage(variableLayout, accessPath);
        }
    }

    // Formatting
    // ----------
    //
    // Here we'll cover the logic for how we implement
    // the various formatting operations used in the
    // code above.
    //
    // ### Indentation
    //
    // We track a global indentation level, and whenever
    // we begin a new line, we'll emit a corresponding
    // amount of space (two spaces per indent, consistent
    // with typical YAML formatting).

    int indentation = 0;

    void printIndentation()
    {
        for (int i = 1; i < indentation; ++i)
        {
            printf("  ");
        }
    }

    // ### Objects and Arrays
    //
    // Both objects and arrays can be marked up purely
    // with indentation in YAML. If we eventually
    // change the output format to something like JSON,
    // these operations would need to do more actual
    // work.

    void beginObject() { indentation++; }

    void endObject() { indentation--; }

    void beginArray() { indentation++; }

    void endArray() { indentation--; }

    // #### Scope-Based Objects
    //
    // In order to make it easier to keep the `beginObject()`
    // and `endObject()` calls properly paired, we introduce
    // a helper type that uses an RAII idiom to automatically
    // pair up the calls.
    //
    struct ScopedObject
    {
        ScopedObject(ReflectingPrinting* outer)
            : outer(outer)
        {
            outer->beginObject();
        }

        ~ScopedObject() { outer->endObject(); }

        ReflectingPrinting* outer = nullptr;
    };

    // ### Starting New Lines
    //
    // Typically, when we are about to emit a key
    // in an object, or an element in an array,
    // we need to start a new line (and print
    // the appropriate indentation).
    //
    void newLine()
    {
        printf("\n");
        printIndentation();
    }


    // The main exception is that if we've just
    // emitted the `- ` for an array element then
    // we don't need to start a new line if
    // the next thing we emit is an object key.
    //
    // We *also* don't need to start a new line
    // at the very beginning of the output, so
    // we handle that by setting the intial state
    // *as if* we have just started an array element.

    bool afterArrayElement = true;

    // ### Array Elements
    //
    void element()
    {
        newLine();
        printf("- ");
        afterArrayElement = true;
    }

    // ### Object Keys
    //
    void key(char const* key)
    {
        if (!afterArrayElement)
        {
            newLine();
        }
        afterArrayElement = false;

        printf("%s: ", key);
    }

    // ### Printing Simple Values
    //
    // Simple scalar values like strings,
    // `bool`s, and numbers don't need
    // much special handling.

    void printQuotedString(char const* text)
    {
        if (text)
        {
            printf("\"%s\"", text);
        }
        else
        {
            printf("null");
        }
    }

    void printBool(bool value) { printf(value ? "true" : "false"); }

    void print(size_t value) { printf("%u", unsigned(value)); }

    // YAML supports comments, but JSON doesn't.
    // This function could be stubbed out if
    // we switch up the output format.
    //
    void printComment(char const* text) { printf("# %s", text); }


    // Printing Enumerants
    // -------------------
    //
    // Here we'll gather all the logic for printing the various
    // `enum` types that we've worked with in the logic above.

    void printTypeKind(slang::TypeReflection::Kind kind)
    {
        switch (kind)
        {
#define CASE(TAG)                          \
    case slang::TypeReflection::Kind::TAG: \
        printf("%s", #TAG);                \
        break

            CASE(None);
            CASE(Struct);
            CASE(Array);
            CASE(Matrix);
            CASE(Vector);
            CASE(Scalar);
            CASE(ConstantBuffer);
            CASE(Resource);
            CASE(SamplerState);
            CASE(TextureBuffer);
            CASE(ShaderStorageBuffer);
            CASE(ParameterBlock);
            CASE(GenericTypeParameter);
            CASE(Interface);
            CASE(OutputStream);
            CASE(Specialized);
            CASE(Feedback);
            CASE(Pointer);
            CASE(DynamicResource);
#undef CASE

        default:
            printf("%d # unexpected enumerant", int(kind));
            break;
        }
    }

    void printResourceShape(SlangResourceShape shape)
    {
        SCOPED_OBJECT();

        key("base");
        auto baseShape = shape & SLANG_RESOURCE_BASE_SHAPE_MASK;
        switch (baseShape)
        {
#define CASE(TAG)           \
    case SLANG_##TAG:       \
        printf("%s", #TAG); \
        break

            CASE(TEXTURE_1D);
            CASE(TEXTURE_2D);
            CASE(TEXTURE_3D);
            CASE(TEXTURE_CUBE);
            CASE(TEXTURE_BUFFER);
            CASE(STRUCTURED_BUFFER);
            CASE(BYTE_ADDRESS_BUFFER);
            CASE(RESOURCE_UNKNOWN);
            CASE(ACCELERATION_STRUCTURE);
            CASE(TEXTURE_SUBPASS);
#undef CASE

        default:
            printf("%d # unexpected enumerant", int(baseShape));
            break;
        }

#define CASE(TAG)                               \
    do                                          \
    {                                           \
        if (shape & SLANG_TEXTURE_##TAG##_FLAG) \
        {                                       \
            key(#TAG);                          \
            printf("true");                     \
        }                                       \
    } while (0)

        CASE(FEEDBACK);
        CASE(SHADOW);
        CASE(ARRAY);
        CASE(MULTISAMPLE);
#undef CASE
    }

    void printResourceAccess(SlangResourceAccess access)
    {
        switch (access)
        {
#define CASE(TAG)                     \
    case SLANG_RESOURCE_ACCESS_##TAG: \
        printf("%s", #TAG);           \
        break

            CASE(NONE);
            CASE(READ);
            CASE(READ_WRITE);
            CASE(RASTER_ORDERED);
            CASE(APPEND);
            CASE(CONSUME);
            CASE(WRITE);
            CASE(FEEDBACK);
#undef CASE

        default:
            printf("%d # unexpected enumerant", int(access));
            break;
        }
    }

    void printLayoutUnit(slang::ParameterCategory layoutUnit)
    {
        switch (layoutUnit)
        {
#define CASE(TAG, DESCRIPTION)                \
    case slang::ParameterCategory::TAG:       \
        printf("%s # %s", #TAG, DESCRIPTION); \
        break

            CASE(ConstantBuffer, "constant buffer slots");
            CASE(ShaderResource, "texture slots");
            CASE(UnorderedAccess, "uav slots");
            CASE(VaryingInput, "varying input slots");
            CASE(VaryingOutput, "varying output slots");
            CASE(SamplerState, "sampler slots");
            CASE(Uniform, "bytes");
            CASE(DescriptorTableSlot, "bindings");
            CASE(SpecializationConstant, "specialization constant ids");
            CASE(PushConstantBuffer, "push-constant buffers");
            CASE(RegisterSpace, "register space offset for a variable");
            CASE(GenericResource, "generic resources");
            CASE(RayPayload, "ray payloads");
            CASE(HitAttributes, "hit attributes");
            CASE(CallablePayload, "callable payloads");
            CASE(ShaderRecord, "shader records");
            CASE(ExistentialTypeParam, "existential type parameters");
            CASE(ExistentialObjectParam, "existential object parameters");
            CASE(SubElementRegisterSpace, "register spaces / descriptor sets");
            CASE(InputAttachmentIndex, "subpass input attachments");
            CASE(MetalArgumentBufferElement, "Metal argument buffer elements");
            CASE(MetalAttribute, "Metal attributes");
            CASE(MetalPayload, "Metal payloads");
#undef CASE

        default:
            printf("%d # unknown enumerant", int(layoutUnit));
            break;
        }
    }

    void printStage(SlangStage stage)
    {
        switch (stage)
        {
#define CASE(NAME)           \
    case SLANG_STAGE_##NAME: \
        printf(#NAME);       \
        break

            CASE(NONE);
            CASE(VERTEX);
            CASE(HULL);
            CASE(DOMAIN);
            CASE(GEOMETRY);
            CASE(FRAGMENT);
            CASE(COMPUTE);
            CASE(RAY_GENERATION);
            CASE(INTERSECTION);
            CASE(ANY_HIT);
            CASE(CLOSEST_HIT);
            CASE(MISS);
            CASE(CALLABLE);
            CASE(MESH);
            CASE(AMPLIFICATION);
#undef CASE

        default:
            printf("%d # unexpected enumerant", int(stage));
            break;
        };
    }
    void printTargetFormat(SlangCompileTarget targetFormat)
    {
        switch (targetFormat)
        {
#define CASE(TAG)           \
    case SLANG_##TAG:       \
        printf("%s", #TAG); \
        break

            CASE(TARGET_UNKNOWN);
            CASE(TARGET_NONE);
            CASE(GLSL);
            CASE(GLSL_VULKAN_DEPRECATED);
            CASE(GLSL_VULKAN_ONE_DESC_DEPRECATED);
            CASE(HLSL);
            CASE(SPIRV);
            CASE(SPIRV_ASM);
            CASE(DXBC);
            CASE(DXBC_ASM);
            CASE(DXIL);
            CASE(DXIL_ASM);
            CASE(C_SOURCE);
            CASE(CPP_SOURCE);
            CASE(HOST_EXECUTABLE);
            CASE(SHADER_SHARED_LIBRARY);
            CASE(SHADER_HOST_CALLABLE);
            CASE(CUDA_SOURCE);
            CASE(PTX);
            CASE(CUDA_OBJECT_CODE);
            CASE(OBJECT_CODE);
            CASE(HOST_CPP_SOURCE);
            CASE(HOST_HOST_CALLABLE);
            CASE(CPP_PYTORCH_BINDING);
            CASE(METAL);
            CASE(METAL_LIB);
            CASE(METAL_LIB_ASM);
            CASE(HOST_SHARED_LIBRARY);
            CASE(WGSL);
            CASE(WGSL_SPIRV_ASM);
            CASE(WGSL_SPIRV);
#undef CASE

        default:
            printf("%d # unhandled enumerant", int(targetFormat));
        }
    }

    void printScalarType(slang::TypeReflection::ScalarType scalarType)
    {
        switch (scalarType)
        {
#define CASE(TAG)                    \
    case slang::TypeReflection::TAG: \
        printf("%s", #TAG);          \
        break

            CASE(None);
            CASE(Void);
            CASE(Bool);
            CASE(Int32);
            CASE(UInt32);
            CASE(Int64);
            CASE(UInt64);
            CASE(Float16);
            CASE(Float32);
            CASE(Float64);
            CASE(Int8);
            CASE(UInt8);
            CASE(Int16);
            CASE(UInt16);
#undef CASE

        default:
            printf("%d # unhandled enumerant", int(scalarType));
        }
    }

    void printMatrixLayoutMode(SlangMatrixLayoutMode mode)
    {
        switch (mode)
        {
#define CASE(TAG)                   \
    case SLANG_MATRIX_LAYOUT_##TAG: \
        printf("%s", #TAG);         \
        break

            CASE(MODE_UNKNOWN);
            CASE(ROW_MAJOR);
            CASE(COLUMN_MAJOR);
#undef CASE

        default:
            printf("%d # unhandled enumerant", int(mode));
        }
    }
};

struct ExampleProgram : public TestBase
{
    Result execute(int argc, char* argv[])
    {
        parseOption(argc, argv);

        ComPtr<slang::IGlobalSession> globalSession;
        SLANG_RETURN_ON_FAIL(slang::createGlobalSession(globalSession.writeRef()));

        Slang::List<slang::TargetDesc> targetDescs;
        for (auto target : kTargets)
        {
            auto profile = globalSession->findProfile(target.profile);

            slang::TargetDesc targetDesc;
            targetDesc.format = target.format;
            targetDesc.profile = profile;
            targetDescs.add(targetDesc);
        }

        slang::SessionDesc sessionDesc;
        sessionDesc.targetCount = targetDescs.getCount();
        sessionDesc.targets = targetDescs.getBuffer();

        ComPtr<slang::ISession> session;
        SLANG_RETURN_ON_FAIL(globalSession->createSession(sessionDesc, session.writeRef()));

        ReflectingPrinting printingContext;
        printingContext.compileAndReflectPrograms(session);

        return SLANG_OK;
    }
};

int exampleMain(int argc, char** argv)
{
    ExampleProgram app;
    if (SLANG_FAILED(app.execute(argc, argv)))
    {
        return -1;
    }
    return 0;
}
