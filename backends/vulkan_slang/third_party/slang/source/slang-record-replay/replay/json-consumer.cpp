#include "json-consumer.h"

#include "../../core/slang-io.h"
#include "../../core/slang-string-util.h"
#include "../util/emum-to-string.h"
#include "../util/record-utility.h"
#include "slang.h"

namespace SlangRecord
{
#define SANITY_CHECK()  \
    if (!m_isFileValid) \
    return

static inline void _writeIndent(Slang::StringBuilder& builder, int indent)
{
    for (int i = 0; i < indent; i++)
    {
        builder << "  ";
    }
}

template<typename T>
static inline void _writeString(Slang::StringBuilder& builder, int indent, T const& str)
{
    _writeIndent(builder, indent);
    builder << str;
}

template<typename T, typename U>
static inline void _writePair(
    Slang::StringBuilder& builder,
    int indent,
    T const& name,
    U const& value)
{
    _writeIndent(builder, indent);
    builder << name << ": " << value << ",\n";
}

template<typename T, typename U>
static inline void _writePairNoComma(
    Slang::StringBuilder& builder,
    int indent,
    T const& name,
    U const& value)
{
    _writeIndent(builder, indent);
    builder << name << ": " << value << "\n";
}

class ScopeWritterForKey
{
public:
    ScopeWritterForKey(
        Slang::StringBuilder* pBuilder,
        int* pIndent,
        Slang::String const& keyName,
        bool isOutterScope = true)
        : m_pBuilder(pBuilder), m_pIndent(pIndent), m_isOutterScope(isOutterScope)
    {
        _writeString((*m_pBuilder), (*m_pIndent), keyName + ": {\n");
        (*m_pIndent)++;
    }

    ~ScopeWritterForKey()
    {
        (*m_pIndent)--;

        if (m_isOutterScope)
            _writeString((*m_pBuilder), (*m_pIndent), "}\n");
        else
            _writeString((*m_pBuilder), (*m_pIndent), "},\n");
    }

private:
    Slang::StringBuilder* m_pBuilder;
    int* m_pIndent;
    bool m_isOutterScope;
};

void CommonInterfaceWriter::getSession(ObjectID objectId, ObjectID outSessionId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getSession";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "retSession",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outSessionId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getLayout";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "targetIndex", targetIndex);
            _writePair(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
            _writePairNoComma(
                builder,
                indent,
                "retProgramLayout",
                Slang::StringUtil::makeStringWithFormat("0x%llX", retProgramLayoutId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getEntryPointCode";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "entryPointIndex", entryPointIndex);
            _writePair(builder, indent, "targetIndex", targetIndex);
            _writePair(
                builder,
                indent,
                "outCode",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outCodeId));
            _writePairNoComma(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getTargetCode";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "targetIndex", targetIndex);
            _writePair(
                builder,
                indent,
                "outCode",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outCodeId));
            _writePairNoComma(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystemId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getResultAsFileSystem";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "entryPointIndex", entryPointIndex);
            _writePair(builder, indent, "targetIndex", targetIndex);
            _writePairNoComma(
                builder,
                indent,
                "outFileSystem",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outFileSystemId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getEntryPointHash";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "entryPointIndex", entryPointIndex);
            _writePair(builder, indent, "targetIndex", targetIndex);
            _writePairNoComma(
                builder,
                indent,
                "outHash",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outHashId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::specialize";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            if (specializationArgCount)
            {
                ScopeWritterForKey scopeWritterForArgs(
                    &builder,
                    &indent,
                    "specializationArgs",
                    false);
                for (int i = 0; i < specializationArgCount; i++)
                {
                    bool isLastField = (i == specializationArgCount - 1);
                    ScopeWritterForKey scopeWritterForArg(
                        &builder,
                        &indent,
                        Slang::StringUtil::makeStringWithFormat("[%d]", i),
                        isLastField);
                    {
                        _writePair(
                            builder,
                            indent,
                            "kind",
                            SpecializationArgKindToString(specializationArgs[i].kind));
                        _writePairNoComma(
                            builder,
                            indent,
                            "type",
                            Slang::StringUtil::makeStringWithFormat(
                                "0x%llX",
                                specializationArgs[i].type));
                    }
                }
            }
            else
            {
                _writePair(builder, indent, "specializationArgs", "nullptr");
            }
            _writePair(builder, indent, "specializationArgCount", specializationArgCount);
            _writePair(
                builder,
                indent,
                "outSpecializedComponentType",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outSpecializedComponentTypeId));
            _writePairNoComma(
                builder,
                indent,
                "outSpecializedComponentType",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::link";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "outLinkedComponentType",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outLinkedComponentTypeId));
            _writePairNoComma(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibraryId,
    ObjectID outDiagnosticsId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::getEntryPointHostCallable";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "entryPointIndex", entryPointIndex);
            _writePair(builder, indent, "targetIndex", targetIndex);
            _writePair(
                builder,
                indent,
                "outSharedLibrary",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outSharedLibraryId));
            _writePairNoComma(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::renameEntryPoint";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "newName",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    newName != nullptr ? newName : "nullptr"));
            _writePairNoComma(
                builder,
                indent,
                "outEntryPoint",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outEntryPointId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}

void CommonInterfaceWriter::linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{

    Slang::StringBuilder builder;
    int indent = 0;

    Slang::String functionName = m_className;
    functionName = functionName + "::linkWithOptions";

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, functionName);
        {

            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "compilerOptionEntryCount", compilerOptionEntryCount);

            JsonConsumer::_writeCompilerOptionEntryHelper(
                builder,
                indent,
                compilerOptionEntries,
                compilerOptionEntryCount);

            _writePair(
                builder,
                indent,
                "outLinkedComponentType",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outLinkedComponentTypeId));
            _writePairNoComma(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        }
    }

    m_fileStream.write(builder.begin(), builder.getLength());
    m_fileStream.flush();
}


JsonConsumer::JsonConsumer(const Slang::String& filePath)
{
    if (!Slang::File::exists(Slang::Path::getParentDirectory(filePath)))
    {
        slangRecordLog(
            LogLevel::Error,
            "Directory for json file does not exist: %s\n",
            filePath.getBuffer());
    }

    Slang::FileMode fileMode = Slang::FileMode::Create;
    Slang::FileAccess fileAccess = Slang::FileAccess::Write;
    Slang::FileShare fileShare = Slang::FileShare::None;

    SlangResult res = m_fileStream.init(filePath, fileMode, fileAccess, fileShare);

    if (res != SLANG_OK)
    {
        slangRecordLog(LogLevel::Error, "Failed to open file %s\n", filePath.getBuffer());
    }

    m_isFileValid = true;
}

void JsonConsumer::_writeCompilerOptionEntryHelper(
    Slang::StringBuilder& builder,
    int indent,
    slang::CompilerOptionEntry* compilerOptionEntries,
    uint32_t compilerOptionEntryCount,
    bool isLastField)
{
    if (compilerOptionEntryCount)
    {
        ScopeWritterForKey scopeWritterForCompilerOptionEntries(
            &builder,
            &indent,
            "compilerOptionEntries",
            isLastField);

        for (uint32_t j = 0; j < compilerOptionEntryCount; j++)
        {
            ScopeWritterForKey scopeWritterForCompileOptionElement(
                &builder,
                &indent,
                Slang::StringUtil::makeStringWithFormat("[%d]\n", j));
            {
                _writePair(
                    builder,
                    indent,
                    "name",
                    CompilerOptionNameToString(compilerOptionEntries[j].name));

                bool isLastEntry = (j == compilerOptionEntryCount - 1);
                ScopeWritterForKey scopeWritterValue(&builder, &indent, "value", isLastEntry);
                {
                    _writePair(
                        builder,
                        indent,
                        "kind",
                        CompilerOptionValueKindToString(compilerOptionEntries[j].value.kind));
                    _writePair(
                        builder,
                        indent,
                        "intValue0",
                        compilerOptionEntries[j].value.intValue0);
                    _writePair(
                        builder,
                        indent,
                        "intValue1",
                        compilerOptionEntries[j].value.intValue1);
                    _writePair(
                        builder,
                        indent,
                        "stringValue0",
                        compilerOptionEntries[j].value.stringValue0);
                    _writePairNoComma(
                        builder,
                        indent,
                        "stringValue1",
                        compilerOptionEntries[j].value.stringValue1);
                }
            }
        }
    }
    else
    {
        _writePairNoComma(builder, indent, "compilerOptionEntries", "nullptr");
    }
}

void JsonConsumer::_writeGlobalSessionDescHelper(
    Slang::StringBuilder& builder,
    int indent,
    SlangGlobalSessionDesc const& desc,
    Slang::String keyName,
    bool isLastField)
{
    SLANG_UNUSED(isLastField);

    ScopeWritterForKey scopeWritterForGlobalSessionDesc(&builder, &indent, keyName);
    {
        _writePair(builder, indent, "structureSize", (uint32_t)desc.structureSize);
        _writePair(builder, indent, "apiVersion", (uint32_t)desc.apiVersion);
        _writePair(builder, indent, "languageVersion", (uint32_t)desc.languageVersion);
        _writePair(builder, indent, "enablGLSL", (uint32_t)desc.enableGLSL);
    }
}

void JsonConsumer::_writeSessionDescHelper(
    Slang::StringBuilder& builder,
    int indent,
    slang::SessionDesc const& desc,
    Slang::String keyName,
    bool isLastField)
{
    ScopeWritterForKey scopeWritterForSessionDesc(&builder, &indent, keyName);
    {
        _writePair(builder, indent, "structureSize", (uint32_t)desc.structureSize);

        if (desc.targetCount)
        {
            ScopeWritterForKey scopeWritterForTarget(
                &builder,
                &indent,
                Slang::StringUtil::makeStringWithFormat("targets (0x%llX)", desc.targets),
                isLastField);
            {
                for (int i = 0; i < desc.targetCount; i++)
                {
                    bool isLastEntry = (i == desc.targetCount - 1);
                    ScopeWritterForKey scopeWritterForTargetElement(
                        &builder,
                        &indent,
                        Slang::StringUtil::makeStringWithFormat("[%d]", i),
                        isLastEntry);
                    {
                        _writePair(
                            builder,
                            indent,
                            "structureSize",
                            (uint32_t)desc.targets[i].structureSize);
                        _writePair(
                            builder,
                            indent,
                            "format",
                            SlangCompileTargetToString(desc.targets[i].format));
                        _writePair(
                            builder,
                            indent,
                            "profile",
                            SlangProfileIDToString(desc.targets[i].profile));
                        _writePair(
                            builder,
                            indent,
                            "flags",
                            SlangTargetFlagsToString(desc.targets[i].flags));
                        _writePair(
                            builder,
                            indent,
                            "floatingPointMode",
                            SlangFloatingPointModeToString(desc.targets[i].floatingPointMode));
                        _writePair(
                            builder,
                            indent,
                            "lineDirectiveMode",
                            SlangLineDirectiveModeToString(desc.targets[i].lineDirectiveMode));
                        _writePair(
                            builder,
                            indent,
                            "forceGLSLScalarBufferLayout",
                            (desc.targets[i].floatingPointMode ? "true" : "false"));

                        _writeCompilerOptionEntryHelper(
                            builder,
                            indent,
                            desc.targets[i].compilerOptionEntries,
                            desc.targets[i].compilerOptionEntryCount);
                    }
                }
            }
        }
        else
        {
            _writePair(builder, indent, "targets", "nullptr");
        }

        _writePair(builder, indent, "targetCount", desc.targetCount);
        _writePair(builder, indent, "flags", SessionFlagsToString(desc.flags));
        _writePair(
            builder,
            indent,
            "defaultMatrixLayoutMode",
            SlangMatrixLayoutModeToString(desc.defaultMatrixLayoutMode));

        if (desc.searchPathCount)
        {
            ScopeWritterForKey scopeWritterForSearchPath(&builder, &indent, "searchPaths", false);
            for (int i = 0; i < desc.searchPathCount; i++)
            {
                Slang::String searchPath(desc.searchPaths[i]);
                searchPath = searchPath + ",\n";
                _writeString(builder, indent, searchPath);
            }
        }
        else
        {
            _writePair(builder, indent, "searchPaths", "nullptr");
        }
        _writePair(builder, indent, "searchPathCount", desc.searchPathCount);

        if (desc.preprocessorMacroCount)
        {
            ScopeWritterForKey scopeWritterForMacro(&builder, &indent, "preprocessorMacros", false);
            for (int i = 0; i < desc.preprocessorMacroCount; i++)
            {
                bool isLastField = (i == desc.preprocessorMacroCount - 1);
                ScopeWritterForKey scopeWritterForMacroElement(
                    &builder,
                    &indent,
                    Slang::StringUtil::makeStringWithFormat("[%d]", i),
                    isLastField);

                _writePair(
                    builder,
                    indent,
                    "name",
                    Slang::StringUtil::makeStringWithFormat(
                        "\"%s\"",
                        desc.preprocessorMacros[i].name != nullptr ? desc.preprocessorMacros[i].name
                                                                   : "nullptr"));

                _writePairNoComma(
                    builder,
                    indent,
                    "value",
                    Slang::StringUtil::makeStringWithFormat(
                        "\"%s\"",
                        desc.preprocessorMacros[i].value != nullptr
                            ? desc.preprocessorMacros[i].value
                            : "nullptr"));
            }
        }
        else
        {
            _writePair(builder, indent, "preprocessorMacros", "nullptr");
        }
        _writePair(builder, indent, "preprocessorMacroCount", desc.preprocessorMacroCount);

        AddressFormat address = reinterpret_cast<AddressFormat>(desc.fileSystem);
        _writePair(
            builder,
            indent,
            "fileSystem",
            Slang::StringUtil::makeStringWithFormat("0x%llX", address));
        _writePair(
            builder,
            indent,
            "enableEffectAnnotations",
            (desc.enableEffectAnnotations ? "true" : "false"));
        _writePair(builder, indent, "allowGLSLSyntax", (desc.allowGLSLSyntax ? "true" : "false"));
        _writePair(builder, indent, "compilerOptionEntryCount", desc.compilerOptionEntryCount);
        _writeCompilerOptionEntryHelper(
            builder,
            indent,
            desc.compilerOptionEntries,
            desc.compilerOptionEntryCount);
    }
}

void JsonConsumer::CreateGlobalSession(
    SlangGlobalSessionDesc const& desc,
    ObjectID outGlobalSessionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::createGlobalSession");
        _writeGlobalSessionDescHelper(builder, indent, desc, "inDesc");
        _writePairNoComma(
            builder,
            indent,
            "outGlobalSession",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outGlobalSessionId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::IGlobalSession_createSession(
    ObjectID objectId,
    slang::SessionDesc const& desc,
    ObjectID outSessionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::createSession");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));

            _writeSessionDescHelper(builder, indent, desc, "inDesc");

            _writePairNoComma(
                builder,
                indent,
                "outSession",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outSessionId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::IGlobalSession_findProfile(ObjectID objectId, char const* name)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::findProfile");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "name",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    name != nullptr ? name : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setDownstreamCompilerPath(
    ObjectID objectId,
    SlangPassThrough passThrough,
    char const* path)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::setDownstreamCompilerPath");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "passThrough", SlangPassThroughToString(passThrough));
            _writePairNoComma(
                builder,
                indent,
                "path",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    path != nullptr ? path : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setDownstreamCompilerPrelude(
    ObjectID objectId,
    SlangPassThrough inPassThrough,
    char const* prelude)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::setDownstreamCompilerPrelude");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "passThrough", SlangPassThroughToString(inPassThrough));
            _writePairNoComma(
                builder,
                indent,
                "preludeText",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    prelude != nullptr ? prelude : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_getDownstreamCompilerPrelude(
    ObjectID objectId,
    SlangPassThrough inPassThrough,
    ObjectID outPreludeId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::getDownstreamCompilerPrelude");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "passThrough", SlangPassThroughToString(inPassThrough));
            _writePairNoComma(
                builder,
                indent,
                "outPrelude",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outPreludeId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setDefaultDownstreamCompiler(
    ObjectID objectId,
    SlangSourceLanguage sourceLanguage,
    SlangPassThrough defaultCompiler)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::setDefaultDownstreamCompiler");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "sourceLanguage",
                SlangSourceLanguageToString(sourceLanguage));
            _writePairNoComma(
                builder,
                indent,
                "defaultCompiler",
                SlangPassThroughToString(defaultCompiler));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_getDefaultDownstreamCompiler(
    ObjectID objectId,
    SlangSourceLanguage sourceLanguage)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::getDefaultDownstreamCompiler");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "sourceLanguage",
                SlangSourceLanguageToString(sourceLanguage));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setLanguagePrelude(
    ObjectID objectId,
    SlangSourceLanguage inSourceLanguage,
    char const* prelude)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::setLanguagePrelude");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "sourceLanguage",
                SlangSourceLanguageToString(inSourceLanguage));
            _writePairNoComma(
                builder,
                indent,
                "preludeText",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    prelude != nullptr ? prelude : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_getLanguagePrelude(
    ObjectID objectId,
    SlangSourceLanguage inSourceLanguage,
    ObjectID outPreludeId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::getLanguagePrelude");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "sourceLanguage",
                SlangSourceLanguageToString(inSourceLanguage));
            _writePairNoComma(
                builder,
                indent,
                "outPrelude",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outPreludeId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_createCompileRequest(
    ObjectID objectId,
    ObjectID outCompileRequest)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;
    _writeString(builder, indent, "IGlobalSession::createCompileRequest: {\n");

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::createCompileRequest");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "outCompileRequest",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outCompileRequest));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_addBuiltins(
    ObjectID objectId,
    char const* sourcePath,
    char const* sourceString)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;
    _writeString(builder, indent, "IGlobalSession::addBuiltins: {\n");

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::addBuiltins");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "sourcePath",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    sourcePath != nullptr ? sourcePath : "nullptr"));
            _writePairNoComma(
                builder,
                indent,
                "sourceString",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    sourceString != nullptr ? sourceString : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setSharedLibraryLoader(ObjectID objectId, ObjectID loaderId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::setSharedLibraryLoader");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "loader",
                Slang::StringUtil::makeStringWithFormat("0x%llX", loaderId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_getSharedLibraryLoader(ObjectID objectId, ObjectID outLoaderId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::getSharedLibraryLoader");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "retLoader",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outLoaderId));
        }
    }

    _writeString(builder, indent, "}\n");

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_checkCompileTargetSupport(
    ObjectID objectId,
    SlangCompileTarget target)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::checkCompileTargetSupport");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(builder, indent, "target", SlangCompileTargetToString(target));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_checkPassThroughSupport(
    ObjectID objectId,
    SlangPassThrough passThrough)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::checkPassThroughSupport");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "passThrough",
                SlangPassThroughToString(passThrough));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_compileCoreModule(
    ObjectID objectId,
    slang::CompileCoreModuleFlags flags)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::compileCoreModule");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(builder, indent, "flags", CompileCoreModuleFlagsToString(flags));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_loadCoreModule(
    ObjectID objectId,
    const void* coreModule,
    size_t coreModuleSizeInBytes)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;
    _writeString(builder, indent, "IGlobalSession::loadCoreModule: {\n");

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::loadCoreModule");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "coreModule-Ignore-Data",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "coreModuleSizeInBytes",
                (uint32_t)coreModuleSizeInBytes);
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::IGlobalSession_saveCoreModule(
    ObjectID objectId,
    SlangArchiveType archiveType,
    ObjectID outBlobId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::saveCoreModule");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "archiveType", SlangArchiveTypeToString(archiveType));
            _writePairNoComma(
                builder,
                indent,
                "outBlobId",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outBlobId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_findCapability(ObjectID objectId, char const* name)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::findCapability");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "name",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    name != nullptr ? name : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setDownstreamCompilerForTransition(
    ObjectID objectId,
    SlangCompileTarget source,
    SlangCompileTarget target,
    SlangPassThrough compiler)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::setDownstreamCompilerForTransition");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "source", SlangCompileTargetToString(source));
            _writePair(builder, indent, "target", SlangCompileTargetToString(target));
            _writePairNoComma(builder, indent, "compiler", SlangPassThroughToString(compiler));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_getDownstreamCompilerForTransition(
    ObjectID objectId,
    SlangCompileTarget source,
    SlangCompileTarget target)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::getDownstreamCompilerForTransition");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "source", SlangCompileTargetToString(source));
            _writePairNoComma(builder, indent, "target", SlangCompileTargetToString(target));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_setSPIRVCoreGrammar(ObjectID objectId, char const* jsonPath)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::setSPIRVCoreGrammar");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "jsonPath",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    jsonPath != nullptr ? jsonPath : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_parseCommandLineArguments(
    ObjectID objectId,
    int argc,
    const char* const* argv,
    ObjectID outSessionDescId,
    ObjectID outAllocationId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "IGlobalSession::parseCommandLineArguments");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(builder, indent, "argc", argc);

        if (argv)
        {
            _writeString(builder, indent, "argv: {\n");
            ScopeWritterForKey scopeWritteriForArgv(&builder, &indent, "argv");
            for (int i = 0; i < argc; i++)
            {
                if (i == (argc - 1))
                    _writePairNoComma(
                        builder,
                        indent,
                        Slang::StringUtil::makeStringWithFormat("[%d]", i),
                        argv[i]);
                else
                    _writePair(
                        builder,
                        indent,
                        Slang::StringUtil::makeStringWithFormat("[%d]", i),
                        argv[i]);
            }
        }
        else
        {
            _writePair(builder, indent, "argv", "nullptr");
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IGlobalSession_getSessionDescDigest(
    ObjectID objectId,
    slang::SessionDesc* sessionDesc,
    ObjectID outBlobId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IGlobalSession::getSessionDescDigest");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));

        if (sessionDesc)
        {
            _writeSessionDescHelper(
                builder,
                indent,
                *sessionDesc,
                Slang::StringUtil::makeStringWithFormat("sessionDesc (0x%llX)\n", sessionDesc));
        }
        else
        {
            _writePair(builder, indent, "sessionDesc", "nullptr");
        }

        _writePair(
            builder,
            indent,
            "outBlob",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outBlobId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

// ISession
void JsonConsumer::ISession_getGlobalSession(ObjectID objectId, ObjectID outGlobalSessionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::getGlobalSession");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "retGlobalSession",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outGlobalSessionId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_loadModule(
    ObjectID objectId,
    const char* moduleName,
    ObjectID outDiagnostics,
    ObjectID outModuleId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::loadModule");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "moduleName",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    moduleName != nullptr ? moduleName : "nullptr"));
            _writePair(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnostics));
            _writePairNoComma(
                builder,
                indent,
                "retIModule",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outModuleId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_loadModuleFromIRBlob(
    ObjectID objectId,
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::loadModuleFromIRBlob");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "moduleName",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    moduleName != nullptr ? moduleName : "nullptr"));
            _writePair(
                builder,
                indent,
                "path",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    path != nullptr ? path : "nullptr"));
            if (source)
            {
                void const* bufPtr = source->getBufferPointer();
                size_t bufSize = source->getBufferSize();

                ScopeWritterForKey scopeWritterForSource(
                    &builder,
                    &indent,
                    Slang::StringUtil::makeStringWithFormat("source (0x%llX): {\n", source),
                    false);
                {
                    _writePair(
                        builder,
                        indent,
                        "bufferPointer",
                        Slang::StringUtil::makeStringWithFormat("0x%llX", bufPtr));
                    _writePairNoComma(builder, indent, "bufferSize", (uint32_t)bufSize);
                }
            }
            else
            {
                _writePair(builder, indent, "source", "nullptr");
            }

            _writePair(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
            _writePairNoComma(
                builder,
                indent,
                "retIModule",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outModuleId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_loadModuleFromSource(
    ObjectID objectId,
    const char* moduleName,
    const char* path,
    slang::IBlob* source,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::loadModuleFromSource");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "moduleName",
            Slang::StringUtil::makeStringWithFormat(
                "\"%s\"",
                moduleName != nullptr ? moduleName : "nullptr"));
        _writePair(
            builder,
            indent,
            "path",
            Slang::StringUtil::makeStringWithFormat("\"%s\"", path != nullptr ? path : "nullptr"));
        if (source)
        {
            void const* bufPtr = source->getBufferPointer();
            size_t bufSize = source->getBufferSize();
            ScopeWritterForKey scopeWritterForSource(
                &builder,
                &indent,
                Slang::StringUtil::makeStringWithFormat("source (0x%llX): {\n", source),
                false);
            {
                _writePair(
                    builder,
                    indent,
                    "bufferPointer",
                    Slang::StringUtil::makeStringWithFormat("0x%llX", bufPtr));
                _writePairNoComma(builder, indent, "bufferSize", (uint32_t)bufSize);
            }
        }
        else
        {
            _writePair(builder, indent, "source", "nullptr");
        }

        _writePair(
            builder,
            indent,
            "outDiagnostics",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        _writePairNoComma(
            builder,
            indent,
            "retIModule",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outModuleId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_loadModuleFromSourceString(
    ObjectID objectId,
    const char* moduleName,
    const char* path,
    const char* string,
    ObjectID outDiagnosticsId,
    ObjectID outModuleId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::loadModuleFromSourceString");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "moduleName",
            Slang::StringUtil::makeStringWithFormat(
                "\"%s\"",
                moduleName != nullptr ? moduleName : "nullptr"));

        _writePair(
            builder,
            indent,
            "path",
            Slang::StringUtil::makeStringWithFormat("\"%s\"", path != nullptr ? path : "nullptr"));

        _writePair(
            builder,
            indent,
            "string",
            Slang::StringUtil::makeStringWithFormat(
                "\"%s\"",
                string != nullptr ? string : "nullptr"));

        _writePair(
            builder,
            indent,
            "outDiagnostics",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        _writePairNoComma(
            builder,
            indent,
            "retIModule",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outModuleId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_createCompositeComponentType(
    ObjectID objectId,
    ObjectID* componentTypeIds,
    SlangInt componentTypeCount,
    ObjectID outCompositeComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "ISession::createCompositeComponentType");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            if (componentTypeCount)
            {
                ScopeWritterForKey scopeWritterForComponentTypes(
                    &builder,
                    &indent,
                    "componentTypes",
                    false);
                for (int i = 0; i < componentTypeCount; i++)
                {
                    if (i != componentTypeCount - 1)
                    {
                        _writeString(
                            builder,
                            indent,
                            Slang::StringUtil::makeStringWithFormat(
                                "[%d]: 0x%llX,\n",
                                i,
                                componentTypeIds[i]));
                    }
                    else
                    {
                        _writeString(
                            builder,
                            indent,
                            Slang::StringUtil::makeStringWithFormat(
                                "[%d]: 0x%llX\n",
                                i,
                                componentTypeIds[i]));
                    }
                }
            }
            else
            {
                _writePair(builder, indent, "componentTypes", "nullptr");
            }
        }
        _writePair(builder, indent, "componentTypeCount", componentTypeCount);
        _writePair(
            builder,
            indent,
            "outCompositeComponentType",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outCompositeComponentTypeId));
        _writePairNoComma(
            builder,
            indent,
            "outDiagnostics",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_specializeType(
    ObjectID objectId,
    ObjectID typeId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outDiagnosticsId,
    ObjectID outTypeReflectionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::specializeType");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "type",
            Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));

        if (specializationArgCount)
        {
            ScopeWritterForKey scopeWritterForArgs(&builder, &indent, "specializationArgs", false);
            for (int i = 0; i < specializationArgCount; i++)
            {
                ScopeWritterForKey scopeWritterForArg(
                    &builder,
                    &indent,
                    Slang::StringUtil::makeStringWithFormat("[%d]\n", i),
                    false);
                {
                    _writePair(
                        builder,
                        indent,
                        "kind",
                        SpecializationArgKindToString(specializationArgs[i].kind));
                    _writePairNoComma(
                        builder,
                        indent,
                        "type",
                        Slang::StringUtil::makeStringWithFormat(
                            "0x%llX",
                            specializationArgs[i].type));
                }
            }
        }
        else
        {
            _writePair(builder, indent, "specializationArgs", "nullptr");
        }

        _writePair(builder, indent, "specializationArgCount", specializationArgCount);
        _writePair(builder, indent, "outDiagnostics", outDiagnosticsId);
        _writePairNoComma(builder, indent, "retTypeReflectionId", outTypeReflectionId);
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_getTypeLayout(
    ObjectID objectId,
    ObjectID typeId,
    SlangInt targetIndex,
    slang::LayoutRules rules,
    ObjectID outDiagnosticsId,
    ObjectID outTypeLayoutReflectionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::getTypeLayout");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "type",
                Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));
            _writePair(builder, indent, "rules", LayoutRulesToString(rules));
            _writePair(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
            _writePairNoComma(
                builder,
                indent,
                "retTypeReflectionId",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outTypeLayoutReflectionId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_getContainerType(
    ObjectID objectId,
    ObjectID elementTypeId,
    slang::ContainerType containerType,
    ObjectID outDiagnosticsId,
    ObjectID outTypeReflectionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::getContainerType");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "elementType",
            Slang::StringUtil::makeStringWithFormat("0x%llX", elementTypeId));
        _writePair(builder, indent, "containerType", ContainerTypeToString(containerType));
        _writePair(
            builder,
            indent,
            "outDiagnosticsId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnosticsId));
        _writePairNoComma(
            builder,
            indent,
            "outTypeReflectionId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outTypeReflectionId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_getDynamicType(ObjectID objectId, ObjectID outTypeReflectionId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::getDynamicType");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePairNoComma(
            builder,
            indent,
            "outTypeReflectionId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outTypeReflectionId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_getTypeRTTIMangledName(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID outNameBlobId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::getTypeRTTIMangledName");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "type",
            Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));
        _writePairNoComma(
            builder,
            indent,
            "outNameBlobId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outNameBlobId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_getTypeConformanceWitnessMangledName(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID interfaceTypeId,
    ObjectID outNameBlobId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "ISession::getTypeConformanceWitnessMangledName");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "type",
            Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));
        _writePair(
            builder,
            indent,
            "interfaceType",
            Slang::StringUtil::makeStringWithFormat("0x%llX", interfaceTypeId));
        _writePairNoComma(
            builder,
            indent,
            "outNameBlobId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outNameBlobId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_getTypeConformanceWitnessSequentialID(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID interfaceTypeId,
    uint32_t outId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "ISession::getTypeConformanceWitnessSequentialID");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePairNoComma(
            builder,
            indent,
            "type",
            Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::ISession_createTypeConformanceComponentType(
    ObjectID objectId,
    ObjectID typeId,
    ObjectID interfaceTypeId,
    ObjectID outConformanceId,
    SlangInt conformanceIdOverride,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(
            &builder,
            &indent,
            "ISession::createTypeConformanceComponentType");
        _writePair(
            builder,
            indent,
            "this",
            Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
        _writePair(
            builder,
            indent,
            "type",
            Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));
        _writePair(
            builder,
            indent,
            "interfaceTypeId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", interfaceTypeId));
        _writePair(
            builder,
            indent,
            "outConformanceId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", outConformanceId));
        _writePair(builder, indent, "conformanceIdOverride", conformanceIdOverride);
        _writePairNoComma(
            builder,
            indent,
            "outDiagnosticsId",
            Slang::StringUtil::makeStringWithFormat("0x%llX", typeId));
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_createCompileRequest(ObjectID objectId, ObjectID outCompileRequestId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::createCompileRequest");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "outCompileRequest", outCompileRequestId);
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::ISession_getLoadedModule(ObjectID objectId, SlangInt index, ObjectID outModuleId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "ISession::getLoadedModule");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "index", index);
            _writePairNoComma(
                builder,
                indent,
                "retModule",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outModuleId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


// IModule
void JsonConsumer::IModule_findEntryPointByName(
    ObjectID objectId,
    char const* name,
    ObjectID outEntryPointId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IModule::findEntryPointByName");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "name",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    name != nullptr ? name : "nullptr"));

            _writePairNoComma(
                builder,
                indent,
                "outEntryPoint",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outEntryPointId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IModule_getDefinedEntryPoint(
    ObjectID objectId,
    SlangInt32 index,
    ObjectID outEntryPointId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IModule::getDefinedEntryPoint");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(builder, indent, "index", index);
            _writePairNoComma(
                builder,
                indent,
                "outEntryPoint",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outEntryPointId));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IModule_serialize(ObjectID objectId, ObjectID outSerializedBlobId)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IModule::serialize");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(builder, indent, "outSerializedBlob", outSerializedBlobId);
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IModule_writeToFile(ObjectID objectId, char const* fileName)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IModule::writeToFile");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePairNoComma(
                builder,
                indent,
                "fileName",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    fileName != nullptr ? fileName : "nullptr"));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}


void JsonConsumer::IModule_findAndCheckEntryPoint(
    ObjectID objectId,
    char const* name,
    SlangStage stage,
    ObjectID outEntryPointId,
    ObjectID outDiagnostics)
{
    SANITY_CHECK();
    Slang::StringBuilder builder;
    int indent = 0;

    {
        ScopeWritterForKey scopeWritter(&builder, &indent, "IModule::findAndCheckEntryPoint");
        {
            _writePair(
                builder,
                indent,
                "this",
                Slang::StringUtil::makeStringWithFormat("0x%llX", objectId));
            _writePair(
                builder,
                indent,
                "name",
                Slang::StringUtil::makeStringWithFormat(
                    "\"%s\"",
                    name != nullptr ? name : "nullptr"));
            _writePair(builder, indent, "stage", SlangStageToString(stage));
            _writePair(
                builder,
                indent,
                "outEntryPoint",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outEntryPointId));
            _writePairNoComma(
                builder,
                indent,
                "outDiagnostics",
                Slang::StringUtil::makeStringWithFormat("0x%llX", outDiagnostics));
        }
    }

    m_fileStream.write(builder.produceString().begin(), builder.produceString().getLength());
    m_fileStream.flush();
}

void JsonConsumer::IModule_getSession(ObjectID objectId, ObjectID outSessionId)
{
    SANITY_CHECK();
    m_moduleHelper.getSession(objectId, outSessionId);
}

void JsonConsumer::IModule_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SANITY_CHECK();
    m_moduleHelper.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
}


void JsonConsumer::IModule_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_moduleHelper
        .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::IModule_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_moduleHelper.getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::IModule_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystemId)
{
    SANITY_CHECK();
    m_moduleHelper.getResultAsFileSystem(objectId, entryPointIndex, targetIndex, outFileSystemId);
}


void JsonConsumer::IModule_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SANITY_CHECK();
    m_moduleHelper.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
}


void JsonConsumer::IModule_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_moduleHelper.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
}


void JsonConsumer::IModule_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_moduleHelper.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
}


void JsonConsumer::IModule_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibraryId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_moduleHelper.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibraryId,
        outDiagnosticsId);
}


void JsonConsumer::IModule_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SANITY_CHECK();
    m_moduleHelper.renameEntryPoint(objectId, newName, outEntryPointId);
}


void JsonConsumer::IModule_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_moduleHelper.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
}

// IEntryPoint
void JsonConsumer::IEntryPoint_getSession(ObjectID objectId, ObjectID outSessionId)
{
    SANITY_CHECK();
    m_entryPointHelper.getSession(objectId, outSessionId);
}


void JsonConsumer::IEntryPoint_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SANITY_CHECK();
    m_entryPointHelper.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
}


void JsonConsumer::IEntryPoint_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_entryPointHelper
        .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::IEntryPoint_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_entryPointHelper.getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::IEntryPoint_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystem)
{
    SANITY_CHECK();
    m_entryPointHelper.getResultAsFileSystem(objectId, entryPointIndex, targetIndex, outFileSystem);
}


void JsonConsumer::IEntryPoint_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SANITY_CHECK();
    m_entryPointHelper.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
}


void JsonConsumer::IEntryPoint_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_entryPointHelper.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
}


void JsonConsumer::IEntryPoint_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_entryPointHelper.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
}


void JsonConsumer::IEntryPoint_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SANITY_CHECK();
    m_entryPointHelper.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
}


void JsonConsumer::IEntryPoint_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SANITY_CHECK();
    m_entryPointHelper.renameEntryPoint(objectId, newName, outEntryPointId);
}


void JsonConsumer::IEntryPoint_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_entryPointHelper.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
}


// ICompositeComponentType
void JsonConsumer::ICompositeComponentType_getSession(ObjectID objectId, ObjectID outSessionId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper.getSession(objectId, outSessionId);
}


void JsonConsumer::ICompositeComponentType_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper
        .getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
}


void JsonConsumer::ICompositeComponentType_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper
        .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::ICompositeComponentType_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper
        .getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::ICompositeComponentType_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystem)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper
        .getResultAsFileSystem(objectId, entryPointIndex, targetIndex, outFileSystem);
}


void JsonConsumer::ICompositeComponentType_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper
        .getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
}


void JsonConsumer::ICompositeComponentType_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
}


void JsonConsumer::ICompositeComponentType_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
}


void JsonConsumer::ICompositeComponentType_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
}


void JsonConsumer::ICompositeComponentType_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper.renameEntryPoint(objectId, newName, outEntryPointId);
}


void JsonConsumer::ICompositeComponentType_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_compositeComponentTypeHelper.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
}


// ITypeConformance
void JsonConsumer::ITypeConformance_getSession(ObjectID objectId, ObjectID outSessionId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.getSession(objectId, outSessionId);
}


void JsonConsumer::ITypeConformance_getLayout(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outDiagnosticsId,
    ObjectID retProgramLayoutId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.getLayout(objectId, targetIndex, outDiagnosticsId, retProgramLayoutId);
}


void JsonConsumer::ITypeConformance_getEntryPointCode(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_typeConformanceHelper
        .getEntryPointCode(objectId, entryPointIndex, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::ITypeConformance_getTargetCode(
    ObjectID objectId,
    SlangInt targetIndex,
    ObjectID outCodeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.getTargetCode(objectId, targetIndex, outCodeId, outDiagnosticsId);
}


void JsonConsumer::ITypeConformance_getResultAsFileSystem(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outFileSystem)
{
    SANITY_CHECK();
    m_typeConformanceHelper
        .getResultAsFileSystem(objectId, entryPointIndex, targetIndex, outFileSystem);
}


void JsonConsumer::ITypeConformance_getEntryPointHash(
    ObjectID objectId,
    SlangInt entryPointIndex,
    SlangInt targetIndex,
    ObjectID outHashId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.getEntryPointHash(objectId, entryPointIndex, targetIndex, outHashId);
}


void JsonConsumer::ITypeConformance_specialize(
    ObjectID objectId,
    slang::SpecializationArg const* specializationArgs,
    SlangInt specializationArgCount,
    ObjectID outSpecializedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.specialize(
        objectId,
        specializationArgs,
        specializationArgCount,
        outSpecializedComponentTypeId,
        outDiagnosticsId);
}


void JsonConsumer::ITypeConformance_link(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.link(objectId, outLinkedComponentTypeId, outDiagnosticsId);
}


void JsonConsumer::ITypeConformance_getEntryPointHostCallable(
    ObjectID objectId,
    int entryPointIndex,
    int targetIndex,
    ObjectID outSharedLibrary,
    ObjectID outDiagnostics)
{
    SANITY_CHECK();
    m_typeConformanceHelper.getEntryPointHostCallable(
        objectId,
        entryPointIndex,
        targetIndex,
        outSharedLibrary,
        outDiagnostics);
}


void JsonConsumer::ITypeConformance_renameEntryPoint(
    ObjectID objectId,
    const char* newName,
    ObjectID outEntryPointId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.renameEntryPoint(objectId, newName, outEntryPointId);
}


void JsonConsumer::ITypeConformance_linkWithOptions(
    ObjectID objectId,
    ObjectID outLinkedComponentTypeId,
    uint32_t compilerOptionEntryCount,
    slang::CompilerOptionEntry* compilerOptionEntries,
    ObjectID outDiagnosticsId)
{
    SANITY_CHECK();
    m_typeConformanceHelper.linkWithOptions(
        objectId,
        outLinkedComponentTypeId,
        compilerOptionEntryCount,
        compilerOptionEntries,
        outDiagnosticsId);
}
}; // namespace SlangRecord
