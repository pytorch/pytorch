#pragma once

#include "slang.h"

/* DEPRECATED DEFINITIONS

Everything in this file represents deprecated APIs/definition that are only
being kept around for source/binary compatibility with old client code. New
code should not use any of these declarations, and the Slang API will drop these
declarations over time.
*/

#ifdef __cplusplus
extern "C"
{
#endif

    /*!
    @brief Initialize an instance of the Slang library.
    */
    SLANG_API SlangSession* spCreateSession(const char* deprecated = 0);

    /*!
    @brief Clean up after an instance of the Slang library.
    */
    SLANG_API void spDestroySession(SlangSession* session);

    /** @see slang::IGlobalSession::setSharedLibraryLoader
     */
    SLANG_API void spSessionSetSharedLibraryLoader(
        SlangSession* session,
        ISlangSharedLibraryLoader* loader);

    /** @see slang::IGlobalSession::getSharedLibraryLoader
     */
    SLANG_API ISlangSharedLibraryLoader* spSessionGetSharedLibraryLoader(SlangSession* session);

    /** @see slang::IGlobalSession::checkCompileTargetSupport
     */
    SLANG_API SlangResult
    spSessionCheckCompileTargetSupport(SlangSession* session, SlangCompileTarget target);

    /** @see slang::IGlobalSession::checkPassThroughSupport
     */
    SLANG_API SlangResult
    spSessionCheckPassThroughSupport(SlangSession* session, SlangPassThrough passThrough);

    /** @see slang::IGlobalSession::addBuiltins
     */
    SLANG_API void spAddBuiltins(
        SlangSession* session,
        char const* sourcePath,
        char const* sourceString);

    /* @see slang::IGlobalSession::createCompileRequest
     */
    SLANG_API SlangCompileRequest* spCreateCompileRequest(SlangSession* session);

    /*!
    @brief Destroy a compile request.
    Note a request is a COM object and can be destroyed via 'Release'.
    */
    SLANG_API void spDestroyCompileRequest(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::setFileSystem */
    SLANG_API void spSetFileSystem(SlangCompileRequest* request, ISlangFileSystem* fileSystem);

    /*! @see slang::ICompileRequest::setCompileFlags */
    SLANG_API void spSetCompileFlags(SlangCompileRequest* request, SlangCompileFlags flags);

    /*! @see slang::ICompileRequest::getCompileFlags */
    SLANG_API SlangCompileFlags spGetCompileFlags(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::setDumpIntermediates */
    SLANG_API void spSetDumpIntermediates(SlangCompileRequest* request, int enable);

    /*! @see slang::ICompileRequest::setDumpIntermediatePrefix */
    SLANG_API void spSetDumpIntermediatePrefix(SlangCompileRequest* request, const char* prefix);

    /*! DEPRECATED: use `spSetTargetLineDirectiveMode` instead.
        @see slang::ICompileRequest::setLineDirectiveMode */
    SLANG_API void spSetLineDirectiveMode(
        SlangCompileRequest* request,
        SlangLineDirectiveMode mode);

    /*! @see slang::ICompileRequest::setTargetLineDirectiveMode */
    SLANG_API void spSetTargetLineDirectiveMode(
        SlangCompileRequest* request,
        int targetIndex,
        SlangLineDirectiveMode mode);

    /*! @see slang::ICompileRequest::setTargetLineDirectiveMode */
    SLANG_API void spSetTargetForceGLSLScalarBufferLayout(
        SlangCompileRequest* request,
        int targetIndex,
        bool forceScalarLayout);

    /*! @see slang::ICompileRequest::setTargetUseMinimumSlangOptimization */
    SLANG_API void spSetTargetUseMinimumSlangOptimization(
        slang::ICompileRequest* request,
        int targetIndex,
        bool val);

    /*! @see slang::ICompileRequest::setIgnoreCapabilityCheck */
    SLANG_API void spSetIgnoreCapabilityCheck(slang::ICompileRequest* request, bool val);

    /*! @see slang::ICompileRequest::setCodeGenTarget */
    SLANG_API void spSetCodeGenTarget(SlangCompileRequest* request, SlangCompileTarget target);

    /*! @see slang::ICompileRequest::addCodeGenTarget */
    SLANG_API int spAddCodeGenTarget(SlangCompileRequest* request, SlangCompileTarget target);

    /*! @see slang::ICompileRequest::setTargetProfile */
    SLANG_API void spSetTargetProfile(
        SlangCompileRequest* request,
        int targetIndex,
        SlangProfileID profile);

    /*! @see slang::ICompileRequest::setTargetFlags */
    SLANG_API void spSetTargetFlags(
        SlangCompileRequest* request,
        int targetIndex,
        SlangTargetFlags flags);


    /*! @see slang::ICompileRequest::setTargetFloatingPointMode */
    SLANG_API void spSetTargetFloatingPointMode(
        SlangCompileRequest* request,
        int targetIndex,
        SlangFloatingPointMode mode);

    /*! @see slang::ICompileRequest::addTargetCapability */
    SLANG_API void spAddTargetCapability(
        slang::ICompileRequest* request,
        int targetIndex,
        SlangCapabilityID capability);

    /* DEPRECATED: use `spSetMatrixLayoutMode` instead. */
    SLANG_API void spSetTargetMatrixLayoutMode(
        SlangCompileRequest* request,
        int targetIndex,
        SlangMatrixLayoutMode mode);

    /*! @see slang::ICompileRequest::setMatrixLayoutMode */
    SLANG_API void spSetMatrixLayoutMode(SlangCompileRequest* request, SlangMatrixLayoutMode mode);

    /*! @see slang::ICompileRequest::setDebugInfoLevel */
    SLANG_API void spSetDebugInfoLevel(SlangCompileRequest* request, SlangDebugInfoLevel level);

    /*! @see slang::ICompileRequest::setDebugInfoFormat */
    SLANG_API void spSetDebugInfoFormat(SlangCompileRequest* request, SlangDebugInfoFormat format);

    /*! @see slang::ICompileRequest::setOptimizationLevel */
    SLANG_API void spSetOptimizationLevel(
        SlangCompileRequest* request,
        SlangOptimizationLevel level);


    /*! @see slang::ICompileRequest::setOutputContainerFormat */
    SLANG_API void spSetOutputContainerFormat(
        SlangCompileRequest* request,
        SlangContainerFormat format);

    /*! @see slang::ICompileRequest::setPassThrough */
    SLANG_API void spSetPassThrough(SlangCompileRequest* request, SlangPassThrough passThrough);

    /*! @see slang::ICompileRequest::setDiagnosticCallback */
    SLANG_API void spSetDiagnosticCallback(
        SlangCompileRequest* request,
        SlangDiagnosticCallback callback,
        void const* userData);

    /*! @see slang::ICompileRequest::setWriter */
    SLANG_API void spSetWriter(
        SlangCompileRequest* request,
        SlangWriterChannel channel,
        ISlangWriter* writer);

    /*! @see slang::ICompileRequest::getWriter */
    SLANG_API ISlangWriter* spGetWriter(SlangCompileRequest* request, SlangWriterChannel channel);

    /*! @see slang::ICompileRequest::addSearchPath */
    SLANG_API void spAddSearchPath(SlangCompileRequest* request, const char* searchDir);

    /*! @see slang::ICompileRequest::addPreprocessorDefine */
    SLANG_API void spAddPreprocessorDefine(
        SlangCompileRequest* request,
        const char* key,
        const char* value);

    /*! @see slang::ICompileRequest::processCommandLineArguments */
    SLANG_API SlangResult spProcessCommandLineArguments(
        SlangCompileRequest* request,
        char const* const* args,
        int argCount);

    /*! @see slang::ICompileRequest::addTranslationUnit */
    SLANG_API int spAddTranslationUnit(
        SlangCompileRequest* request,
        SlangSourceLanguage language,
        char const* name);


    /*! @see slang::ICompileRequest::setDefaultModuleName */
    SLANG_API void spSetDefaultModuleName(
        SlangCompileRequest* request,
        const char* defaultModuleName);

    /*! @see slang::ICompileRequest::addPreprocessorDefine */
    SLANG_API void spTranslationUnit_addPreprocessorDefine(
        SlangCompileRequest* request,
        int translationUnitIndex,
        const char* key,
        const char* value);


    /*! @see slang::ICompileRequest::addTranslationUnitSourceFile */
    SLANG_API void spAddTranslationUnitSourceFile(
        SlangCompileRequest* request,
        int translationUnitIndex,
        char const* path);

    /*! @see slang::ICompileRequest::addTranslationUnitSourceString */
    SLANG_API void spAddTranslationUnitSourceString(
        SlangCompileRequest* request,
        int translationUnitIndex,
        char const* path,
        char const* source);


    /*! @see slang::ICompileRequest::addLibraryReference */
    SLANG_API SlangResult spAddLibraryReference(
        SlangCompileRequest* request,
        const char* basePath,
        const void* libData,
        size_t libDataSize);

    /*! @see slang::ICompileRequest::addTranslationUnitSourceStringSpan */
    SLANG_API void spAddTranslationUnitSourceStringSpan(
        SlangCompileRequest* request,
        int translationUnitIndex,
        char const* path,
        char const* sourceBegin,
        char const* sourceEnd);

    /*! @see slang::ICompileRequest::addTranslationUnitSourceBlob */
    SLANG_API void spAddTranslationUnitSourceBlob(
        SlangCompileRequest* request,
        int translationUnitIndex,
        char const* path,
        ISlangBlob* sourceBlob);

    /*! @see slang::IGlobalSession::findProfile */
    SLANG_API SlangProfileID spFindProfile(SlangSession* session, char const* name);

    /*! @see slang::IGlobalSession::findCapability */
    SLANG_API SlangCapabilityID spFindCapability(SlangSession* session, char const* name);

    /*! @see slang::ICompileRequest::addEntryPoint */
    SLANG_API int spAddEntryPoint(
        SlangCompileRequest* request,
        int translationUnitIndex,
        char const* name,
        SlangStage stage);

    /*! @see slang::ICompileRequest::addEntryPointEx */
    SLANG_API int spAddEntryPointEx(
        SlangCompileRequest* request,
        int translationUnitIndex,
        char const* name,
        SlangStage stage,
        int genericArgCount,
        char const** genericArgs);

    /*! @see slang::ICompileRequest::setGlobalGenericArgs */
    SLANG_API SlangResult spSetGlobalGenericArgs(
        SlangCompileRequest* request,
        int genericArgCount,
        char const** genericArgs);

    /*! @see slang::ICompileRequest::setTypeNameForGlobalExistentialTypeParam */
    SLANG_API SlangResult spSetTypeNameForGlobalExistentialTypeParam(
        SlangCompileRequest* request,
        int slotIndex,
        char const* typeName);

    /*! @see slang::ICompileRequest::setTypeNameForEntryPointExistentialTypeParam */
    SLANG_API SlangResult spSetTypeNameForEntryPointExistentialTypeParam(
        SlangCompileRequest* request,
        int entryPointIndex,
        int slotIndex,
        char const* typeName);

    /*! @see slang::ICompileRequest::compile */
    SLANG_API SlangResult spCompile(SlangCompileRequest* request);


    /*! @see slang::ICompileRequest::getDiagnosticOutput */
    SLANG_API char const* spGetDiagnosticOutput(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::getDiagnosticOutputBlob */
    SLANG_API SlangResult
    spGetDiagnosticOutputBlob(SlangCompileRequest* request, ISlangBlob** outBlob);


    /*! @see slang::ICompileRequest::getDependencyFileCount */
    SLANG_API int spGetDependencyFileCount(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::getDependencyFilePath */
    SLANG_API char const* spGetDependencyFilePath(SlangCompileRequest* request, int index);

    /*! @see slang::ICompileRequest::getTranslationUnitCount */
    SLANG_API int spGetTranslationUnitCount(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::getEntryPointSource */
    SLANG_API char const* spGetEntryPointSource(SlangCompileRequest* request, int entryPointIndex);

    /*! @see slang::ICompileRequest::getEntryPointCode */
    SLANG_API void const* spGetEntryPointCode(
        SlangCompileRequest* request,
        int entryPointIndex,
        size_t* outSize);

    /*! @see slang::ICompileRequest::getEntryPointCodeBlob */
    SLANG_API SlangResult spGetEntryPointCodeBlob(
        SlangCompileRequest* request,
        int entryPointIndex,
        int targetIndex,
        ISlangBlob** outBlob);

    /*! @see slang::ICompileRequest::getEntryPointHostCallable */
    SLANG_API SlangResult spGetEntryPointHostCallable(
        SlangCompileRequest* request,
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary);

    /*! @see slang::ICompileRequest::getTargetCodeBlob */
    SLANG_API SlangResult
    spGetTargetCodeBlob(SlangCompileRequest* request, int targetIndex, ISlangBlob** outBlob);

    /*! @see slang::ICompileRequest::getTargetHostCallable */
    SLANG_API SlangResult spGetTargetHostCallable(
        SlangCompileRequest* request,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary);

    /*! @see slang::ICompileRequest::getCompileRequestCode */
    SLANG_API void const* spGetCompileRequestCode(SlangCompileRequest* request, size_t* outSize);

    /*! @see slang::ICompileRequest::getContainerCode */
    SLANG_API SlangResult spGetContainerCode(SlangCompileRequest* request, ISlangBlob** outBlob);

    /*! @see slang::ICompileRequest::loadRepro */
    SLANG_API SlangResult spLoadRepro(
        SlangCompileRequest* request,
        ISlangFileSystem* fileSystem,
        const void* data,
        size_t size);

    /*! @see slang::ICompileRequest::saveRepro */
    SLANG_API SlangResult spSaveRepro(SlangCompileRequest* request, ISlangBlob** outBlob);

    /*! @see slang::ICompileRequest::enableReproCapture */
    SLANG_API SlangResult spEnableReproCapture(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::getCompileTimeProfile */
    SLANG_API SlangResult spGetCompileTimeProfile(
        SlangCompileRequest* request,
        ISlangProfiler** compileTimeProfile,
        bool shouldClear);


    /** Extract contents of a repro.

    Writes the contained files and manifest with their 'unique' names into fileSystem. For more
    details read the docs/repro.md documentation.

    @param session          The slang session
    @param reproData        Holds the repro data
    @param reproDataSize    The size of the repro data
    @param fileSystem       File system that the contents of the repro will be written to
    @returns                A `SlangResult` to indicate success or failure.
    */
    SLANG_API SlangResult spExtractRepro(
        SlangSession* session,
        const void* reproData,
        size_t reproDataSize,
        ISlangMutableFileSystem* fileSystem);

    /* Turns a repro into a file system.

    Makes the contents of the repro available as a file system - that is able to access the files
    with the same paths as were used on the original repro file system.

    @param session          The slang session
    @param reproData        The repro data
    @param reproDataSize    The size of the repro data
    @param replaceFileSystem  Will attempt to load by unique names from this file system before
    using contents of the repro. Optional.
    @param outFileSystem    The file system that can be used to access contents
    @returns                A `SlangResult` to indicate success or failure.
    */
    SLANG_API SlangResult spLoadReproAsFileSystem(
        SlangSession* session,
        const void* reproData,
        size_t reproDataSize,
        ISlangFileSystem* replaceFileSystem,
        ISlangFileSystemExt** outFileSystem);

    /*! @see slang::ICompileRequest::overrideDiagnosticSeverity */
    SLANG_API void spOverrideDiagnosticSeverity(
        SlangCompileRequest* request,
        SlangInt messageID,
        SlangSeverity overrideSeverity);

    /*! @see slang::ICompileRequest::getDiagnosticFlags */
    SLANG_API SlangDiagnosticFlags spGetDiagnosticFlags(SlangCompileRequest* request);

    /*! @see slang::ICompileRequest::setDiagnosticFlags */
    SLANG_API void spSetDiagnosticFlags(SlangCompileRequest* request, SlangDiagnosticFlags flags);


    // get reflection data from a compilation request
    SLANG_API SlangReflection* spGetReflection(SlangCompileRequest* request);

    // User Attribute
    SLANG_API char const* spReflectionUserAttribute_GetName(SlangReflectionUserAttribute* attrib);
    SLANG_API unsigned int spReflectionUserAttribute_GetArgumentCount(
        SlangReflectionUserAttribute* attrib);
    SLANG_API SlangReflectionType* spReflectionUserAttribute_GetArgumentType(
        SlangReflectionUserAttribute* attrib,
        unsigned int index);
    SLANG_API SlangResult spReflectionUserAttribute_GetArgumentValueInt(
        SlangReflectionUserAttribute* attrib,
        unsigned int index,
        int* rs);
    SLANG_API SlangResult spReflectionUserAttribute_GetArgumentValueFloat(
        SlangReflectionUserAttribute* attrib,
        unsigned int index,
        float* rs);

    /** Returns the string-typed value of a user attribute argument
        The string returned is not null-terminated. The length of the string is returned via
       `outSize`. If index of out of range, or if the specified argument is not a string, the
       function will return nullptr.
    */
    SLANG_API const char* spReflectionUserAttribute_GetArgumentValueString(
        SlangReflectionUserAttribute* attrib,
        unsigned int index,
        size_t* outSize);

    // Type Reflection

    SLANG_API SlangTypeKind spReflectionType_GetKind(SlangReflectionType* type);
    SLANG_API unsigned int spReflectionType_GetUserAttributeCount(SlangReflectionType* type);
    SLANG_API SlangReflectionUserAttribute* spReflectionType_GetUserAttribute(
        SlangReflectionType* type,
        unsigned int index);
    SLANG_API SlangReflectionUserAttribute* spReflectionType_FindUserAttributeByName(
        SlangReflectionType* type,
        char const* name);
    SLANG_API SlangReflectionType* spReflectionType_applySpecializations(
        SlangReflectionType* type,
        SlangReflectionGeneric* generic);

    SLANG_API unsigned int spReflectionType_GetFieldCount(SlangReflectionType* type);
    SLANG_API SlangReflectionVariable* spReflectionType_GetFieldByIndex(
        SlangReflectionType* type,
        unsigned index);

    /** Returns the number of elements in the given type.

    This operation is valid for vector and array types. For other types it returns zero.

    When invoked on an unbounded-size array it will return `SLANG_UNBOUNDED_SIZE`,
    which is defined to be `~size_t(0)`.

    If the size of a type cannot be statically computed, perhaps because it depends on
    a generic parameter that has not been bound to a specific value, this function returns zero.
    */
    SLANG_API size_t spReflectionType_GetElementCount(SlangReflectionType* type);

    SLANG_API SlangReflectionType* spReflectionType_GetElementType(SlangReflectionType* type);

    SLANG_API unsigned int spReflectionType_GetRowCount(SlangReflectionType* type);
    SLANG_API unsigned int spReflectionType_GetColumnCount(SlangReflectionType* type);
    SLANG_API SlangScalarType spReflectionType_GetScalarType(SlangReflectionType* type);

    SLANG_API SlangResourceShape spReflectionType_GetResourceShape(SlangReflectionType* type);
    SLANG_API SlangResourceAccess spReflectionType_GetResourceAccess(SlangReflectionType* type);
    SLANG_API SlangReflectionType* spReflectionType_GetResourceResultType(
        SlangReflectionType* type);

    SLANG_API char const* spReflectionType_GetName(SlangReflectionType* type);
    SLANG_API SlangResult
    spReflectionType_GetFullName(SlangReflectionType* type, ISlangBlob** outNameBlob);
    SLANG_API SlangReflectionGeneric* spReflectionType_GetGenericContainer(
        SlangReflectionType* type);

    // Type Layout Reflection

    SLANG_API SlangReflectionType* spReflectionTypeLayout_GetType(SlangReflectionTypeLayout* type);
    SLANG_API SlangTypeKind spReflectionTypeLayout_getKind(SlangReflectionTypeLayout* type);
    SLANG_API size_t spReflectionTypeLayout_GetSize(
        SlangReflectionTypeLayout* type,
        SlangParameterCategory category);
    SLANG_API size_t spReflectionTypeLayout_GetStride(
        SlangReflectionTypeLayout* type,
        SlangParameterCategory category);
    SLANG_API int32_t spReflectionTypeLayout_getAlignment(
        SlangReflectionTypeLayout* type,
        SlangParameterCategory category);

    SLANG_API uint32_t spReflectionTypeLayout_GetFieldCount(SlangReflectionTypeLayout* type);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetFieldByIndex(
        SlangReflectionTypeLayout* type,
        unsigned index);

    SLANG_API SlangInt spReflectionTypeLayout_findFieldIndexByName(
        SlangReflectionTypeLayout* typeLayout,
        const char* nameBegin,
        const char* nameEnd);

    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetExplicitCounter(
        SlangReflectionTypeLayout* typeLayout);

    SLANG_API size_t spReflectionTypeLayout_GetElementStride(
        SlangReflectionTypeLayout* type,
        SlangParameterCategory category);
    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_GetElementTypeLayout(
        SlangReflectionTypeLayout* type);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_GetElementVarLayout(
        SlangReflectionTypeLayout* type);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getContainerVarLayout(
        SlangReflectionTypeLayout* type);

    SLANG_API SlangParameterCategory
    spReflectionTypeLayout_GetParameterCategory(SlangReflectionTypeLayout* type);

    SLANG_API unsigned spReflectionTypeLayout_GetCategoryCount(SlangReflectionTypeLayout* type);
    SLANG_API SlangParameterCategory
    spReflectionTypeLayout_GetCategoryByIndex(SlangReflectionTypeLayout* type, unsigned index);

    SLANG_API SlangMatrixLayoutMode
    spReflectionTypeLayout_GetMatrixLayoutMode(SlangReflectionTypeLayout* type);

    SLANG_API int spReflectionTypeLayout_getGenericParamIndex(SlangReflectionTypeLayout* type);

    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getPendingDataTypeLayout(
        SlangReflectionTypeLayout* type);

    SLANG_API SlangReflectionVariableLayout*
    spReflectionTypeLayout_getSpecializedTypePendingDataVarLayout(SlangReflectionTypeLayout* type);
    SLANG_API SlangInt spReflectionType_getSpecializedTypeArgCount(SlangReflectionType* type);
    SLANG_API SlangReflectionType* spReflectionType_getSpecializedTypeArgType(
        SlangReflectionType* type,
        SlangInt index);

    SLANG_API SlangInt
    spReflectionTypeLayout_getBindingRangeCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangBindingType spReflectionTypeLayout_getBindingRangeType(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_isBindingRangeSpecializable(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeBindingCount(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getBindingRangeLeafTypeLayout(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangReflectionVariable* spReflectionTypeLayout_getBindingRangeLeafVariable(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangImageFormat spReflectionTypeLayout_getBindingRangeImageFormat(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getFieldBindingRangeOffset(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt fieldIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getExplicitCounterBindingRangeOffset(
        SlangReflectionTypeLayout* inTypeLayout);

    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeDescriptorSetIndex(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeFirstDescriptorRangeIndex(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getBindingRangeDescriptorRangeCount(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt index);

    SLANG_API SlangInt
    spReflectionTypeLayout_getDescriptorSetCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetSpaceOffset(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt setIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeCount(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt setIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeIndexOffset(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt setIndex,
        SlangInt rangeIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getDescriptorSetDescriptorRangeDescriptorCount(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt setIndex,
        SlangInt rangeIndex);
    SLANG_API SlangBindingType spReflectionTypeLayout_getDescriptorSetDescriptorRangeType(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt setIndex,
        SlangInt rangeIndex);
    SLANG_API SlangParameterCategory spReflectionTypeLayout_getDescriptorSetDescriptorRangeCategory(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt setIndex,
        SlangInt rangeIndex);

    SLANG_API SlangInt
    spReflectionTypeLayout_getSubObjectRangeCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt subObjectRangeIndex);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeSpaceOffset(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt subObjectRangeIndex);
    SLANG_API SlangReflectionVariableLayout* spReflectionTypeLayout_getSubObjectRangeOffset(
        SlangReflectionTypeLayout* typeLayout,
        SlangInt subObjectRangeIndex);

#if 0
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeCount(SlangReflectionTypeLayout* typeLayout);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeObjectCount(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeBindingRangeIndex(SlangReflectionTypeLayout* typeLayout, SlangInt index);
    SLANG_API SlangReflectionTypeLayout* spReflectionTypeLayout_getSubObjectRangeTypeLayout(SlangReflectionTypeLayout* typeLayout, SlangInt index);

    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeCount(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex);
    SLANG_API SlangBindingType spReflectionTypeLayout_getSubObjectRangeDescriptorRangeBindingType(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeBindingCount(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeIndexOffset(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
    SLANG_API SlangInt spReflectionTypeLayout_getSubObjectRangeDescriptorRangeSpaceOffset(SlangReflectionTypeLayout* typeLayout, SlangInt subObjectRangeIndex, SlangInt bindingRangeIndexInSubObject);
#endif

    // Variable Reflection

    SLANG_API char const* spReflectionVariable_GetName(SlangReflectionVariable* var);
    SLANG_API SlangReflectionType* spReflectionVariable_GetType(SlangReflectionVariable* var);
    SLANG_API SlangReflectionModifier* spReflectionVariable_FindModifier(
        SlangReflectionVariable* var,
        SlangModifierID modifierID);
    SLANG_API unsigned int spReflectionVariable_GetUserAttributeCount(SlangReflectionVariable* var);
    SLANG_API SlangReflectionUserAttribute* spReflectionVariable_GetUserAttribute(
        SlangReflectionVariable* var,
        unsigned int index);
    SLANG_API SlangReflectionUserAttribute* spReflectionVariable_FindUserAttributeByName(
        SlangReflectionVariable* var,
        SlangSession* globalSession,
        char const* name);
    SLANG_API bool spReflectionVariable_HasDefaultValue(SlangReflectionVariable* inVar);
    SLANG_API SlangResult
    spReflectionVariable_GetDefaultValueInt(SlangReflectionVariable* inVar, int64_t* rs);
    SLANG_API SlangReflectionGeneric* spReflectionVariable_GetGenericContainer(
        SlangReflectionVariable* var);
    SLANG_API SlangReflectionVariable* spReflectionVariable_applySpecializations(
        SlangReflectionVariable* var,
        SlangReflectionGeneric* generic);

    // Variable Layout Reflection

    SLANG_API SlangReflectionVariable* spReflectionVariableLayout_GetVariable(
        SlangReflectionVariableLayout* var);

    SLANG_API SlangReflectionTypeLayout* spReflectionVariableLayout_GetTypeLayout(
        SlangReflectionVariableLayout* var);

    SLANG_API size_t spReflectionVariableLayout_GetOffset(
        SlangReflectionVariableLayout* var,
        SlangParameterCategory category);
    SLANG_API size_t spReflectionVariableLayout_GetSpace(
        SlangReflectionVariableLayout* var,
        SlangParameterCategory category);
    SLANG_API SlangImageFormat
    spReflectionVariableLayout_GetImageFormat(SlangReflectionVariableLayout* var);

    SLANG_API char const* spReflectionVariableLayout_GetSemanticName(
        SlangReflectionVariableLayout* var);
    SLANG_API size_t
    spReflectionVariableLayout_GetSemanticIndex(SlangReflectionVariableLayout* var);


    // Function Reflection

    SLANG_API SlangReflectionDecl* spReflectionFunction_asDecl(SlangReflectionFunction* func);
    SLANG_API char const* spReflectionFunction_GetName(SlangReflectionFunction* func);
    SLANG_API SlangReflectionModifier* spReflectionFunction_FindModifier(
        SlangReflectionFunction* var,
        SlangModifierID modifierID);
    SLANG_API unsigned int spReflectionFunction_GetUserAttributeCount(
        SlangReflectionFunction* func);
    SLANG_API SlangReflectionUserAttribute* spReflectionFunction_GetUserAttribute(
        SlangReflectionFunction* func,
        unsigned int index);
    SLANG_API SlangReflectionUserAttribute* spReflectionFunction_FindUserAttributeByName(
        SlangReflectionFunction* func,
        SlangSession* globalSession,
        char const* name);
    SLANG_API unsigned int spReflectionFunction_GetParameterCount(SlangReflectionFunction* func);
    SLANG_API SlangReflectionVariable* spReflectionFunction_GetParameter(
        SlangReflectionFunction* func,
        unsigned index);
    SLANG_API SlangReflectionType* spReflectionFunction_GetResultType(
        SlangReflectionFunction* func);
    SLANG_API SlangReflectionGeneric* spReflectionFunction_GetGenericContainer(
        SlangReflectionFunction* func);
    SLANG_API SlangReflectionFunction* spReflectionFunction_applySpecializations(
        SlangReflectionFunction* func,
        SlangReflectionGeneric* generic);
    SLANG_API SlangReflectionFunction* spReflectionFunction_specializeWithArgTypes(
        SlangReflectionFunction* func,
        SlangInt argTypeCount,
        SlangReflectionType* const* argTypes);
    SLANG_API bool spReflectionFunction_isOverloaded(SlangReflectionFunction* func);
    SLANG_API unsigned int spReflectionFunction_getOverloadCount(SlangReflectionFunction* func);
    SLANG_API SlangReflectionFunction* spReflectionFunction_getOverload(
        SlangReflectionFunction* func,
        unsigned int index);

    // Abstract Decl Reflection

    SLANG_API unsigned int spReflectionDecl_getChildrenCount(SlangReflectionDecl* parentDecl);
    SLANG_API SlangReflectionDecl* spReflectionDecl_getChild(
        SlangReflectionDecl* parentDecl,
        unsigned int index);
    SLANG_API char const* spReflectionDecl_getName(SlangReflectionDecl* decl);
    SLANG_API SlangDeclKind spReflectionDecl_getKind(SlangReflectionDecl* decl);
    SLANG_API SlangReflectionFunction* spReflectionDecl_castToFunction(SlangReflectionDecl* decl);
    SLANG_API SlangReflectionVariable* spReflectionDecl_castToVariable(SlangReflectionDecl* decl);
    SLANG_API SlangReflectionGeneric* spReflectionDecl_castToGeneric(SlangReflectionDecl* decl);
    SLANG_API SlangReflectionType* spReflection_getTypeFromDecl(SlangReflectionDecl* decl);
    SLANG_API SlangReflectionDecl* spReflectionDecl_getParent(SlangReflectionDecl* decl);

    // Generic Reflection

    SLANG_API SlangReflectionDecl* spReflectionGeneric_asDecl(SlangReflectionGeneric* generic);
    SLANG_API char const* spReflectionGeneric_GetName(SlangReflectionGeneric* generic);
    SLANG_API unsigned int spReflectionGeneric_GetTypeParameterCount(
        SlangReflectionGeneric* generic);
    SLANG_API SlangReflectionVariable* spReflectionGeneric_GetTypeParameter(
        SlangReflectionGeneric* generic,
        unsigned index);
    SLANG_API unsigned int spReflectionGeneric_GetValueParameterCount(
        SlangReflectionGeneric* generic);
    SLANG_API SlangReflectionVariable* spReflectionGeneric_GetValueParameter(
        SlangReflectionGeneric* generic,
        unsigned index);
    SLANG_API unsigned int spReflectionGeneric_GetTypeParameterConstraintCount(
        SlangReflectionGeneric* generic,
        SlangReflectionVariable* typeParam);
    SLANG_API SlangReflectionType* spReflectionGeneric_GetTypeParameterConstraintType(
        SlangReflectionGeneric* generic,
        SlangReflectionVariable* typeParam,
        unsigned index);
    SLANG_API SlangDeclKind spReflectionGeneric_GetInnerKind(SlangReflectionGeneric* generic);
    SLANG_API SlangReflectionDecl* spReflectionGeneric_GetInnerDecl(
        SlangReflectionGeneric* generic);
    SLANG_API SlangReflectionGeneric* spReflectionGeneric_GetOuterGenericContainer(
        SlangReflectionGeneric* generic);
    SLANG_API SlangReflectionType* spReflectionGeneric_GetConcreteType(
        SlangReflectionGeneric* generic,
        SlangReflectionVariable* typeParam);
    SLANG_API int64_t spReflectionGeneric_GetConcreteIntVal(
        SlangReflectionGeneric* generic,
        SlangReflectionVariable* valueParam);
    SLANG_API SlangReflectionGeneric* spReflectionGeneric_applySpecializations(
        SlangReflectionGeneric* currGeneric,
        SlangReflectionGeneric* generic);


    /** Get the stage that a variable belongs to (if any).

    A variable "belongs" to a specific stage when it is a varying input/output
    parameter either defined as part of the parameter list for an entry
    point *or* at the global scope of a stage-specific GLSL code file (e.g.,
    an `in` parameter in a GLSL `.vs` file belongs to the vertex stage).
    */
    SLANG_API SlangStage spReflectionVariableLayout_getStage(SlangReflectionVariableLayout* var);


    SLANG_API SlangReflectionVariableLayout* spReflectionVariableLayout_getPendingDataLayout(
        SlangReflectionVariableLayout* var);

    // Shader Parameter Reflection

    SLANG_API unsigned spReflectionParameter_GetBindingIndex(SlangReflectionParameter* parameter);
    SLANG_API unsigned spReflectionParameter_GetBindingSpace(SlangReflectionParameter* parameter);

    SLANG_API SlangResult spIsParameterLocationUsed(
        SlangCompileRequest* request,
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        SlangParameterCategory category, // is this a `t` register? `s` register?
        SlangUInt spaceIndex,            // `space` for D3D12, `set` for Vulkan
        SlangUInt registerIndex,         // `register` for D3D12, `binding` for Vulkan
        bool& outUsed);

    // Entry Point Reflection

    SLANG_API char const* spReflectionEntryPoint_getName(SlangReflectionEntryPoint* entryPoint);

    SLANG_API char const* spReflectionEntryPoint_getNameOverride(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionFunction* spReflectionEntryPoint_getFunction(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API unsigned spReflectionEntryPoint_getParameterCount(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getParameterByIndex(
        SlangReflectionEntryPoint* entryPoint,
        unsigned index);

    SLANG_API SlangStage spReflectionEntryPoint_getStage(SlangReflectionEntryPoint* entryPoint);

    SLANG_API void spReflectionEntryPoint_getComputeThreadGroupSize(
        SlangReflectionEntryPoint* entryPoint,
        SlangUInt axisCount,
        SlangUInt* outSizeAlongAxis);

    SLANG_API void spReflectionEntryPoint_getComputeWaveSize(
        SlangReflectionEntryPoint* entryPoint,
        SlangUInt* outWaveSize);

    SLANG_API int spReflectionEntryPoint_usesAnySampleRateInput(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getVarLayout(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API SlangReflectionVariableLayout* spReflectionEntryPoint_getResultVarLayout(
        SlangReflectionEntryPoint* entryPoint);

    SLANG_API int spReflectionEntryPoint_hasDefaultConstantBuffer(
        SlangReflectionEntryPoint* entryPoint);

    // SlangReflectionTypeParameter
    SLANG_API char const* spReflectionTypeParameter_GetName(
        SlangReflectionTypeParameter* typeParam);
    SLANG_API unsigned spReflectionTypeParameter_GetIndex(SlangReflectionTypeParameter* typeParam);
    SLANG_API unsigned spReflectionTypeParameter_GetConstraintCount(
        SlangReflectionTypeParameter* typeParam);
    SLANG_API SlangReflectionType* spReflectionTypeParameter_GetConstraintByIndex(
        SlangReflectionTypeParameter* typeParam,
        unsigned int index);

    // Shader Reflection

    SLANG_API SlangResult spReflection_ToJson(
        SlangReflection* reflection,
        SlangCompileRequest* request,
        ISlangBlob** outBlob);

    SLANG_API unsigned spReflection_GetParameterCount(SlangReflection* reflection);
    SLANG_API SlangReflectionParameter* spReflection_GetParameterByIndex(
        SlangReflection* reflection,
        unsigned index);

    SLANG_API unsigned int spReflection_GetTypeParameterCount(SlangReflection* reflection);
    SLANG_API SlangReflectionTypeParameter* spReflection_GetTypeParameterByIndex(
        SlangReflection* reflection,
        unsigned int index);
    SLANG_API SlangReflectionTypeParameter* spReflection_FindTypeParameter(
        SlangReflection* reflection,
        char const* name);

    SLANG_API SlangReflectionType* spReflection_FindTypeByName(
        SlangReflection* reflection,
        char const* name);
    SLANG_API SlangReflectionTypeLayout* spReflection_GetTypeLayout(
        SlangReflection* reflection,
        SlangReflectionType* reflectionType,
        SlangLayoutRules rules);

    SLANG_API SlangReflectionFunction* spReflection_FindFunctionByName(
        SlangReflection* reflection,
        char const* name);
    SLANG_API SlangReflectionFunction* spReflection_FindFunctionByNameInType(
        SlangReflection* reflection,
        SlangReflectionType* reflType,
        char const* name);
    SLANG_API SlangReflectionVariable* spReflection_FindVarByNameInType(
        SlangReflection* reflection,
        SlangReflectionType* reflType,
        char const* name);

    SLANG_API SlangUInt spReflection_getEntryPointCount(SlangReflection* reflection);
    SLANG_API SlangReflectionEntryPoint* spReflection_getEntryPointByIndex(
        SlangReflection* reflection,
        SlangUInt index);
    SLANG_API SlangReflectionEntryPoint* spReflection_findEntryPointByName(
        SlangReflection* reflection,
        char const* name);

    SLANG_API SlangUInt spReflection_getGlobalConstantBufferBinding(SlangReflection* reflection);
    SLANG_API size_t spReflection_getGlobalConstantBufferSize(SlangReflection* reflection);

    SLANG_API SlangReflectionType* spReflection_specializeType(
        SlangReflection* reflection,
        SlangReflectionType* type,
        SlangInt specializationArgCount,
        SlangReflectionType* const* specializationArgs,
        ISlangBlob** outDiagnostics);

    SLANG_API SlangReflectionGeneric* spReflection_specializeGeneric(
        SlangReflection* inProgramLayout,
        SlangReflectionGeneric* generic,
        SlangInt argCount,
        SlangReflectionGenericArgType const* argTypes,
        SlangReflectionGenericArg const* args,
        ISlangBlob** outDiagnostics);

    SLANG_API bool spReflection_isSubType(
        SlangReflection* reflection,
        SlangReflectionType* subType,
        SlangReflectionType* superType);

    /// Get the number of hashed strings
    SLANG_API SlangUInt spReflection_getHashedStringCount(SlangReflection* reflection);

    /// Get a hashed string. The number of chars is written in outCount.
    /// The count does *NOT* including terminating 0. The returned string will be 0 terminated.
    SLANG_API const char* spReflection_getHashedString(
        SlangReflection* reflection,
        SlangUInt index,
        size_t* outCount);

    /// Compute a string hash.
    /// Count should *NOT* include terminating zero.
    SLANG_API SlangUInt32 spComputeStringHash(const char* chars, size_t count);

    /// Get a type layout representing reflection information for the global-scope parameters.
    SLANG_API SlangReflectionTypeLayout* spReflection_getGlobalParamsTypeLayout(
        SlangReflection* reflection);

    /// Get a variable layout representing reflection information for the global-scope parameters.
    SLANG_API SlangReflectionVariableLayout* spReflection_getGlobalParamsVarLayout(
        SlangReflection* reflection);

    SLANG_API char const* spGetTranslationUnitSource(
        SlangCompileRequest* request,
        int translationUnitIndex);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
SLANG_API slang::ISession* spReflection_GetSession(SlangReflection* reflection);

namespace slang
{
struct IComponentType;
struct IModule;
} // namespace slang

extern "C"
{
    /** @see slang::ICompileRequest::getProgram
     */
    SLANG_API SlangResult
    spCompileRequest_getProgram(SlangCompileRequest* request, slang::IComponentType** outProgram);

    /** @see slang::ICompileRequest::getProgramWithEntryPoints
     */
    SLANG_API SlangResult spCompileRequest_getProgramWithEntryPoints(
        SlangCompileRequest* request,
        slang::IComponentType** outProgram);

    /** @see slang::ICompileRequest::getEntryPoint
     */
    SLANG_API SlangResult spCompileRequest_getEntryPoint(
        SlangCompileRequest* request,
        SlangInt entryPointIndex,
        slang::IComponentType** outEntryPoint);

    /** @see slang::ICompileRequest::getModule
     */
    SLANG_API SlangResult spCompileRequest_getModule(
        SlangCompileRequest* request,
        SlangInt translationUnitIndex,
        slang::IModule** outModule);

    /** @see slang::ICompileRequest::getSession
     */
    SLANG_API SlangResult
    spCompileRequest_getSession(SlangCompileRequest* request, slang::ISession** outSession);
}

namespace slang
{
/*!
@brief A request for one or more compilation actions to be performed.
*/
struct ICompileRequest : public ISlangUnknown
{
    SLANG_COM_INTERFACE(
        0x96d33993,
        0x317c,
        0x4db5,
        {0xaf, 0xd8, 0x66, 0x6e, 0xe7, 0x72, 0x48, 0xe2})

    /** Set the filesystem hook to use for a compile request

    The provided `fileSystem` will be used to load any files that
    need to be loaded during processing of the compile `request`.
    This includes:

      - Source files loaded via `spAddTranslationUnitSourceFile`
      - Files referenced via `#include`
      - Files loaded to resolve `#import` operations
        */
    virtual SLANG_NO_THROW void SLANG_MCALL setFileSystem(ISlangFileSystem* fileSystem) = 0;

    /*!
    @brief Set flags to be used for compilation.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setCompileFlags(SlangCompileFlags flags) = 0;

    /*!
    @brief Returns the compilation flags previously set with `setCompileFlags`
    */
    virtual SLANG_NO_THROW SlangCompileFlags SLANG_MCALL getCompileFlags() = 0;

    /*!
    @brief Set whether to dump intermediate results (for debugging) or not.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setDumpIntermediates(int enable) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setDumpIntermediatePrefix(const char* prefix) = 0;

    /*!
    @brief Set whether (and how) `#line` directives should be output.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setLineDirectiveMode(SlangLineDirectiveMode mode) = 0;

    /*!
    @brief Sets the target for code generation.
    @param target The code generation target. Possible values are:
    - SLANG_GLSL. Generates GLSL code.
    - SLANG_HLSL. Generates HLSL code.
    - SLANG_SPIRV. Generates SPIR-V code.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setCodeGenTarget(SlangCompileTarget target) = 0;

    /*!
    @brief Add a code-generation target to be used.
    */
    virtual SLANG_NO_THROW int SLANG_MCALL addCodeGenTarget(SlangCompileTarget target) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetProfile(int targetIndex, SlangProfileID profile) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetFlags(int targetIndex, SlangTargetFlags flags) = 0;

    /*!
    @brief Set the floating point mode (e.g., precise or fast) to use a target.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetFloatingPointMode(int targetIndex, SlangFloatingPointMode mode) = 0;

    /* DEPRECATED: use `spSetMatrixLayoutMode` instead. */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetMatrixLayoutMode(int targetIndex, SlangMatrixLayoutMode mode) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setMatrixLayoutMode(SlangMatrixLayoutMode mode) = 0;

    /*!
    @brief Set the level of debug information to produce.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setDebugInfoLevel(SlangDebugInfoLevel level) = 0;

    /*!
    @brief Set the level of optimization to perform.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setOptimizationLevel(SlangOptimizationLevel level) = 0;


    /*!
    @brief Set the container format to be used for binary output.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setOutputContainerFormat(SlangContainerFormat format) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setPassThrough(SlangPassThrough passThrough) = 0;


    virtual SLANG_NO_THROW void SLANG_MCALL
    setDiagnosticCallback(SlangDiagnosticCallback callback, void const* userData) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setWriter(SlangWriterChannel channel, ISlangWriter* writer) = 0;

    virtual SLANG_NO_THROW ISlangWriter* SLANG_MCALL getWriter(SlangWriterChannel channel) = 0;

    /*!
    @brief Add a path to use when searching for referenced files.
    This will be used for both `#include` directives and also for explicit `__import` declarations.
    @param ctx The compilation context.
    @param searchDir The additional search directory.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL addSearchPath(const char* searchDir) = 0;

    /*!
    @brief Add a macro definition to be used during preprocessing.
    @param key The name of the macro to define.
    @param value The value of the macro to define.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    addPreprocessorDefine(const char* key, const char* value) = 0;

    /*!
    @brief Set options using arguments as if specified via command line.
    @return Returns SlangResult. On success SLANG_SUCCEEDED(result) is true.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    processCommandLineArguments(char const* const* args, int argCount) = 0;

    /** Add a distinct translation unit to the compilation request

    `name` is optional.
    Returns the zero-based index of the translation unit created.
    */
    virtual SLANG_NO_THROW int SLANG_MCALL
    addTranslationUnit(SlangSourceLanguage language, char const* name) = 0;


    /** Set a default module name. Translation units will default to this module name if one is not
    passed. If not set each translation unit will get a unique name.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL setDefaultModuleName(const char* defaultModuleName) = 0;

    /** Add a preprocessor definition that is scoped to a single translation unit.

    @param translationUnitIndex The index of the translation unit to get the definition.
    @param key The name of the macro to define.
    @param value The value of the macro to define.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitPreprocessorDefine(
        int translationUnitIndex,
        const char* key,
        const char* value) = 0;


    /** Add a source file to the given translation unit.

    If a user-defined file system has been specified via
    `spSetFileSystem`, then it will be used to load the
    file at `path`. Otherwise, Slang will use the OS
    file system.

    This function does *not* search for a file using
    the registered search paths (`spAddSearchPath`),
    and instead using the given `path` as-is.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    addTranslationUnitSourceFile(int translationUnitIndex, char const* path) = 0;

    /** Add a source string to the given translation unit.

    @param translationUnitIndex The index of the translation unit to add source to.
    @param path The file-system path that should be assumed for the source code.
    @param source A null-terminated UTF-8 encoded string of source code.

    The implementation will make a copy of the source code data.
    An application may free the buffer immediately after this call returns.

    The `path` will be used in any diagnostic output, as well
    as to determine the base path when resolving relative
    `#include`s.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceString(
        int translationUnitIndex,
        char const* path,
        char const* source) = 0;


    /** Add a slang library - such that its contents can be referenced during linking.
    This is equivalent to the -r command line option.

    @param basePath The base path used to lookup referenced modules.
    @param libData The library data
    @param libDataSize The size of the library data
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    addLibraryReference(const char* basePath, const void* libData, size_t libDataSize) = 0;

    /** Add a source string to the given translation unit.

    @param translationUnitIndex The index of the translation unit to add source to.
    @param path The file-system path that should be assumed for the source code.
    @param sourceBegin A pointer to a buffer of UTF-8 encoded source code.
    @param sourceEnd A pointer to to the end of the buffer specified in `sourceBegin`

    The implementation will make a copy of the source code data.
    An application may free the buffer immediately after this call returns.

    The `path` will be used in any diagnostic output, as well
    as to determine the base path when resolving relative
    `#include`s.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceStringSpan(
        int translationUnitIndex,
        char const* path,
        char const* sourceBegin,
        char const* sourceEnd) = 0;

    /** Add a blob of source code to the given translation unit.

    @param translationUnitIndex The index of the translation unit to add source to.
    @param path The file-system path that should be assumed for the source code.
    @param sourceBlob A blob containing UTF-8 encoded source code.
    @param sourceEnd A pointer to to the end of the buffer specified in `sourceBegin`

    The compile request will retain a reference to the blob.

    The `path` will be used in any diagnostic output, as well
    as to determine the base path when resolving relative
    `#include`s.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL addTranslationUnitSourceBlob(
        int translationUnitIndex,
        char const* path,
        ISlangBlob* sourceBlob) = 0;

    /** Add an entry point in a particular translation unit
     */
    virtual SLANG_NO_THROW int SLANG_MCALL
    addEntryPoint(int translationUnitIndex, char const* name, SlangStage stage) = 0;

    /** Add an entry point in a particular translation unit,
        with additional arguments that specify the concrete
        type names for entry-point generic type parameters.
    */
    virtual SLANG_NO_THROW int SLANG_MCALL addEntryPointEx(
        int translationUnitIndex,
        char const* name,
        SlangStage stage,
        int genericArgCount,
        char const** genericArgs) = 0;

    /** Specify the arguments to use for global generic parameters.
     */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    setGlobalGenericArgs(int genericArgCount, char const** genericArgs) = 0;

    /** Specify the concrete type to be used for a global "existential slot."

    Every shader parameter (or leaf field of a `struct`-type shader parameter)
    that has an interface or array-of-interface type introduces an existential
    slot. The number of slots consumed by a shader parameter, and the starting
    slot of each parameter can be queried via the reflection API using
    `SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM`.

    In order to generate specialized code, a concrete type needs to be specified
    for each existential slot. This function specifies the name of the type
    (or in general a type *expression*) to use for a specific slot at the
    global scope.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    setTypeNameForGlobalExistentialTypeParam(int slotIndex, char const* typeName) = 0;

    /** Specify the concrete type to be used for an entry-point "existential slot."

    Every shader parameter (or leaf field of a `struct`-type shader parameter)
    that has an interface or array-of-interface type introduces an existential
    slot. The number of slots consumed by a shader parameter, and the starting
    slot of each parameter can be queried via the reflection API using
    `SLANG_PARAMETER_CATEGORY_EXISTENTIAL_TYPE_PARAM`.

    In order to generate specialized code, a concrete type needs to be specified
    for each existential slot. This function specifies the name of the type
    (or in general a type *expression*) to use for a specific slot at the
    entry-point scope.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL setTypeNameForEntryPointExistentialTypeParam(
        int entryPointIndex,
        int slotIndex,
        char const* typeName) = 0;

    /** Enable or disable an experimental, best-effort GLSL frontend
     */
    virtual SLANG_NO_THROW void SLANG_MCALL setAllowGLSLInput(bool value) = 0;

    /** Execute the compilation request.

    @returns  SlangResult, SLANG_OK on success. Use SLANG_SUCCEEDED() and SLANG_FAILED() to test
    SlangResult.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL compile() = 0;


    /** Get any diagnostic messages reported by the compiler.

    @returns A null-terminated UTF-8 encoded string of diagnostic messages.

    The returned pointer is only guaranteed to be valid
    until `request` is destroyed. Applications that wish to
    hold on to the diagnostic output for longer should use
    `getDiagnosticOutputBlob`.
    */
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDiagnosticOutput() = 0;

    /** Get diagnostic messages reported by the compiler.

    @param outBlob A pointer to receive a blob holding a nul-terminated UTF-8 encoded string of
    diagnostic messages.
    @returns A `SlangResult` indicating success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getDiagnosticOutputBlob(ISlangBlob** outBlob) = 0;


    /** Get the number of files that this compilation depended on.

    This includes both the explicit source files, as well as any
    additional files that were transitively referenced (e.g., via
    a `#include` directive).
    */
    virtual SLANG_NO_THROW int SLANG_MCALL getDependencyFileCount() = 0;

    /** Get the path to a file this compilation depended on.
     */
    virtual SLANG_NO_THROW char const* SLANG_MCALL getDependencyFilePath(int index) = 0;

    /** Get the number of translation units associated with the compilation request
     */
    virtual SLANG_NO_THROW int SLANG_MCALL getTranslationUnitCount() = 0;

    /** Get the output source code associated with a specific entry point.

    The lifetime of the output pointer is the same as `request`.
    */
    virtual SLANG_NO_THROW char const* SLANG_MCALL getEntryPointSource(int entryPointIndex) = 0;

    /** Get the output bytecode associated with a specific entry point.

    The lifetime of the output pointer is the same as `request`.
    */
    virtual SLANG_NO_THROW void const* SLANG_MCALL
    getEntryPointCode(int entryPointIndex, size_t* outSize) = 0;

    /** Get the output code associated with a specific entry point.

    @param entryPointIndex The index of the entry point to get code for.
    @param targetIndex The index of the target to get code for (default: zero).
    @param outBlob A pointer that will receive the blob of code
    @returns A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getEntryPointCodeBlob(int entryPointIndex, int targetIndex, ISlangBlob** outBlob) = 0;

    /** Get entry point 'callable' functions accessible through the ISlangSharedLibrary interface.

    That the functions remain in scope as long as the ISlangSharedLibrary interface is in scope.

    NOTE! Requires a compilation target of SLANG_HOST_CALLABLE.

    @param entryPointIndex  The index of the entry point to get code for.
    @param targetIndex      The index of the target to get code for (default: zero).
    @param outSharedLibrary A pointer to a ISharedLibrary interface which functions can be queried
    on.
    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getEntryPointHostCallable(
        int entryPointIndex,
        int targetIndex,
        ISlangSharedLibrary** outSharedLibrary) = 0;

    /** Get the output code associated with a specific target.

    @param targetIndex The index of the target to get code for (default: zero).
    @param outBlob A pointer that will receive the blob of code
    @returns A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getTargetCodeBlob(int targetIndex, ISlangBlob** outBlob) = 0;

    /** Get 'callable' functions for a target accessible through the ISlangSharedLibrary interface.

    That the functions remain in scope as long as the ISlangSharedLibrary interface is in scope.

    NOTE! Requires a compilation target of SLANG_HOST_CALLABLE.

    @param targetIndex      The index of the target to get code for (default: zero).
    @param outSharedLibrary A pointer to a ISharedLibrary interface which functions can be queried
    on.
    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getTargetHostCallable(int targetIndex, ISlangSharedLibrary** outSharedLibrary) = 0;

    /** Get the output bytecode associated with an entire compile request.

    The lifetime of the output pointer is the same as `request` and the last spCompile.

    @param outSize          The size of the containers contents in bytes. Will be zero if there is
    no code available.
    @returns                Pointer to start of the contained data, or nullptr if there is no code
    available.
    */
    virtual SLANG_NO_THROW void const* SLANG_MCALL getCompileRequestCode(size_t* outSize) = 0;

    /** Get the compilation result as a file system.
    The result is not written to the actual OS file system, but is made available as an
    in memory representation.
    */
    virtual SLANG_NO_THROW ISlangMutableFileSystem* SLANG_MCALL
    getCompileRequestResultAsFileSystem() = 0;

    /** Return the container code as a blob. The container blob is created as part of a compilation
    (with spCompile), and a container is produced with a suitable ContainerFormat.

    @param outSize          The blob containing the container data.
    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getContainerCode(ISlangBlob** outBlob) = 0;

    /** Load repro from memory specified.

    Should only be performed on a newly created request.

    NOTE! When using the fileSystem, files will be loaded via their `unique names` as if they are
    part of the flat file system. This mechanism is described more fully in docs/repro.md.

    @param fileSystem       An (optional) filesystem. Pass nullptr to just use contents of repro
    held in data.
    @param data             The data to load from.
    @param size             The size of the data to load from.
    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadRepro(ISlangFileSystem* fileSystem, const void* data, size_t size) = 0;

    /** Save repro state. Should *typically* be performed after spCompile, so that everything
    that is needed for a compilation is available.

    @param outBlob          Blob that will hold the serialized state
    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL saveRepro(ISlangBlob** outBlob) = 0;

    /** Enable repro capture.

    Should be set after any ISlangFileSystem has been set, but before any compilation. It ensures
    that everything that the ISlangFileSystem accesses will be correctly recorded. Note that if a
    ISlangFileSystem/ISlangFileSystemExt isn't explicitly set (ie the default is used), then the
    request will automatically be set up to record everything appropriate.

    @returns                A `SlangResult` to indicate success or failure.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL enableReproCapture() = 0;

    /** Get the (linked) program for a compile request.

    The linked program will include all of the global-scope modules for the
    translation units in the program, plus any modules that they `import`
    (transitively), specialized to any global specialization arguments that
    were provided via the API.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getProgram(slang::IComponentType** outProgram) = 0;

    /** Get the (partially linked) component type for an entry point.

    The returned component type will include the entry point at the
    given index, and will be specialized using any specialization arguments
    that were provided for it via the API.

    The returned component will *not* include the modules representing
    the global scope and its dependencies/specialization, so a client
    program will typically want to compose this component type with
    the one returned by `spCompileRequest_getProgram` to get a complete
    and usable component type from which kernel code can be requested.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getEntryPoint(SlangInt entryPointIndex, slang::IComponentType** outEntryPoint) = 0;

    /** Get the (un-linked) module for a translation unit.

    The returned module will not be linked against any dependencies,
    nor against any entry points (even entry points declared inside
    the module). Similarly, the module will not be specialized
    to the arguments that might have been provided via the API.

    This function provides an atomic unit of loaded code that
    is suitable for looking up types and entry points in the
    given module, and for linking together to produce a composite
    program that matches the needs of an application.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getModule(SlangInt translationUnitIndex, slang::IModule** outModule) = 0;

    /** Get the `ISession` handle behind the `SlangCompileRequest`.
    TODO(JS): Arguably this should just return the session pointer.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL getSession(slang::ISession** outSession) = 0;

    /** get reflection data from a compilation request */
    virtual SLANG_NO_THROW SlangReflection* SLANG_MCALL getReflection() = 0;

    /** Make output specially handled for command line output */
    virtual SLANG_NO_THROW void SLANG_MCALL setCommandLineCompilerMode() = 0;

    /** Add a defined capability that should be assumed available on the target */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    addTargetCapability(SlangInt targetIndex, SlangCapabilityID capability) = 0;

    /** Get the (linked) program for a compile request, including all entry points.

    The resulting program will include all of the global-scope modules for the
    translation units in the program, plus any modules that they `import`
    (transitively), specialized to any global specialization arguments that
    were provided via the API, as well as all entry points specified for compilation,
    specialized to their entry-point specialization arguments.
    */
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getProgramWithEntryPoints(slang::IComponentType** outProgram) = 0;

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL isParameterLocationUsed(
        SlangInt entryPointIndex,
        SlangInt targetIndex,
        SlangParameterCategory category,
        SlangUInt spaceIndex,
        SlangUInt registerIndex,
        bool& outUsed) = 0;

    /** Set the line directive mode for a target.
     */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetLineDirectiveMode(SlangInt targetIndex, SlangLineDirectiveMode mode) = 0;

    /** Set whether to use scalar buffer layouts for GLSL/Vulkan targets.
        If true, the generated GLSL/Vulkan code will use `scalar` layout for storage buffers.
        If false, the resulting code will std430 for storage buffers.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetForceGLSLScalarBufferLayout(int targetIndex, bool forceScalarLayout) = 0;

    /** Overrides the severity of a specific diagnostic message.

    @param messageID            Numeric identifier of the message to override,
                                as defined in the 1st parameter of the DIAGNOSTIC macro.
    @param overrideSeverity     New severity of the message. If the message is originally Error or
    Fatal, the new severity cannot be lower than that.
    */
    virtual SLANG_NO_THROW void SLANG_MCALL
    overrideDiagnosticSeverity(SlangInt messageID, SlangSeverity overrideSeverity) = 0;

    /** Returns the currently active flags of the request's diagnostic sink. */
    virtual SLANG_NO_THROW SlangDiagnosticFlags SLANG_MCALL getDiagnosticFlags() = 0;

    /** Sets the flags of the request's diagnostic sink.
        The previously specified flags are discarded. */
    virtual SLANG_NO_THROW void SLANG_MCALL setDiagnosticFlags(SlangDiagnosticFlags flags) = 0;

    /** Set the debug format to be used for debugging information */
    virtual SLANG_NO_THROW void SLANG_MCALL
    setDebugInfoFormat(SlangDebugInfoFormat debugFormat) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setEnableEffectAnnotations(bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setReportDownstreamTime(bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setReportPerfBenchmark(bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setSkipSPIRVValidation(bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetUseMinimumSlangOptimization(int targetIndex, bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setIgnoreCapabilityCheck(bool value) = 0;

    // return a copy of internal profiling results, and if `shouldClear` is true, clear the internal
    // profiling results before returning.
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    getCompileTimeProfile(ISlangProfiler** compileTimeProfile, bool shouldClear) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetGenerateWholeProgram(int targetIndex, bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL setTargetForceDXLayout(int targetIndex, bool value) = 0;

    virtual SLANG_NO_THROW void SLANG_MCALL
    setTargetEmbedDownstreamIR(int targetIndex, bool value) = 0;
};

    #define SLANG_UUID_ICompileRequest ICompileRequest::getTypeGuid()

} // namespace slang
#endif
