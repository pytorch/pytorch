#ifdef SLANG_IN_SPIRV_EMIT_CONTEXT

// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc#DebugCompilationUnit
template<typename T>
SpvInst* emitOpDebugCompilationUnit(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* version,
    SpvInst* dwarfVersion,
    SpvInst* source,
    SpvInst* language)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(1),
        version,
        dwarfVersion,
        source,
        language);
}

// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc#DebugSource
template<typename T>
SpvInst* emitOpDebugSource(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* file,
    SpvInst* text = nullptr)
{
    static_assert(isSingular<T>);
    if (text)
        return emitInst(
            parent,
            inst,
            SpvOpExtInst,
            idResultType,
            kResultID,
            set,
            SpvWord(35),
            file,
            text);
    else
        return emitInst(
            parent,
            inst,
            SpvOpExtInst,
            idResultType,
            kResultID,
            set,
            SpvWord(35),
            file);
}

template<typename T>
SpvInst* emitOpDebugSourceContinued(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* text)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpExtInst, idResultType, kResultID, set, SpvWord(102), text);
}

// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc#DebugLine
template<typename T>
SpvInst* emitOpDebugLine(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* source,
    IRInst* lineStart,
    IRInst* lineEnd,
    IRInst* colStart,
    IRInst* colEnd)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(103),
        source,
        lineStart,
        lineEnd,
        colStart,
        colEnd);
}

SpvInst* emitOpDebugEntryPoint(
    SpvInstParent* parent,
    IRInst* resultType,
    SpvInst* set,
    SpvInst* entryPoint,
    SpvInst* scope,
    IRInst* compiler,
    IRInst* args)
{
    return emitInst(
        parent,
        nullptr,
        SpvOpExtInst,
        resultType,
        kResultID,
        set,
        SpvWord(107),
        entryPoint,
        scope,
        compiler,
        args);
}

// https://github.com/KhronosGroup/SPIRV-Registry/blob/main/nonsemantic/NonSemantic.Shader.DebugInfo.100.asciidoc#DebugFunction
template<typename T>
SpvInst* emitOpDebugFunction(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* name,
    SpvInst* type,
    IRInst* source,
    IRInst* lineStart,
    IRInst* colStart,
    SpvInst* scope,
    IRInst* linkageName,
    IRInst* flag,
    IRInst* scopeLine)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(20),
        name,
        type,
        source,
        lineStart,
        colStart,
        scope,
        linkageName,
        flag,
        scopeLine);
}

template<typename T>
SpvInst* emitOpDebugFunctionDefinition(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* debugFunc,
    SpvInst* spvFunc)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(101),
        debugFunc,
        spvFunc);
}

template<typename T, typename Ts>
SpvInst* emitOpDebugTypeFunction(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* flags,
    SpvInst* returnType,
    const Ts& argTypes)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(8),
        flags,
        returnType,
        argTypes);
}

template<typename T, typename Ts>
SpvInst* emitOpDebugTypeComposite(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* name,
    IRInst* tag,
    IRInst* source,
    IRInst* line,
    IRInst* col,
    SpvInst* scope,
    IRInst* linkageName,
    IRInst* size,
    IRInst* flags,
    const Ts& members)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(10),
        name,
        tag,
        source,
        line,
        col,
        scope,
        linkageName,
        size,
        flags,
        members);
}

template<typename T>
SpvInst* emitOpDebugTypeMember(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* name,
    SpvInst* type,
    IRInst* source,
    IRInst* line,
    IRInst* col,
    IRInst* offset,
    IRInst* size,
    IRInst* flags)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(11),
        name,
        type,
        source,
        line,
        col,
        offset,
        size,
        flags);
}

template<typename T>
SpvInst* emitOpDebugTypeArray(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* baseType,
    IRInst* elementCount)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(5),
        baseType,
        elementCount);
}

template<typename T>
SpvInst* emitOpDebugTypeBasic(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* name,
    IRInst* size,
    IRInst* encoding,
    IRInst* flags)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(2),
        name,
        size,
        encoding,
        flags);
}

template<typename T>
SpvInst* emitOpDebugTypeVector(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* baseType,
    IRInst* elementCount)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(6),
        baseType,
        elementCount);
}

template<typename T>
SpvInst* emitOpDebugTypeMatrix(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* vectorType,
    IRInst* vectorCount,
    IRInst* columnMajor)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(108),
        vectorType,
        vectorCount,
        columnMajor);
}

template<typename T>
SpvInst* emitOpDebugTypePointer(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* baseType,
    IRInst* storageClass,
    IRInst* flags)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(3),
        baseType,
        storageClass,
        flags);
}

template<typename T>
SpvInst* emitOpDebugTypeQualifier(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* baseType,
    IRInst* typeQualifier)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(4),
        baseType,
        typeQualifier);
}

template<typename T>
SpvInst* emitOpDebugScope(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* scope)
{
    static_assert(isSingular<T>);
    return emitInst(parent, inst, SpvOpExtInst, idResultType, kResultID, set, SpvWord(23), scope);
}

template<typename T>
SpvInst* emitOpDebugLocalVariable(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* name,
    SpvInst* type,
    IRInst* source,
    IRInst* line,
    IRInst* col,
    SpvInst* scope,
    IRInst* flags,
    IRInst* argIndex)
{
    static_assert(isSingular<T>);
    if (argIndex)
        return emitInst(
            parent,
            inst,
            SpvOpExtInst,
            idResultType,
            kResultID,
            set,
            SpvWord(26),
            name,
            type,
            source,
            line,
            col,
            scope,
            flags,
            argIndex);
    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(26),
        name,
        type,
        source,
        line,
        col,
        scope,
        flags);
}

template<typename T, typename Ts>
SpvInst* emitOpDebugValue(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* localVar,
    IRInst* value,
    SpvInst* expression,
    const Ts& indices)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);

    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(29),
        localVar,
        value,
        expression,
        indices);
}

template<typename T, typename Ts>
SpvInst* emitOpDebugDeclare(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    SpvInst* debugLocalVar,
    IRInst* actualLocalVar,
    SpvInst* expression,
    const Ts& indices)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);

    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(28),
        debugLocalVar,
        actualLocalVar,
        expression,
        indices);
}
template<typename T, typename Ts>
SpvInst* emitOpDebugExpression(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    const Ts& operations)
{
    static_assert(isSingular<T>);
    static_assert(isPlural<Ts>);

    return emitInst(
        parent,
        inst,
        SpvOpExtInst,
        idResultType,
        kResultID,
        set,
        SpvWord(31),
        operations);
}

template<typename T>
SpvInst* emitOpDebugForwardRefsComposite(
    SpvInstParent* parent,
    IRInst* inst,
    const T& idResultType,
    SpvInst* set,
    IRInst* name,
    IRInst* tag,
    IRInst* source,
    IRInst* line,
    IRInst* col,
    SpvInst* scope,
    IRInst* linkageName,
    IRInst* size,
    IRInst* flags)
{
    static_assert(isSingular<T>);
    return emitInst(
        parent,
        inst,
        SpvOpExtInstWithForwardRefsKHR,
        idResultType,
        kResultID,
        set,
        SpvWord(10),
        name,
        tag,
        source,
        line,
        col,
        scope,
        linkageName,
        size,
        flags);
}

#endif // SLANG_IN_SPIRV_EMIT_CONTEXT
