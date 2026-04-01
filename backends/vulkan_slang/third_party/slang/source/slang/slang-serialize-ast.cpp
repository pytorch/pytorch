// slang-serialize-ast.cpp
#include "slang-serialize-ast.h"

#include "slang-ast-dispatch.h"
#include "slang-compiler.h"
#include "slang-diagnostics.h"
#include "slang-mangle.h"

namespace Slang
{
// TODO(tfoley): have the parser export this, or a utility function
// for initializing a `SyntaxDecl` in the common case.
//
NodeBase* parseSimpleSyntax(Parser* parser, void* userData);


struct ASTEncodingContext
{
private:
    Encoder* encoder;
    struct UnhandledCase
    {
    };

    typedef Int DeclID;
    Dictionary<Decl*, DeclID> mapDeclToID;
    List<Decl*> decls;

    struct ImportedDeclInfo
    {
        Int moduleIndex = -1;
        Decl* decl;
    };
    List<ImportedDeclInfo> importedDecls;

    typedef Int ValID;
    Dictionary<Val*, ValID> mapValToID;
    List<Val*> vals;

    ModuleDecl* _module = nullptr;

    SerialSourceLocWriter* _sourceLocWriter = nullptr;

public:
    ASTEncodingContext(Encoder* encoder, ModuleDecl* module, SerialSourceLocWriter* sourceLocWriter)
        : encoder(encoder), _module(module), _sourceLocWriter(sourceLocWriter)
    {
    }

    template<typename T>
    void encodeASTNodeContent(T* node)
    {
        Encoder::WithObject withObject(encoder);

        ASTNodeDispatcher<T, void>::dispatch(node, [&](auto n) { _encodeDataOf(n); });
    }

    void flush()
    {
        auto containerChunk = encoder->getRIFFChunk();

        RiffContainer::Chunk* declChunk = nullptr;
        RiffContainer::Chunk* importedDeclChunk = nullptr;
        RiffContainer::Chunk* valChunk = nullptr;
        {
            Encoder::WithArray withList(encoder);
            declChunk = encoder->getRIFFChunk();
        }
        {
            Encoder::WithArray withList(encoder);
            importedDeclChunk = encoder->getRIFFChunk();
        }
        {
            Encoder::WithArray withList(encoder);
            valChunk = encoder->getRIFFChunk();
        }
        Int declIndex = 0;
        Int importedDeclIndex = 0;
        Int valIndex = 0;

        bool done = false;
        do
        {
            done = true;
            while (declIndex < decls.getCount())
            {
                done = false;
                encoder->setRIFFChunk(declChunk);
                encodeASTNodeContent(decls[declIndex++]);
            }
            while (importedDeclIndex < importedDecls.getCount())
            {
                done = false;
                encoder->setRIFFChunk(importedDeclChunk);
                encodeImportedDecl(importedDecls[importedDeclIndex++]);
            }
            while (valIndex < vals.getCount())
            {
                done = false;
                encoder->setRIFFChunk(valChunk);
                encodeASTNodeContent(vals[valIndex++]);
            }
        } while (!done);

        RiffContainer::calcAndSetSize(containerChunk);
        encoder->setRIFFChunk(containerChunk);
    }

    ModuleDecl* findModuleForDecl(Decl* decl)
    {
        for (auto d = decl; d; d = d->parentDecl)
        {
            if (auto m = as<ModuleDecl>(d))
                return m;
        }
        return nullptr;
    }

    ModuleDecl* findModuleDeclWasImportedFrom(Decl* decl)
    {
        auto declModule = findModuleForDecl(decl);
        if (declModule == nullptr)
            return nullptr;
        if (declModule == _module)
            return nullptr;
        return declModule;
    }

    DeclID getDeclID(Decl* decl)
    {
        SLANG_ASSERT(decl != nullptr);

        if (auto found = mapDeclToID.tryGetValue(decl))
            return *found;

        // We need to detect whether the declaration is an
        // imported one, or one from this module itself.
        //
        // Imported declarations need to be handled very
        // differently, since they'll involve resolving
        // references to those other modules, and the
        // declarations within them.
        //
        if (auto importedFromModule = findModuleDeclWasImportedFrom(decl))
        {
            DeclID importedFromModuleDeclID = 0;
            if (decl != importedFromModule)
            {
                importedFromModuleDeclID = getDeclID(importedFromModule);
            }

            DeclID id = ~importedDecls.getCount();
            mapDeclToID.add(decl, id);

            ImportedDeclInfo info;
            info.moduleIndex = ~importedFromModuleDeclID;
            info.decl = decl;
            importedDecls.add(info);

            return id;
        }
        else
        {
            DeclID id = decls.getCount();
            decls.add(decl);
            mapDeclToID.add(decl, id);

            return id;
        }
    }

    void encodePtr(Decl* decl)
    {
        DeclID id = getDeclID(decl);
        encoder->encode(id);
    }

    ValID getValID(Val* val)
    {
        SLANG_ASSERT(val != nullptr);

        if (auto found = mapValToID.tryGetValue(val))
            return *found;

        // In order to ensure that values can be fully constructed
        // from the get-go (so that they will get cached correctly),
        // we conspire to ensure that every value is preceded by
        // all of its operands.
        //
        for (auto operand : val->m_operands)
        {
            switch (operand.kind)
            {
            default:
                break;

            case ValNodeOperandKind::ValNode:
                if (auto operandNode = operand.values.nodeOperand)
                {
                    SLANG_ASSERT(as<Val>(operandNode));
                    getValID(static_cast<Val*>(operandNode));
                }
                break;

            case ValNodeOperandKind::ASTNode:
                if (auto operandNode = operand.values.nodeOperand)
                {
                    SLANG_ASSERT(as<Decl>(operandNode));
                    getDeclID(static_cast<Decl*>(operandNode));
                }
                break;
            }
        }
        auto resolved = val->resolve();
        if (resolved != val)
        {
            getValID(resolved);
        }

        ValID id = vals.getCount();
        vals.add(val);
        mapValToID.add(val, id);
        return id;
    }

    void encodePtr(Val* val)
    {
        ValID id = getValID(val);
        encoder->encode(id);
    }

    void encodeImportedDecl(ImportedDeclInfo const& info)
    {
        Encoder::WithKeyValuePair withPair(encoder);
        encode(info.moduleIndex);
        auto decl = info.decl;
        if (auto importedModuleDecl = as<ModuleDecl>(decl))
        {
            SLANG_ASSERT(info.moduleIndex == -1);
            encode(importedModuleDecl->getName());
        }
        else
        {
            auto mangledName = getMangledName(getCurrentASTBuilder(), decl);
            encode(mangledName);
        }
    }

    void encodePtr(Modifier* modifier) { encodeASTNodeContent(modifier); }
    void encodePtr(Expr* expr) { encodeASTNodeContent(expr); }
    void encodePtr(Stmt* stmt) { encodeASTNodeContent(stmt); }

    void encodePtr(Name* name) { encode(name->text); }

    void encodePtr(MarkupEntry* entry)
    {
        // TODO: is this case needed?
        SLANG_UNUSED(entry);
    }

    void encodePtr(DeclAssociationList* list)
    {
        // We serialize this as if it were a simple list
        // of key-value pairs because... well... that's
        // what it amounts to in practice.
        //
        Encoder::WithArray withArray(encoder);
        for (auto association : list->associations)
        {
            Encoder::WithKeyValuePair withPair(encoder);
            encode(association->kind);
            encode(association->decl);
        }
    }

    void encodePtr(CandidateExtensionList* list) { encode(list->candidateExtensions); }

    void encodePtr(WitnessTable* witnessTable)
    {
        Encoder::WithObject withObject(encoder);
        encode(witnessTable->baseType);
        encode(witnessTable->witnessedType);
        encode(witnessTable->isExtern);

        // TODO(tfoley): In theory we should be able to streamline
        // this so that we only encode the requirements that we
        // absolutely need to (which basically amounts to `associatedtype`
        // requirements where the satisfying type is part of the public
        // API of the type).
        //
        encode(witnessTable->m_requirementDictionary);
    }

    void encodeValue(RequirementWitness const& witness)
    {
        Encoder::WithKeyValuePair withPair(encoder);
        encodeEnum(witness.m_flavor);
        switch (witness.m_flavor)
        {
        case RequirementWitness::Flavor::none:
            break;

        case RequirementWitness::Flavor::declRef:
            encode(witness.m_declRef);
            break;

        case RequirementWitness::Flavor::val:
            encode(witness.m_val);
            break;

        case RequirementWitness::Flavor::witnessTable:
            encode((WitnessTable*)witness.m_obj.Ptr());
            break;
        }
    }

    void encodePtr(DiagnosticInfo* info) { encode(Int(info->id)); }

    void encodePtr(DeclBase* declBase)
    {
        if (auto decl = as<Decl>(declBase))
        {
            encodePtr(decl);
        }
        else
        {
            encodeASTNodeContent(declBase);
        }
    }

    void encodeValue(UnhandledCase);

    void encodeValue(String const& value) { encoder->encode(value); }

    void encodeValue(Token const& value)
    {
        encode(value.type);
        encode(TokenFlags(value.flags & ~TokenFlag::Name));
        encode(value.loc);
        if (value.hasContent())
            encoder->encodeString(value.getContent());
        else
            encode(nullptr);
    }

    void encodeValue(NameLoc const& value) { encode(value.name); }

    void encodeValue(SemanticVersion value) { encoder->encode(value.toInteger()); }

    void encodeValue(CapabilitySet const& value)
    {
        // While the `CapabilityTargetSets` type is a dictionary,
        // in practice each entry already embeds its own key
        // (the target atom), so we can encode this as just
        // an array of the `CapabilityTargetSet` values.
        //
        Encoder::WithArray withArray(encoder);
        for (auto pair : value.getCapabilityTargetSets())
        {
            encode(pair.second);
        }
    }

    void encodeValue(CapabilityTargetSet const& value)
    {
        Encoder::WithKeyValuePair withPair(encoder);
        encode(value.target);

        // Similar to the case for the `CapabilityTargetSets` above,
        // each `CapabilityStageSet` already includes the stage atom,
        // so we can simply encode the values from the dictionary.
        //
        Encoder::WithArray withArray(encoder);
        for (auto pair : value.shaderStageSets)
        {
            encode(pair.second);
        }
    }

    void encodeValue(CapabilityStageSet const& value)
    {
        Encoder::WithKeyValuePair withPair(encoder);
        encode(value.stage);
        encode(value.atomSet);
    }

    void encodeValue(CapabilityAtomSet const& value)
    {
        Encoder::WithArray withArray(encoder);
        for (auto rawAtom : value)
        {
            encode(CapabilityAtom(rawAtom));
        }
    }

    template<typename T>
    void encodeValue(std::optional<T> const& value)
    {
        if (value)
            encodeValue(*value);
        else
            encoder->encode(nullptr);
    }

    void encodeValue(SyntaxClass<NodeBase> const& value) { encode(value.getTag()); }

    template<typename T>
    void encodeValue(DeclRef<T> const& value)
    {
        encode((DeclRefBase*)value);
    }

    void encodeValue(ValNodeOperand value)
    {
        Encoder::WithKeyValuePair withPair(encoder);

        encodeEnum(value.kind);
        switch (value.kind)
        {
        case ValNodeOperandKind::ConstantValue:
            encode(value.values.intOperand);
            break;

        case ValNodeOperandKind::ValNode:
            encode(static_cast<Val*>(value.values.nodeOperand));
            break;

        case ValNodeOperandKind::ASTNode:
            {
                if (auto decl = as<Decl>(value.values.nodeOperand))
                {
                    encode(decl);
                }
                else
                {
                    SLANG_UNEXPECTED("AST node operand of `Val` was expected to be a `Decl`");
                }
            }
            break;
        }
    }

    void encodeValue(TypeExp value) { encode(value.type); }

    void encodeValue(QualType value)
    {
        Encoder::WithObject withObject(encoder);
        encode(value.type);
        encode(value.isLeftValue);
        encode(value.hasReadOnlyOnTarget);
        encode(value.isWriteOnly);
    }

    void encodeValue(MatrixCoord value)
    {
        Encoder::WithObject withObject(encoder);
        encode(value.row);
        encode(value.col);
    }

    void encodeValue(SPIRVAsmOperand::Flavor const& value) { encodeEnum(value); }

    void encodeValue(SPIRVAsmOperand const& value)
    {
        Encoder::WithObject withObject(encoder);
        encode(value.flavor);
        encode(value.token);
        encode(value.expr);
        encode(value.bitwiseOrWith);
        encode(value.knownValue);
        encode(value.wrapInId);
        encode(value.type);
    }

    void encodeValue(SPIRVAsmInst const& value)
    {
        Encoder::WithObject withObject(encoder);
        encode(value.opcode);
        encode(value.operands);
    }


    template<typename T, typename = std::enable_if_t<std::is_same_v<T, bool>>>
    void encodeValue(T value)
    {
        encoder->encodeBool(value);
    }

    void encodeValue(Int32 value) { encoder->encode(value); }
    void encodeValue(UInt32 value) { encoder->encode(value); }
    void encodeValue(Int64 value) { encoder->encode(value); }
    void encodeValue(UInt64 value) { encoder->encode(value); }
    void encodeValue(float value) { encoder->encode(value); }
    void encodeValue(double value) { encoder->encode(value); }

    void encodeValue(uint8_t value) { encoder->encode(UInt32(value)); }

    void encodeValue(nullptr_t) { encoder->encode(nullptr); }

    template<typename T>
    void encodeEnum(T value)
    {
        encoder->encode(Int32(value));
    }

    void encodeValue(DeclVisibility value) { encodeEnum(value); }
    void encodeValue(BaseType value) { encodeEnum(value); }
    void encodeValue(BuiltinRequirementKind value) { encodeEnum(value); }
    void encodeValue(ASTNodeType value) { encodeEnum(value); }
    void encodeValue(ImageFormat value) { encodeEnum(value); }
    void encodeValue(TypeTag value) { encodeEnum(value); }
    void encodeValue(TryClauseType value) { encodeEnum(value); }
    void encodeValue(CapabilityAtom value) { encodeEnum(value); }
    void encodeValue(DeclAssociationKind value) { encodeEnum(value); }
    void encodeValue(TokenType value) { encodeEnum(value); }

    void encodeValue(SourceLoc value)
    {
        if (!_sourceLocWriter)
        {
            encoder->encode(nullptr);
        }
        else
        {
            auto intermediate = _sourceLocWriter->addSourceLoc(value);
            encoder->encode(intermediate);
        }
    }

    template<typename T>
    void encodeValue(T const* ptr)
    {
        if (!ptr)
        {
            encoder->encode(nullptr);
        }
        else
        {
            encodePtr(const_cast<T*>(ptr));
        }
    }

    template<typename T>
    void encodeValue(RefPtr<T> const& ptr)
    {
        if (!ptr)
        {
            encoder->encode(nullptr);
        }
        else
        {
            encodePtr(ptr.Ptr());
        }
    }

    void encodeValue(Modifiers const& modifiers)
    {
        Encoder::WithArray withArray(encoder);
        for (auto m : const_cast<Modifiers&>(modifiers))
        {
            encode(m);
        }
    }

    template<typename T, int N>
    void encodeValue(ShortList<T, N> const& array)
    {
        Encoder::WithArray withArray(encoder);
        for (auto element : array)
        {
            encode(element);
        }
    }


    template<typename T>
    void encode(List<T> const& array)
    {
        Encoder::WithArray withArray(encoder);
        for (auto element : array)
        {
            encode(element);
        }
    }

    template<typename T, size_t N>
    void encode(T const (&array)[N])
    {
        Encoder::WithArray withArray(encoder);
        for (auto element : array)
        {
            encode(element);
        }
    }

    template<typename K, typename V>
    void encode(OrderedDictionary<K, V> const& dictionary)
    {
        Encoder::WithArray withArray(encoder);
        for (auto p : dictionary)
        {
            Encoder::WithKeyValuePair withPair(encoder);
            encode(p.key);
            encode(p.value);
        }
    }

    template<typename K, typename V>
    void encode(Dictionary<K, V> const& dictionary)
    {
        Encoder::WithArray withArray(encoder);
        for (auto p : dictionary)
        {
            Encoder::WithKeyValuePair withPair(encoder);
            encode(p.first);
            encode(p.second);
        }
    }

    template<typename T>
    void encode(T const& value)
    {
        encodeValue(value);
    }

    // for each class of node, we generate
    // code to recursively serialize each
    // of its fields.

#if 0 // FIDDLE TEMPLATE:
%for _,T in ipairs(Slang.NodeBase.subclasses) do
    void _encodeDataOf($T* obj)
    {
%if T.directSuperClass then
        _encodeDataOf(static_cast<$(T.directSuperClass)*>(obj));
%end
%for _,f in ipairs(T.directFields) do
        encode(obj->$f);
%end
    }
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 0
#include "slang-serialize-ast.cpp.fiddle"
#endif // FIDDLE END
};

void writeSerializedModuleAST(
    Encoder* encoder,
    ModuleDecl* moduleDecl,
    SerialSourceLocWriter* sourceLocWriter)
{
    Encoder::WithObject withObject(encoder);

    // TODO: we should have a more careful pass here,
    // where we only encode the public declarations
    //

    ASTEncodingContext context(encoder, moduleDecl, sourceLocWriter);
    context.getDeclID(moduleDecl);
    context.flush();
}

struct ASTDecodingContext
{
public:
    ASTDecodingContext(
        Linkage* linkage,
        ASTBuilder* astBuilder,
        DiagnosticSink* sink,
        RiffContainer::Chunk* rootChunk,
        SerialSourceLocReader* sourceLocReader,
        SourceLoc requestingSourceLoc)
        : _linkage(linkage)
        , _astBuilder(astBuilder)
        , _sink(sink)
        , _rootChunk(static_cast<RiffContainer::ListChunk*>(rootChunk))
        , _sourceLocReader(sourceLocReader)
        , _requestingSourceLoc(requestingSourceLoc)
    {
    }

    Linkage* _linkage = nullptr;
    DiagnosticSink* _sink = nullptr;
    SerialSourceLocReader* _sourceLocReader = nullptr;
    SourceLoc _requestingSourceLoc;

    SlangResult decodeAll()
    {
        auto cursor = _rootChunk->getFirstContainedChunk();

        // There are a few different top-level chunks that
        // hold different arrays that we need in order
        // to decode the entire module hierarchy.
        //
        // Basically, these lists correspond to the kinds
        // of nodes in the AST hierarchy for which back-references
        // are allowed (all other nodes should, barring
        // weird corner cases, form a single tree-structured
        // ownership hierarchy, rooted at the `ModuleDecl`.
        //

        // First there is the list that actually encodes
        // for the declarations in the module, including
        // the `ModuleDecl` itself, which should be the
        // first entry in the list.
        //
        auto declChunk = cursor;
        cursor = cursor->m_next;

        // Next there is a list of all the declarations
        // referenced inside of the module that need to
        // be imported in from outside.
        //
        auto importedDeclChunk = cursor;
        cursor = cursor->m_next;

        // Then there are all the `Val`-derived nodes that
        // are needed by the module, which will need to be
        // deduplicated so that they are unique within the
        // current compilation context.
        //
        auto valChunk = cursor;
        cursor = cursor->m_next;

        // The process of decoding the module is then spread
        // over a number of steps.
        //
        // The first step is to process all of the imported
        // declarations, so that other nodes can refer to
        // them.
        //
        SLANG_RETURN_ON_FAIL(decodeImportedDecls(importedDeclChunk));

        // Next we process the declarations that are within
        // the module itself, first creating an "empty shell"
        // of each declaration that has the right size in
        // memory (and the right `ASTNodeType` tag), so that
        // we can wire up references to it (including circular
        // references)... so long as nothing here tries to
        // look *inside* the empty shell along the way.
        //
        SLANG_RETURN_ON_FAIL(createEmptyShells(declChunk));

        // Once all the `Decl`s that might be needed have
        // been allocated, we can process all the `Val`s
        // that might reference those`Decl`s (and one another).
        //
        // The nature of the `Val` representation ensures
        // that there cannot be cirularities in the references
        // between `Val`s, and the encoding process will have
        // sorted the entries so that a `Val` only ever appears
        // *after* its operands.
        //
        SLANG_RETURN_ON_FAIL(decodeVals(valChunk));

        // Once all the back-reference-able objects have been
        // instantiated in memory, we can go back through the
        // `Decl`s in the module and fill in those empty shells.
        //
        SLANG_RETURN_ON_FAIL(fillEmptyShells(declChunk));

        // As a final pass,  we perform any special cleanup actions
        // that might be required to make the output valid for consumers.
        //
        // For example, this is where we set the `DeclCheckState` of everything
        // we are loading to reflect the fact that everything we deserialize
        // is (supposed to be) fully cheked.
        //
        SLANG_RETURN_ON_FAIL(cleanUpNodes());


        return SLANG_OK;
    }

    typedef Int DeclID;
    Decl* getDeclByID(DeclID id)
    {
        if (id >= 0)
        {
            return _decls[id];
        }
        else
        {
            return _importedDecls[~id];
        }
    }

private:
    struct UnhandledCase
    {
    };

    ASTBuilder* _astBuilder = nullptr;
    RiffContainer::ListChunk* _rootChunk = nullptr;

    List<Decl*> _decls;
    List<Decl*> _importedDecls;
    List<Val*> _vals;

    typedef Int ValID;
    Val* getValByID(ValID id) { return _vals[id]; }

    SlangResult decodeImportedDecls(RiffContainer::Chunk* importedDeclChunk)
    {
        Decoder decoder(importedDeclChunk);

        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            Decoder::WithKeyValuePair withPair(decoder);

            Int moduleIndex;
            decode(moduleIndex, decoder);

            if (moduleIndex == -1)
            {
                Name* moduleName = nullptr;
                decode(moduleName, decoder);

                Decl* importedModule = getImportedModule(moduleName);
                _importedDecls.add(importedModule);
            }
            else
            {
                auto importedFromModuleDecl = as<ModuleDecl>(_importedDecls[moduleIndex]);
                auto importedFromModule = importedFromModuleDecl->module;

                String mangledName;
                decode(mangledName, decoder);

                auto importedNode =
                    importedFromModule->findExportFromMangledName(mangledName.getUnownedSlice());
                auto importedDecl = as<Decl>(importedNode);
                _importedDecls.add(importedDecl);
            }
        }
        return SLANG_OK;
    }

    ModuleDecl* getImportedModule(Name* moduleName)
    {
        Module* module = _linkage->findOrImportModule(moduleName, _requestingSourceLoc, _sink);
        if (!module)
        {
            SLANG_ABORT_COMPILATION("failed to load an imported module during deserialization");
        }

        return module->getModuleDecl();
    }

    SlangResult decodeVals(RiffContainer::Chunk* valChunk)
    {
        Decoder decoder(valChunk);

        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            Val* val = decodeValNode(decoder);
            _vals.add(val);
        }
        return SLANG_OK;
    }

    SlangResult createEmptyShells(RiffContainer::Chunk* declChunk)
    {
        Decoder decoder(declChunk);

        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            ASTNodeType nodeType;

            // Each of the declarations is expected to take
            // the form of an object with a first field
            // that holds the node type.
            //
            {
                Decoder::WithObject withObject(decoder);
                decode(nodeType, decoder);
            }

            auto emptyShell = createEmptyShell(nodeType);
            auto declEmptyShell = as<Decl>(emptyShell);
            _decls.add(declEmptyShell);
        }

        return SLANG_OK;
    }

    Val* decodeValNode(Decoder& decoder)
    {
        Decoder::WithObject withObject(decoder);

        ASTNodeType nodeType;
        decode(nodeType, decoder);

        ValNodeDesc desc;
        desc.type = SyntaxClass<NodeBase>(nodeType);

        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            ValNodeOperand operand;
            decode(operand, decoder);
            desc.operands.add(operand);
        }

        desc.init();

        auto val = _astBuilder->_getOrCreateImpl(_Move(desc));

        // Values created during deserialization are
        // not expected to ever resolve further, because
        // they should be coming from fully checked code.
        //
        // val->resolve();
        // val->_setUnique();

        return val;
    }

    NodeBase* createEmptyShell(ASTNodeType nodeType)
    {
        return SyntaxClass<NodeBase>(nodeType).createInstance(_astBuilder);
    }

    SlangResult fillEmptyShells(RiffContainer::Chunk* declChunk)
    {
        Index declIndex = 0;

        Decoder decoder(declChunk);
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            auto declEmptyShell = _decls[declIndex++];
            decodeASTNodeContent(declEmptyShell, decoder);
        }

        return SLANG_OK;
    }

    SlangResult cleanUpNodes()
    {
        for (auto decl : _decls)
        {
            decl->checkState = DeclCheckState::CapabilityChecked;
        }

        return SLANG_OK;
    }


    void assignGenericParameterIndices(GenericDecl* genericDecl)
    {
        int parameterCounter = 0;
        for (auto m : genericDecl->members)
        {
            if (auto typeParam = as<GenericTypeParamDeclBase>(m))
            {
                typeParam->parameterIndex = parameterCounter++;
            }
            else if (auto valParam = as<GenericValueParamDecl>(m))
            {
                valParam->parameterIndex = parameterCounter++;
            }
        }
    }


    void cleanUpASTNode(NodeBase* node)
    {
        if (auto expr = as<Expr>(node))
        {
            expr->checked = true;
        }
        else if (auto genericDecl = as<GenericDecl>(node))
        {
            assignGenericParameterIndices(genericDecl);
        }
        else if (auto syntaxDecl = as<SyntaxDecl>(node))
        {
            syntaxDecl->parseCallback = &parseSimpleSyntax;
            syntaxDecl->parseUserData = (void*)syntaxDecl->syntaxClass.getInfo();
        }
        else if (auto namespaceLikeDecl = as<NamespaceDeclBase>(node))
        {
            auto declScope = _astBuilder->create<Scope>();
            declScope->containerDecl = namespaceLikeDecl;
            namespaceLikeDecl->ownedScope = declScope;
        }
    }

    void decodeASTNodeContent(NodeBase* node, Decoder& decoder)
    {
        Decoder::WithObject withObject(decoder);

        ASTNodeDispatcher<NodeBase, void>::dispatch(
            node,
            [&](auto n) { _decodeDataOf(n, decoder); });

        cleanUpASTNode(node);
    }

    DeclID decodeDeclID(Decoder& decoder)
    {
        DeclID result = decoder.decode<DeclID>();
        return result;
    }

    ValID decodeValID(Decoder& decoder)
    {
        ValID result = decoder.decode<ValID>();
        return result;
    }

    template<typename T>
    void decodeASTNode(T*& node, Decoder& decoder)
    {
        ASTNodeType nodeType;
        auto saved = decoder.getCursor();
        {
            Decoder::WithObject withObject(decoder);
            decode(nodeType, decoder);
        }
        decoder.setCursor(saved);

        auto shell = createEmptyShell(nodeType);
        decodeASTNodeContent(shell, decoder);

        node = as<T>(shell);
    }

    void decodePtr(Name*& name, Decoder& decoder, Name*)
    {
        String text;
        decode(text, decoder);

        name = _astBuilder->getNamePool()->getName(text);
    }

    void decodePtr(DeclAssociationList*& outList, Decoder& decoder, DeclAssociationList*)
    {
        // Mirroring the encoding logic, we decode this
        // as a list of key-value pairs.
        //
        auto list = RefPtr(new DeclAssociationList());
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            auto association = RefPtr(new DeclAssociation());

            Decoder::WithKeyValuePair withPair(decoder);
            decode(association->kind, decoder);
            decode(association->decl, decoder);

            list->associations.add(association);
        }

        outList = list.detach();
    }

    void decodePtr(DiagnosticInfo const*& info, Decoder& decoder, DiagnosticInfo const*)
    {
        Int id;
        decode(id, decoder);
        info = getDiagnosticsLookup()->getDiagnosticById(id);
    }

    void decodePtr(MarkupEntry*& markupEntry, Decoder&, MarkupEntry*)
    {
        // TODO: is this case needed?
        markupEntry = nullptr;
    }

    void decodePtr(CandidateExtensionList*& list, Decoder& decoder, CandidateExtensionList*)
    {
        auto result = RefPtr(new CandidateExtensionList());
        decode(result->candidateExtensions, decoder);
        list = result.detach();
    }

    void decodePtr(WitnessTable*& witnessTable, Decoder& decoder, WitnessTable*)
    {
        Decoder::WithObject withObject(decoder);
        auto wt = RefPtr(new WitnessTable());
        decode(wt->baseType, decoder);
        decode(wt->witnessedType, decoder);
        decode(wt->isExtern, decoder);
        decode(wt->m_requirementDictionary, decoder);
        witnessTable = wt.detach();
    }

    void decodeValue(RequirementWitness& witness, Decoder& decoder)
    {
        Decoder::WithKeyValuePair withPair(decoder);
        decodeEnum(witness.m_flavor, decoder);
        switch (witness.m_flavor)
        {
        case RequirementWitness::Flavor::none:
            break;

        case RequirementWitness::Flavor::declRef:
            decode(witness.m_declRef, decoder);
            break;

        case RequirementWitness::Flavor::val:
            decode(witness.m_val, decoder);
            break;

        case RequirementWitness::Flavor::witnessTable:
            {
                RefPtr<WitnessTable> object;
                decode(object, decoder);
                witness.m_obj = object;
            }
            break;
        }
    }

    template<typename T>
    void decodePtr(T*& node, Decoder& decoder, Val*)
    {
        ValID id = decodeValID(decoder);
        node = static_cast<T*>(getValByID(id));
    }

    template<typename T>
    void decodePtr(T*& node, Decoder& decoder, Decl*)
    {
        DeclID id = decodeDeclID(decoder);
        node = static_cast<T*>(getDeclByID(id));
    }

    template<typename T>
    void decodePtr(T*& node, Decoder& decoder, DeclBase*)
    {
        // This case is a bit of a hack. We need
        // to identify whether we are looking at
        // an indirection to a `Decl` (which would
        // be serialized as an integer `DeclID`),
        // or something else derived from `DeclBase`.
        //
        switch (decoder.getTag())
        {
        default:
            decodeASTNode(node, decoder);
            break;

        case SerialBinary::kInt32FourCC:
        case SerialBinary::kInt64FourCC:
        case SerialBinary::kUInt32FourCC:
        case SerialBinary::kUInt64FourCC:
            {
                DeclID id = decodeDeclID(decoder);
                node = static_cast<T*>(getDeclByID(id));
            }
            break;
        }
    }

    template<typename T>
    void decodePtr(T*& node, Decoder& decoder, NodeBase*)
    {
        decodeASTNode(node, decoder);
    }


    void decodeValue(UnhandledCase, Decoder& decoder);

    void decodeValue(String& value, Decoder& decoder) { value = decoder.decodeString(); }

    void decodeValue(Token& value, Decoder& decoder)
    {
        decode(value.type, decoder);
        decode(value.flags, decoder);
        decode(value.loc, decoder);
        if (decoder.decodeNull())
        {
        }
        else
        {
            Name* name = nullptr;
            decode(name, decoder);
            value.setName(name);
        }
    }

    void decodeValue(NameLoc& value, Decoder& decoder) { decode(value.name, decoder); }

    void decodeValue(SemanticVersion& value, Decoder& decoder)
    {
        SemanticVersion::IntegerType rawValue = decoder.decode<SemanticVersion::IntegerType>();
        value.setFromInteger(rawValue);
    }

    void decodeValue(CapabilitySet& value, Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            CapabilityTargetSet targetSet;
            decode(targetSet, decoder);
            value.getCapabilityTargetSets()[targetSet.target] = targetSet;
        }
    }

    void decodeValue(CapabilityTargetSet& value, Decoder& decoder)
    {
        Decoder::WithKeyValuePair withPair(decoder);
        decode(value.target, decoder);

        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            CapabilityStageSet stageSet;
            decode(stageSet, decoder);
            value.shaderStageSets[stageSet.stage] = stageSet;
        }
    }

    void decodeValue(CapabilityStageSet& value, Decoder& decoder)
    {
        Decoder::WithKeyValuePair withPair(decoder);
        decode(value.stage, decoder);
        decode(value.atomSet, decoder);
    }

    void decodeValue(CapabilityAtomSet& value, Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            CapabilityAtom atom;
            decode(atom, decoder);
            value.add(UInt(atom));
        }
    }

    template<typename T>
    void decodeValue(std::optional<T>& outValue, Decoder& decoder)
    {
        if (decoder.decodeNull())
        {
            outValue.reset();
        }
        else
        {
            T value;
            decode(value, decoder);
            outValue = value;
        }
    }

    void decodeValue(SyntaxClass<NodeBase>& syntaxClass, Decoder& decoder)
    {
        ASTNodeType nodeType;
        decode(nodeType, decoder);
        syntaxClass = SyntaxClass<NodeBase>(nodeType);
    }

    template<typename T>
    void decodeValue(DeclRef<T>& declRef, Decoder& decoder)
    {
        decode(declRef.declRefBase, decoder);
    }

    void decodeValue(ValNodeOperand& value, Decoder& decoder)
    {
        Decoder::WithKeyValuePair withPair(decoder);

        decodeEnum(value.kind, decoder);
        switch (value.kind)
        {
        case ValNodeOperandKind::ConstantValue:
            decode(value.values.intOperand, decoder);
            break;

        case ValNodeOperandKind::ValNode:
            {
                Val* val = nullptr;
                decode(val, decoder);
                value.values.nodeOperand = val;
            }
            break;

        case ValNodeOperandKind::ASTNode:
            {
                Decl* decl = nullptr;
                decode(decl, decoder);
                value.values.nodeOperand = decl;
            }
            break;
        }
    }

    void decodeValue(TypeExp& value, Decoder& decoder) { decode(value.type, decoder); }

    void decodeValue(QualType& value, Decoder& decoder)
    {
        Decoder::WithObject withObject(decoder);
        decode(value.type, decoder);
        decode(value.isLeftValue, decoder);
        decode(value.hasReadOnlyOnTarget, decoder);
        decode(value.isWriteOnly, decoder);
    }

    void decodeValue(MatrixCoord& value, Decoder& decoder)
    {
        Decoder::WithObject withObject(decoder);
        decode(value.row, decoder);
        decode(value.col, decoder);
    }

    void decodeValue(SPIRVAsmOperand::Flavor& value, Decoder& decoder)
    {
        decodeEnum(value, decoder);
    }

    void decodeValue(SPIRVAsmOperand& value, Decoder& decoder)
    {
        Decoder::WithObject withObject(decoder);
        decode(value.flavor, decoder);
        decode(value.token, decoder);
        decode(value.expr, decoder);
        decode(value.bitwiseOrWith, decoder);
        decode(value.knownValue, decoder);
        decode(value.wrapInId, decoder);
        decode(value.type, decoder);
    }

    void decodeValue(SPIRVAsmInst& value, Decoder& decoder)
    {
        Decoder::WithObject withObject(decoder);
        decode(value.opcode, decoder);
        decode(value.operands, decoder);
    }


    template<typename T>
    void decodeEnum(T& value, Decoder& decoder)
    {
        value = T(decoder.decode<Int32>());
    }

    template<typename T>
    void decodeSimpleValue(T& value, Decoder& decoder)
    {
        value = decoder.decode<T>();
    }

    void decodeValue(bool& value, Decoder& decoder) { value = decoder.decodeBool(); }
    void decodeValue(Int32& value, Decoder& decoder) { decodeSimpleValue(value, decoder); }
    void decodeValue(Int64& value, Decoder& decoder) { decodeSimpleValue(value, decoder); }
    void decodeValue(UInt32& value, Decoder& decoder) { decodeSimpleValue(value, decoder); }
    void decodeValue(UInt64& value, Decoder& decoder) { decodeSimpleValue(value, decoder); }
    void decodeValue(float& value, Decoder& decoder) { decodeSimpleValue(value, decoder); }
    void decodeValue(double& value, Decoder& decoder) { decodeSimpleValue(value, decoder); }

    void decodeValue(uint8_t& value, Decoder& decoder)
    {
        value = uint8_t(decoder.decode<UInt32>());
    }

    void decodeValue(DeclVisibility& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(BaseType& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(BuiltinRequirementKind& value, Decoder& decoder)
    {
        decodeEnum(value, decoder);
    }
    void decodeValue(ASTNodeType& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(ImageFormat& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(TypeTag& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(TryClauseType& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(CapabilityAtom& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(PreferRecomputeAttribute::SideEffectBehavior& value, Decoder& decoder)
    {
        decodeEnum(value, decoder);
    }
    void decodeValue(LogicOperatorShortCircuitExpr::Flavor& value, Decoder& decoder)
    {
        decodeEnum(value, decoder);
    }
    void decodeValue(TreatAsDifferentiableExpr::Flavor& value, Decoder& decoder)
    {
        decodeEnum(value, decoder);
    }
    void decodeValue(DeclAssociationKind& value, Decoder& decoder) { decodeEnum(value, decoder); }
    void decodeValue(TokenType& value, Decoder& decoder) { decodeEnum(value, decoder); }


    void decodeValue(SourceLoc& value, Decoder& decoder)
    {
        if (!decoder.decodeNull())
        {
            SerialSourceLocData::SourceLoc intermediate;
            decoder.decode(intermediate);

            if (_sourceLocReader)
            {
                auto sourceLoc = _sourceLocReader->getSourceLoc(intermediate);
                value = sourceLoc;
            }
        }
    }

    template<typename T>
    void decodeValue(T*& ptr, Decoder& decoder)
    {
        if (decoder.decodeNull())
            ptr = nullptr;
        else
            decodePtr(ptr, decoder, (T*)nullptr);
    }

    template<typename T>
    void decodeValue(RefPtr<T>& ptr, Decoder& decoder)
    {
        if (decoder.decodeNull())
            ptr = nullptr;
        else
        {
            // Hi Future Tess,
            //
            // The next step here is decoding logic for `WitnessTable`s.
            //

            decodePtr(*ptr.writeRef(), decoder, (T*)nullptr);
        }
    }

    void decodeValue(Modifiers& modifiers, Decoder& decoder)
    {
        Modifier** link = &modifiers.first;

        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            Modifier* modifier = nullptr;
            decode(modifier, decoder);

            *link = modifier;
            link = &modifier->next;
        }
    }

    template<typename T, int N>
    void decodeValue(ShortList<T, N>& array, Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            T element;
            decode(element, decoder);
            array.add(element);
        }
    }


    template<typename T>
    void decode(List<T>& array, Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            T element;
            decode(element, decoder);
            array.add(element);
        }
    }

    template<typename T, size_t N>
    void decode(T (&array)[N], Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        for (auto& element : array)
        {
            decode(element, decoder);
        }
    }

    template<typename K, typename V>
    void decode(OrderedDictionary<K, V>& dictionary, Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            Decoder::WithKeyValuePair withPair(decoder);

            K key;
            V value;
            decode(key, decoder);
            decode(value, decoder);

            dictionary.add(key, value);
        }
    }

    template<typename K, typename V>
    void decode(Dictionary<K, V>& dictionary, Decoder& decoder)
    {
        Decoder::WithArray withArray(decoder);
        while (decoder.hasElements())
        {
            Decoder::WithKeyValuePair withPair(decoder);

            K key;
            V value;
            decode(key, decoder);
            decode(value, decoder);

            dictionary.add(key, value);
        }
    }

    template<typename T>
    void decode(T& outValue, Decoder& decoder)
    {
        decodeValue(outValue, decoder);
    }

#if 0 // FIDDLE TEMPLATE:
%for _,T in ipairs(Slang.NodeBase.subclasses) do
        void _decodeDataOf($T* obj, Decoder& decoder)
        {
%   if T.directSuperClass then
            _decodeDataOf(static_cast<$(T.directSuperClass)*>(obj), decoder);
%   end
%   for _,f in ipairs(T.directFields) do
            decode(obj->$f, decoder);
%   end
        }
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 1
#include "slang-serialize-ast.cpp.fiddle"
#endif // FIDDLE END
};

ModuleDecl* readSerializedModuleAST(
    Linkage* linkage,
    ASTBuilder* astBuilder,
    DiagnosticSink* sink,
    RiffContainer::Chunk* chunk,
    SerialSourceLocReader* sourceLocReader,
    SourceLoc requestingSourceLoc)
{
    ASTDecodingContext
        context(linkage, astBuilder, sink, chunk, sourceLocReader, requestingSourceLoc);
    context.decodeAll();
    auto node = context.getDeclByID(0);
    auto moduleDecl = as<ModuleDecl>(node);
    return moduleDecl;
}
} // namespace Slang
