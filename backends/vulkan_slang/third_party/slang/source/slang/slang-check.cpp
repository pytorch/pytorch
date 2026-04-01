// slang-check.cpp
#include "slang-check.h"

// This file provides general facilities related to semantic
// checking that don't cleanly land in one of the more
// specialized `slang-check-*` files.

#include "../core/slang-type-text-util.h"
#include "slang-check-impl.h"

namespace Slang
{
namespace
{ // anonymous

class SinkSharedLibraryLoader : public RefObject, public ISlangSharedLibraryLoader
{
public:
    SLANG_REF_OBJECT_IUNKNOWN_ALL

    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadSharedLibrary(const char* path, ISlangSharedLibrary** outSharedLibrary) SLANG_OVERRIDE
    {
        SlangResult res = m_loader->loadSharedLibrary(path, outSharedLibrary);

        // Special handling for failure...
        if (SLANG_FAILED(res) && m_sink)
        {
            String filename = Path::getFileNameWithoutExt(path);
            if (filename == "dxil")
            {
                m_sink->diagnose(SourceLoc(), Diagnostics::dxilNotFound);
            }
            else
            {
                m_sink->diagnose(SourceLoc(), Diagnostics::noteFailedToLoadDynamicLibrary, path);
            }
        }
        return res;
    }

    SinkSharedLibraryLoader(ISlangSharedLibraryLoader* loader, DiagnosticSink* sink)
        : m_loader(loader), m_sink(sink)
    {
    }

protected:
    ISlangUnknown* getInterface(const Guid& guid)
    {
        return (guid == ISlangUnknown::getTypeGuid() ||
                guid == ISlangSharedLibraryLoader::getTypeGuid())
                   ? static_cast<ISlangSharedLibraryLoader*>(this)
                   : nullptr;
    }
    ISlangSharedLibraryLoader* m_loader;
    DiagnosticSink* m_sink;
};

} // namespace


void Session::_setSharedLibraryLoader(ISlangSharedLibraryLoader* loader)
{
    if (m_sharedLibraryLoader != loader)
    {
        // Need to clear all of the libraries
        m_downstreamCompilerSet->clear();
        m_downstreamCompilerInitialized = 0;

        for (Index i = 0; i < Index(SLANG_PASS_THROUGH_COUNT_OF); ++i)
        {
            m_downstreamCompilers[i].setNull();
        }

        // Set the loader
        m_sharedLibraryLoader = loader;
    }
}

void Session::resetDownstreamCompiler(PassThroughMode type)
{
    // Mark as initialized
    m_downstreamCompilerInitialized &= ~(1 << int(type));
    m_downstreamCompilers[int(type)].setNull();
}

IDownstreamCompiler* Session::getOrLoadDownstreamCompiler(
    PassThroughMode type,
    DiagnosticSink* sink)
{
    if (m_downstreamCompilerInitialized & (1 << int(type)))
    {
        return m_downstreamCompilers[int(type)];
    }

    if (type == PassThroughMode::GenericCCpp)
    {
        // try testing for availability on all C/C++ compilers
        getOrLoadDownstreamCompiler(PassThroughMode::Clang, nullptr);
        getOrLoadDownstreamCompiler(PassThroughMode::Gcc, nullptr);
        getOrLoadDownstreamCompiler(PassThroughMode::VisualStudio, nullptr);
        getOrLoadDownstreamCompiler(PassThroughMode::LLVM, nullptr);
    }

    // Mark that we have tried to load it
    m_downstreamCompilerInitialized |= (1 << int(type));
    m_downstreamCompilers[int(type)].setNull();

    // Do we have a locator
    auto locator = m_downstreamCompilerLocators[int(type)];
    if (locator)
    {
        m_downstreamCompilerSet->remove(SlangPassThrough(type));

        // We want to be able to report a diagnostic to the user if a loader
        // was unable to locate the desired downstream compiler, but we
        // also need to deal with the fact that the locator might "probe"
        // multiple possible library versions/names, and failing to load
        // one library should not be taken as a hard error.
        //
        // The approach we use here is to first apply the `locator` directly
        // with our `m_sharedLibraryLoader` and see if it succeeds. If
        // it does, then we will move along.
        //
        if (SLANG_FAILED(locator(
                m_downstreamCompilerPaths[int(type)],
                m_sharedLibraryLoader,
                m_downstreamCompilerSet)))
        {
            // If the locator reported a failure the first time we invoked
            // it, then we will invoke it against with a wrapper shared library
            // loader that reported library load failures to our diagnost `sink`.
            //
            // This means that in the case of failure the user will see a listing
            // of all the libraries that the locator attempted to load but failed
            // to find. The user will know that making one or more of these libraries
            // available could fix the issue, but we cannot communicate precise
            // information to them with this approach (e.g., the difference between
            // "I need all of these libraries" vs. "I need at least one of these
            // libraries").
            //
            if (sink)
            {
                sink->diagnose(SourceLoc(), Diagnostics::failedToLoadDownstreamCompiler, type);
            }
            SinkSharedLibraryLoader loader(m_sharedLibraryLoader, sink);
            locator(m_downstreamCompilerPaths[int(type)], &loader, m_downstreamCompilerSet);
        }

        DownstreamCompilerUtil::updateDefaults(m_downstreamCompilerSet);
    }

    IDownstreamCompiler* compiler = nullptr;

    if (type == PassThroughMode::GenericCCpp)
    {
        compiler = m_downstreamCompilerSet->getDefaultCompiler(SLANG_SOURCE_LANGUAGE_CPP);
    }
    else
    {
        DownstreamCompilerDesc desc;
        desc.type = SlangPassThrough(type);
        compiler = DownstreamCompilerUtil::findCompiler(
            m_downstreamCompilerSet,
            DownstreamCompilerUtil::MatchType::Newest,
            desc);
    }
    m_downstreamCompilers[int(type)] = compiler;
    return compiler;
}

void checkTranslationUnit(
    TranslationUnitRequest* translationUnit,
    LoadedModuleDictionary& loadedModules)
{
    SLANG_AST_BUILDER_RAII(translationUnit->compileRequest->getLinkage()->getASTBuilder());

    SharedSemanticsContext sharedSemanticsContext(
        translationUnit->compileRequest->getLinkage(),
        translationUnit->getModule(),
        translationUnit->compileRequest->getSink(),
        &loadedModules,
        translationUnit);

    SemanticsDeclVisitorBase visitor((SemanticsContext(&sharedSemanticsContext)));

    // Apply the visitor to do the main semantic
    // checking that is required on all declarations
    // in the translation unit.

    visitor.checkModule(translationUnit->getModuleDecl());

    translationUnit->getModule()->_collectShaderParams();
}

void SemanticsVisitor::dispatchStmt(Stmt* stmt, SemanticsContext const& context)
{
    SemanticsStmtVisitor visitor(context);
    try
    {
        visitor.dispatch(stmt);
    }
    catch (const AbortCompilationException&)
    {
        throw;
    }
    catch (...)
    {
        getSink()->noteInternalErrorLoc(stmt->loc);
        throw;
    }
}

Expr* SemanticsVisitor::dispatchExpr(Expr* expr, SemanticsContext const& context)
{
    SemanticsExprVisitor visitor(context);
    try
    {
        return visitor.dispatch(expr);
    }
    catch (const AbortCompilationException&)
    {
        throw;
    }
    catch (...)
    {
        getSink()->noteInternalErrorLoc(expr->loc);
        throw;
    }
}

ASTBuilder* semanticsVisitorGetASTBuilder(SemanticsVisitor* sv)
{
    return sv->getASTBuilder();
}

} // namespace Slang
