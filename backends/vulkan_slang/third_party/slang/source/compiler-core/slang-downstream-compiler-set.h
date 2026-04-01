#ifndef SLANG_DOWNSTREAM_COMPILER_SET_H
#define SLANG_DOWNSTREAM_COMPILER_SET_H

#include "slang-downstream-compiler.h"

namespace Slang
{

class DownstreamCompilerSet : public RefObject
{
public:
    typedef RefObject Super;

    /// Find all the available compilers
    void getCompilerDescs(List<IDownstreamCompiler::Desc>& outCompilerDescs) const;
    /// Returns list of all compilers
    void getCompilers(List<IDownstreamCompiler*>& outCompilers) const;

    /// Get a compiler
    IDownstreamCompiler* getCompiler(const DownstreamCompilerDesc& compilerDesc) const;

    /// Will replace if there is one with same desc
    void addCompiler(IDownstreamCompiler* compiler);

    /// Get a default compiler
    IDownstreamCompiler* getDefaultCompiler(SlangSourceLanguage sourceLanguage) const
    {
        return m_defaultCompilers[int(sourceLanguage)];
    }
    /// Set the default compiler
    void setDefaultCompiler(SlangSourceLanguage sourceLanguage, IDownstreamCompiler* compiler)
    {
        m_defaultCompilers[int(sourceLanguage)] = compiler;
    }

    /// True if has a compiler of the specified type
    bool hasCompiler(SlangPassThrough compilerType) const;

    void remove(SlangPassThrough compilerType);

    void clear() { m_compilers.clear(); }

    bool hasSharedLibrary(ISlangSharedLibrary* lib);
    void addSharedLibrary(ISlangSharedLibrary* lib);

    ~DownstreamCompilerSet()
    {
        // A compiler may be implemented in a shared library, so release all first.
        m_compilers.clearAndDeallocate();
        for (auto& defaultCompiler : m_defaultCompilers)
        {
            defaultCompiler.setNull();
        }

        // Release any shared libraries
        m_sharedLibraries.clearAndDeallocate();
    }

protected:
    Index _findIndex(const DownstreamCompilerDesc& desc) const;


    ComPtr<IDownstreamCompiler> m_defaultCompilers[int(SLANG_SOURCE_LANGUAGE_COUNT_OF)];
    // This could be a dictionary/map - but doing a linear search is going to be fine and it makes
    // somethings easier.
    List<ComPtr<IDownstreamCompiler>> m_compilers;

    List<ComPtr<ISlangSharedLibrary>> m_sharedLibraries;
};

} // namespace Slang

#endif
