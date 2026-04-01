// slang-downstream-compiler-set.cpp
#include "slang-downstream-compiler-set.h"

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DownstreamCompilerSet !!!!!!!!!!!!!!!!!!!!!!*/

void DownstreamCompilerSet::getCompilerDescs(List<DownstreamCompilerDesc>& outCompilerDescs) const
{
    outCompilerDescs.clear();
    for (IDownstreamCompiler* compiler : m_compilers)
    {
        outCompilerDescs.add(compiler->getDesc());
    }
}

Index DownstreamCompilerSet::_findIndex(const DownstreamCompilerDesc& desc) const
{
    const Index count = m_compilers.getCount();
    for (Index i = 0; i < count; ++i)
    {
        if (m_compilers[i]->getDesc() == desc)
        {
            return i;
        }
    }
    return -1;
}

IDownstreamCompiler* DownstreamCompilerSet::getCompiler(
    const DownstreamCompilerDesc& compilerDesc) const
{
    const Index index = _findIndex(compilerDesc);
    return index >= 0 ? m_compilers[index] : nullptr;
}

void DownstreamCompilerSet::getCompilers(List<IDownstreamCompiler*>& outCompilers) const
{
    outCompilers.clear();
    outCompilers.addRange((IDownstreamCompiler* const*)m_compilers.begin(), m_compilers.getCount());
}

bool DownstreamCompilerSet::hasSharedLibrary(ISlangSharedLibrary* lib)
{
    const Index foundIndex = m_sharedLibraries.findFirstIndex(
        [lib](ISlangSharedLibrary* inLib) -> bool { return lib == inLib; });
    return (foundIndex >= 0);
}

void DownstreamCompilerSet::addSharedLibrary(ISlangSharedLibrary* lib)
{
    SLANG_ASSERT(lib);
    if (!hasSharedLibrary(lib))
    {
        m_sharedLibraries.add(ComPtr<ISlangSharedLibrary>(lib));
    }
}

bool DownstreamCompilerSet::hasCompiler(SlangPassThrough compilerType) const
{
    for (IDownstreamCompiler* compiler : m_compilers)
    {
        const auto& desc = compiler->getDesc();
        if (desc.type == compilerType)
        {
            return true;
        }
    }
    return false;
}

void DownstreamCompilerSet::remove(SlangPassThrough compilerType)
{
    for (Index i = 0; i < m_compilers.getCount(); ++i)
    {
        IDownstreamCompiler* compiler = m_compilers[i];
        if (compiler->getDesc().type == compilerType)
        {
            m_compilers.fastRemoveAt(i);
            i--;
        }
    }
}

void DownstreamCompilerSet::addCompiler(IDownstreamCompiler* compiler)
{
    const Index index = _findIndex(compiler->getDesc());
    if (index >= 0)
    {
        m_compilers[index] = compiler;
    }
    else
    {
        m_compilers.add(ComPtr<IDownstreamCompiler>(compiler));
    }
}

} // namespace Slang
