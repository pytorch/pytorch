
#include "slang-std-writers.h"

#if SLANG_WINDOWS_FAMILY
#include <windows.h>
#endif

namespace Slang
{

/* static */ StdWriters* StdWriters::s_singleton = nullptr;

/* static */ RefPtr<StdWriters> StdWriters::createDefault()
{
#if SLANG_WINDOWS_FAMILY
    SetConsoleCP(CP_UTF8);
    SetConsoleOutputCP(CP_UTF8);
#endif
    RefPtr<StdWriters> stdWriters(new StdWriters);
    RefPtr<FileWriter> stdError(
        new FileWriter(stderr, WriterFlag::AutoFlush | WriterFlag::IsUnowned));
    RefPtr<FileWriter> stdOut(
        new FileWriter(stdout, WriterFlag::AutoFlush | WriterFlag::IsUnowned));

    stdWriters->setWriter(SLANG_WRITER_CHANNEL_STD_ERROR, stdError);
    stdWriters->setWriter(SLANG_WRITER_CHANNEL_STD_OUTPUT, stdOut);

    return stdWriters;
}

/* static */ RefPtr<StdWriters> StdWriters::initDefaultSingleton()
{
    if (s_singleton)
    {
        return s_singleton;
    }
    auto defaults = createDefault();
    setSingleton(defaults);
    return defaults;
}

} // namespace Slang
