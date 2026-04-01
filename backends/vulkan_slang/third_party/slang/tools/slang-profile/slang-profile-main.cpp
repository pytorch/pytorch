// slang-profile-main.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-std-writers.h"
#include "../../source/core/slang-string-util.h"
#include "slang-com-helper.h"

using namespace Slang;

SlangResult innerMain(int argc, char** argv)
{
    auto stdWriters = StdWriters::initDefaultSingleton();

    // Time the creation of the session
    {
        const auto startTick = Process::getClockTick();

        for (Int i = 0; i < 32; ++i)
        {
            ComPtr<slang::IGlobalSession> slangSession;
            slangSession.attach(spCreateSession(nullptr));
        }

        const auto endTick = Process::getClockTick();

        printf("Ticks %f\n", double(endTick - startTick) / Process::getClockFrequency());
        return SLANG_OK;
    }

    return SLANG_OK;
}

int main(int argc, char** argv)
{
    const SlangResult res = innerMain(argc, argv);
#ifdef _MSC_VER
    _CrtDumpMemoryLeaks();
#endif
    return SLANG_SUCCEEDED(res) ? 0 : 1;
}
