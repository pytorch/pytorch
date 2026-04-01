#pragma once

#include "core/slang-string-util.h"
#include "slang-com-ptr.h"
#include "slang.h"

using Slang::ComPtr;

class TestBase
{

public:
    // Parses command line options. This example only has one option for testing purpose.
    int parseOption(int argc, char** argv);

    void printEntrypointHashes(
        int entryPointCount,
        int targetCount,
        ComPtr<slang::IComponentType>& composedProgram);

    bool isTestMode() const { return m_isTestMode; }

private:
    bool m_isTestMode = false;
    uint64_t m_globalCounter = 0;
};
