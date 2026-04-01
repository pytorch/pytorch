#include "test-base.h"

#ifdef _WIN32
// clang-format off
// include ordering sensitive
#    include <windows.h>
#    include <shellapi.h>
// clang-format on
#endif

int TestBase::parseOption(int argc, char** argv)
{
    // We only make the parse in a very loose way for only extracting the test option.
#ifdef _WIN32
    wchar_t** szArglist;
    szArglist = CommandLineToArgvW(GetCommandLineW(), &argc);
#endif

    for (int i = 0; i < argc; i++)
    {
#ifdef _WIN32
        if (wcscmp(szArglist[i], L"--test-mode") == 0)
#else
        if (strcmp(argv[i], "--test-mode") == 0)
#endif
        {
            m_isTestMode = true;
        }
    }

#ifdef _WIN32
    LocalFree(szArglist);
#endif

    return 0;
}

void TestBase::printEntrypointHashes(
    int entryPointCount,
    int targetCount,
    ComPtr<slang::IComponentType>& composedProgram)
{
    for (int targetIndex = 0; targetIndex < targetCount; targetIndex++)
    {
        for (int entryPointIndex = 0; entryPointIndex < entryPointCount; entryPointIndex++)
        {
            ComPtr<slang::IBlob> entryPointHashBlob;
            composedProgram->getEntryPointHash(
                entryPointIndex,
                targetIndex,
                entryPointHashBlob.writeRef());

            Slang::StringBuilder strBuilder;
            strBuilder << "callIdx: " << m_globalCounter << ", entrypoint: " << entryPointIndex
                       << ", target: " << targetIndex << ", hash: ";
            m_globalCounter++;

            uint8_t* buffer = (uint8_t*)entryPointHashBlob->getBufferPointer();
            for (size_t i = 0; i < entryPointHashBlob->getBufferSize(); i++)
            {
                strBuilder << Slang::StringUtil::makeStringWithFormat("%.2X", buffer[i]);
            }
            fprintf(stdout, "%s\n", strBuilder.begin());
        }
    }
}
