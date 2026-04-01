#pragma once

#include "../../source/compiler-core/slang-artifact.h"
#include "../../source/core/slang-common.h"
#include "../../tools/unit-test/slang-unit-test.h"

namespace Slang
{

class IFileCheck : public ICastable
{
public:
    SLANG_COM_INTERFACE(
        0x046bfe4a,
        0x99a3,
        0x402f,
        {0x83, 0xd7, 0x81, 0x8d, 0xa1, 0x38, 0xed, 0xfa})

    using ReportDiagnostic = void(SLANG_STDCALL*)(void*, TestMessageType, const char*) noexcept;

    virtual TestResult SLANG_MCALL performTest(
        const char* programName,       // Included in diagnostic messages, for example "slang-test"
        const char* rulesFilePath,     // The file from which to read the FileCheck rules
        const char* fileCheckPrefix,   // The name of the FileCheck files to use in the rules file
        const char* stringToCheck,     // The string to match with the rules
        const char* stringToCheckName, // The name of that string, for example "actual-output"
        ReportDiagnostic testReporter, // A callback for reporting diagnostic messages
        void* reporterData,            // Some data to pass on to the callback
        bool colorDiagnosticOutput     // Include color control codes in the string passed to
                                       // testReporter
        ) noexcept = 0;
};

} // namespace Slang
