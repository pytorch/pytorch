// slangc-tool.h

#ifndef SLANGC_TOOL_H_INCLUDED
#define SLANGC_TOOL_H_INCLUDED

#include "../../source/core/slang-std-writers.h"

/* The slangc 'tool' interface, such that slangc like functionality is available directly without
invoking slangc command line tool, or need for a dll/shared library. */
struct SlangCTool
{
    static SlangResult innerMain(
        Slang::StdWriters* stdWriters,
        SlangSession* session,
        int argc,
        const char* const* argv);
};

#endif // SLANGC_TOOL_H_INCLUDED
