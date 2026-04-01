// slangi-tool.h

#ifndef SLANGI_TOOL_H_INCLUDED
#define SLANGI_TOOL_H_INCLUDED

#include "../../source/core/slang-std-writers.h"

/* The slangi 'tool' interface, such that slangc like functionality is available directly without
invoking slangc command line tool, or need for a dll/shared library. */
namespace SlangITool
{
SlangResult innerMain(
    Slang::StdWriters* stdWriters,
    SlangSession* session,
    int argc,
    const char* const* argv);
};

#endif // SLANGI_TOOL_H_INCLUDED
