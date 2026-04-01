#ifndef SLANG_LZ4_COMPRESSION_SYSTEM_H
#define SLANG_LZ4_COMPRESSION_SYSTEM_H

#include "slang-basic.h"
#include "slang-com-ptr.h"
#include "slang-compression-system.h"

namespace Slang
{

class LZ4CompressionSystem
{
public:
    /* Get the LZ4 compression system singleton. */
    static ICompressionSystem* getSingleton();
};

} // namespace Slang

#endif
