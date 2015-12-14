#ifndef TH_MEMORY_FILE_INC
#define TH_MEMORY_FILE_INC

#include "THFile.h"
#include "THStorage.h"

TH_API THFile *THMemoryFile_newWithStorage(THCharStorage *storage, const char *mode);
TH_API THFile *THMemoryFile_new(const char *mode);

TH_API THCharStorage *THMemoryFile_storage(THFile *self);
TH_API void THMemoryFile_longSize(THFile *self, int size);

#endif
