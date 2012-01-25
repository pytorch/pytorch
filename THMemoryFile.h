#ifndef TH_MEMORY_FILE_INC
#define TH_MEMORY_FILE_INC

#include "THFile.h"
#include "THStorage.h"

THFile *THMemoryFile_newWithStorage(THCharStorage *storage, const char *mode);
THFile *THMemoryFile_new(const char *mode);

THCharStorage *THMemoryFile_storage(THFile *self);

#endif
