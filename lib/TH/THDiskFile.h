#ifndef TH_DISK_FILE_INC
#define TH_DISK_FILE_INC

#include "THFile.h"

TH_API THFile *THDiskFile_new(const char *name, const char *mode, int isQuiet);
TH_API THFile *THPipeFile_new(const char *name, const char *mode, int isQuiet);

TH_API const char *THDiskFile_name(THFile *self);

TH_API int THDiskFile_isLittleEndianCPU(void);
TH_API int THDiskFile_isBigEndianCPU(void);
TH_API void THDiskFile_nativeEndianEncoding(THFile *self);
TH_API void THDiskFile_littleEndianEncoding(THFile *self);
TH_API void THDiskFile_bigEndianEncoding(THFile *self);

#endif
