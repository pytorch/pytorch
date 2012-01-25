#ifndef TH_DISK_FILE_INC
#define TH_DISK_FILE_INC

#include "THFile.h"

THFile *THDiskFile_new(const char *name, const char *mode, int isQuiet);
THFile *THPipeFile_new(const char *name, const char *mode, int isQuiet);

const char *THDiskFile_name(THFile *self);

int THDiskFile_isLittleEndianCPU(void);
int THDiskFile_isBigEndianCPU(void);
void THDiskFile_nativeEndianEncoding(THFile *self);
void THDiskFile_littleEndianEncoding(THFile *self);
void THDiskFile_bigEndianEncoding(THFile *self);

#endif
