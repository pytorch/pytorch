#ifndef TH_FILE_INC
#define TH_FILE_INC

#include "THStorage.h"

typedef struct THFile__ THFile;

int THFile_isOpened(THFile *self);
int THFile_isQuiet(THFile *self);
int THFile_isReadable(THFile *self);
int THFile_isWritable(THFile *self);
int THFile_isBinary(THFile *self);
int THFile_isAutoSpacing(THFile *self);
int THFile_hasError(THFile *self);

void THFile_binary(THFile *self);
void THFile_ascii(THFile *self);
void THFile_autoSpacing(THFile *self);
void THFile_noAutoSpacing(THFile *self);
void THFile_quiet(THFile *self);
void THFile_pedantic(THFile *self);
void THFile_clearError(THFile *self);

/* scalar */
unsigned char THFile_readByteScalar(THFile *self);
char THFile_readCharScalar(THFile *self);
short THFile_readShortScalar(THFile *self);
int THFile_readIntScalar(THFile *self);
long THFile_readLongScalar(THFile *self);
float THFile_readFloatScalar(THFile *self);
double THFile_readDoubleScalar(THFile *self);

void THFile_writeByteScalar(THFile *self, unsigned char scalar);
void THFile_writeCharScalar(THFile *self, char scalar);
void THFile_writeShortScalar(THFile *self, short scalar);
void THFile_writeIntScalar(THFile *self, int scalar);
void THFile_writeLongScalar(THFile *self, long scalar);
void THFile_writeFloatScalar(THFile *self, float scalar);
void THFile_writeDoubleScalar(THFile *self, double scalar);

/* storage */
long THFile_readByte(THFile *self, THByteStorage *storage);
long THFile_readChar(THFile *self, THCharStorage *storage);
long THFile_readShort(THFile *self, THShortStorage *storage);
long THFile_readInt(THFile *self, THIntStorage *storage);
long THFile_readLong(THFile *self, THLongStorage *storage);
long THFile_readFloat(THFile *self, THFloatStorage *storage);
long THFile_readDouble(THFile *self, THDoubleStorage *storage);

long THFile_writeByte(THFile *self, THByteStorage *storage);
long THFile_writeChar(THFile *self, THCharStorage *storage);
long THFile_writeShort(THFile *self, THShortStorage *storage);
long THFile_writeInt(THFile *self, THIntStorage *storage);
long THFile_writeLong(THFile *self, THLongStorage *storage);
long THFile_writeFloat(THFile *self, THFloatStorage *storage);
long THFile_writeDouble(THFile *self, THDoubleStorage *storage);

/* raw */
long THFile_readByteRaw(THFile *self, unsigned char *data, long n);
long THFile_readCharRaw(THFile *self, char *data, long n);
long THFile_readShortRaw(THFile *self, short *data, long n);
long THFile_readIntRaw(THFile *self, int *data, long n);
long THFile_readLongRaw(THFile *self, long *data, long n);
long THFile_readFloatRaw(THFile *self, float *data, long n);
long THFile_readDoubleRaw(THFile *self, double *data, long n);
long THFile_readStringRaw(THFile *self, const char *format, char **str_); /* you must deallocate str_ */

long THFile_writeByteRaw(THFile *self, unsigned char *data, long n);
long THFile_writeCharRaw(THFile *self, char *data, long n);
long THFile_writeShortRaw(THFile *self, short *data, long n);
long THFile_writeIntRaw(THFile *self, int *data, long n);
long THFile_writeLongRaw(THFile *self, long *data, long n);
long THFile_writeFloatRaw(THFile *self, float *data, long n);
long THFile_writeDoubleRaw(THFile *self, double *data, long n);
long THFile_writeStringRaw(THFile *self, const char *str, long size);

void THFile_synchronize(THFile *self);
void THFile_seek(THFile *self, long position);
void THFile_seekEnd(THFile *self);
long THFile_position(THFile *self);
void THFile_close(THFile *self);
void THFile_free(THFile *self);

#endif
