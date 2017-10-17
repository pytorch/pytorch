#include "THGeneral.h"

#include "THHalf.h"


struct THFile__
{
    struct THFileVTable *vtable;

    int isQuiet;
    int isReadable;
    int isWritable;
    int isBinary;
    int isAutoSpacing;
    int hasError;
};

/* virtual table definition */

struct THFileVTable
{
    int (*isOpened)(THFile *self);

    size_t (*readByte)(THFile *self, uint8_t *data, size_t n);
    size_t (*readChar)(THFile *self, int8_t *data, size_t n);
    size_t (*readShort)(THFile *self, int16_t *data, size_t n);
    size_t (*readInt)(THFile *self, int32_t *data, size_t n);
    size_t (*readLong)(THFile *self, int64_t *data, size_t n);
    size_t (*readFloat)(THFile *self, float *data, size_t n);
    size_t (*readDouble)(THFile *self, double *data, size_t n);
    size_t (*readHalf)(THFile *self, THHalf *data, size_t n);
    size_t (*readString)(THFile *self, const char *format, char **str_);

    size_t (*writeByte)(THFile *self, uint8_t *data, size_t n);
    size_t (*writeChar)(THFile *self, int8_t *data, size_t n);
    size_t (*writeShort)(THFile *self, int16_t *data, size_t n);
    size_t (*writeInt)(THFile *self, int32_t *data, size_t n);
    size_t (*writeLong)(THFile *self, int64_t *data, size_t n);
    size_t (*writeFloat)(THFile *self, float *data, size_t n);
    size_t (*writeDouble)(THFile *self, double *data, size_t n);
    size_t (*writeHalf)(THFile *self, THHalf *data, size_t n);
    size_t (*writeString)(THFile *self, const char *str, size_t size);

    void (*synchronize)(THFile *self);
    void (*seek)(THFile *self, size_t position);
    void (*seekEnd)(THFile *self);
    size_t (*position)(THFile *self);
    void (*close)(THFile *self);
    void (*free)(THFile *self);
};
