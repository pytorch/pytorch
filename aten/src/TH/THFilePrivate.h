#include <TH/THGeneral.h>

#include <TH/THHalf.h>


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

    ssize_t (*readByte)(THFile *self, uint8_t *data, ssize_t n);
    ssize_t (*readChar)(THFile *self, int8_t *data, ssize_t n);
    ssize_t (*readShort)(THFile *self, int16_t *data, ssize_t n);
    ssize_t (*readInt)(THFile *self, int32_t *data, ssize_t n);
    ssize_t (*readLong)(THFile *self, int64_t *data, ssize_t n);
    ssize_t (*readFloat)(THFile *self, float *data, ssize_t n);
    ssize_t (*readDouble)(THFile *self, double *data, ssize_t n);
    ssize_t (*readHalf)(THFile *self, THHalf *data, ssize_t n);
    ssize_t (*readString)(THFile *self, const char *format, char **str_);

    ssize_t (*writeByte)(THFile *self, uint8_t *data, ssize_t n);
    ssize_t (*writeChar)(THFile *self, int8_t *data, ssize_t n);
    ssize_t (*writeShort)(THFile *self, int16_t *data, ssize_t n);
    ssize_t (*writeInt)(THFile *self, int32_t *data, ssize_t n);
    ssize_t (*writeLong)(THFile *self, int64_t *data, ssize_t n);
    ssize_t (*writeFloat)(THFile *self, float *data, ssize_t n);
    ssize_t (*writeDouble)(THFile *self, double *data, ssize_t n);
    ssize_t (*writeHalf)(THFile *self, THHalf *data, ssize_t n);
    ssize_t (*writeString)(THFile *self, const char *str, ssize_t size);

    void (*synchronize)(THFile *self);
    void (*seek)(THFile *self, ssize_t position);
    void (*seekEnd)(THFile *self);
    ssize_t (*position)(THFile *self);
    void (*close)(THFile *self);
    void (*free)(THFile *self);
};
