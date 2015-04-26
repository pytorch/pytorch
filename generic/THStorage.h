#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THStorage.h"
#else

/* on pourrait avoir un liste chainee
   qui initialise math, lab structures (or more).
   mouais -- complique.

   Pb: THMapStorage is kind of a class
   THLab_()... comment je m'en sors?
   
   en template, faudrait que je les instancie toutes!!! oh boy!
   Et comment je sais que c'est pour Cuda? Le type float est le meme dans les <>
   
   au bout du compte, ca serait sur des pointeurs float/double... etc... = facile.
   primitives??
 */

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4

typedef struct THStorage
{
    real *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
} THStorage;

TH_API real* THStorage_(data)(const THStorage*);
TH_API long THStorage_(size)(const THStorage*);
TH_API int THStorage_(elementSize)();

/* slow access -- checks everything */
TH_API void THStorage_(set)(THStorage*, long, real);
TH_API real THStorage_(get)(const THStorage*, long);

TH_API THStorage* THStorage_(new)(void);
TH_API THStorage* THStorage_(newWithSize)(long size);
TH_API THStorage* THStorage_(newWithSize1)(real);
TH_API THStorage* THStorage_(newWithSize2)(real, real);
TH_API THStorage* THStorage_(newWithSize3)(real, real, real);
TH_API THStorage* THStorage_(newWithSize4)(real, real, real, real);
TH_API THStorage* THStorage_(newWithMapping)(const char *filename, long size, int shared);

/* takes ownership of data */
TH_API THStorage* THStorage_(newWithData)(real *data, long size);

TH_API THStorage* THStorage_(newWithAllocator)(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
TH_API THStorage* THStorage_(newWithDataAndAllocator)(
    real* data, long size, THAllocator* allocator, void *allocatorContext);

/* should not differ with API */
TH_API void THStorage_(setFlag)(THStorage *storage, const char flag);
TH_API void THStorage_(clearFlag)(THStorage *storage, const char flag);
TH_API void THStorage_(retain)(THStorage *storage);

/* might differ with other API (like CUDA) */
TH_API void THStorage_(free)(THStorage *storage);
TH_API void THStorage_(resize)(THStorage *storage, long size);
TH_API void THStorage_(fill)(THStorage *storage, real value);

#endif
