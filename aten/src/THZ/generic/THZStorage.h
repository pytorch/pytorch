#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorage.h"
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

#define THZ_STORAGE_REFCOUNTED 1
#define THZ_STORAGE_RESIZABLE  2
#define THZ_STORAGE_FREEMEM    4
#define THZ_STORAGE_VIEW       8

typedef struct THZStorage
{
    ntype *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THZStorage *view;
} THZStorage;

TH_API ntype* THZStorage_(data)(const THZStorage*);
TH_API ptrdiff_t THZStorage_(size)(const THZStorage*);
TH_API size_t THZStorage_(elementSize)(void);

/* slow access -- checks everything */
TH_API void THZStorage_(set)(THZStorage*, ptrdiff_t, ntype);
TH_API ntype THZStorage_(get)(const THZStorage*, ptrdiff_t);

TH_API THZStorage* THZStorage_(new)(void);
TH_API THZStorage* THZStorage_(newWithSize)(ptrdiff_t size);
TH_API THZStorage* THZStorage_(newWithSize1)(ntype);
TH_API THZStorage* THZStorage_(newWithSize2)(ntype, ntype);
TH_API THZStorage* THZStorage_(newWithSize3)(ntype, ntype, ntype);
TH_API THZStorage* THZStorage_(newWithSize4)(ntype, ntype, ntype, ntype);
TH_API THZStorage* THZStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);

/* takes ownership of data */
TH_API THZStorage* THZStorage_(newWithData)(ntype *data, ptrdiff_t size);

TH_API THZStorage* THZStorage_(newWithAllocator)(ptrdiff_t size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
TH_API THZStorage* THZStorage_(newWithDataAndAllocator)(
    ntype* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

/* should not differ with API */
TH_API void THZStorage_(setFlag)(THZStorage *storage, const char flag);
TH_API void THZStorage_(clearFlag)(THZStorage *storage, const char flag);
TH_API void THZStorage_(retain)(THZStorage *storage);
TH_API void THZStorage_(swap)(THZStorage *storage1, THZStorage *storage2);

/* might differ with other API (like CUDA) */
TH_API void THZStorage_(free)(THZStorage *storage);
TH_API void THZStorage_(resize)(THZStorage *storage, ptrdiff_t size);
TH_API void THZStorage_(fill)(THZStorage *storage, ntype value);

#endif
