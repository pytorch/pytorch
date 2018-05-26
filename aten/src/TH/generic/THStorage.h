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
#define TH_STORAGE_VIEW       8

// Struct definition is moved to THStorage.hpp (so this file stays C compatible)
typedef struct THStorage THStorage;

namespace at {
  class StorageImpl;
  typedef StorageImpl ByteStorageImpl;
  typedef StorageImpl CharStorageImpl;
  typedef StorageImpl ShortStorageImpl;
  typedef StorageImpl IntStorageImpl;
  typedef StorageImpl LongStorageImpl;
  typedef StorageImpl FloatStorageImpl;
  typedef StorageImpl DoubleStorageImpl;
  typedef StorageImpl HalfStorageImpl;

  TH_API void THStorage_(copyShort)(at::StorageImpl *storage, at::ShortStorageImpl *src);
  TH_API void THStorage_(copyInt)(at::StorageImpl *storage, at::IntStorageImpl *src);
  TH_API void THStorage_(copyLong)(at::StorageImpl *storage, at::LongStorageImpl *src);
  TH_API void THStorage_(copyFloat)(at::StorageImpl *storage, at::FloatStorageImpl *src);
  TH_API void THStorage_(copyDouble)(at::StorageImpl *storage, at::DoubleStorageImpl *src);
  TH_API void THStorage_(copyHalf)(at::StorageImpl *storage, at::HalfStorageImpl *src);
}

TH_API real* THStorage_(data)(const at::StorageImpl *);
TH_API ptrdiff_t THStorage_(size)(const at::StorageImpl *);
TH_API size_t THStorage_(elementSize)(void);

/* slow access -- checks everything */
TH_API void THStorage_(set)(at::StorageImpl *, ptrdiff_t, real);
TH_API real THStorage_(get)(const at::StorageImpl *, ptrdiff_t);

TH_API at::StorageImpl * THStorage_(new)(void);
TH_API at::StorageImpl * THStorage_(newWithSize)(ptrdiff_t size);
TH_API at::StorageImpl * THStorage_(newWithSize1)(real);
TH_API at::StorageImpl * THStorage_(newWithSize2)(real, real);	
TH_API at::StorageImpl * THStorage_(newWithSize3)(real, real, real);	
TH_API at::StorageImpl * THStorage_(newWithSize4)(real, real, real, real);
TH_API at::StorageImpl * THStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);

/* takes ownership of data */
TH_API at::StorageImpl * THStorage_(newWithData)(real *data, ptrdiff_t size);

TH_API at::StorageImpl * THStorage_(newWithAllocator)(ptrdiff_t size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
TH_API at::StorageImpl * THStorage_(newWithDataAndAllocator)(
    real* data, ptrdiff_t size, THAllocator* allocator, void *allocatorContext);

/* should not differ with API */
TH_API void THStorage_(setFlag)(at::StorageImpl *storage, const char flag);
TH_API void THStorage_(clearFlag)(at::StorageImpl *storage, const char flag);
TH_API void THStorage_(retain)(at::StorageImpl *storage);
TH_API void THStorage_(swap)(at::StorageImpl *storage1, at::StorageImpl *storage2);

/* used by StorageSharing */
TH_API int THStorage_(retainIfLive)(at::StorageImpl *storage);

/* might differ with other API (like CUDA) */
TH_API void THStorage_(free)(at::StorageImpl *storage);
TH_API void THStorage_(resize)(at::StorageImpl *storage, ptrdiff_t size);
TH_API void THStorage_(fill)(at::StorageImpl *storage, real value);

#endif
