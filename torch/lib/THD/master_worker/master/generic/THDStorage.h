#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDStorage.h"
#else

typedef struct THDStorage {
  uint64_t storage_id;
  ptrdiff_t size;
  int refcount;
  char flag;
  // these are here only so that the struct has a similar structure to TH
  void* allocator;
  void* allocatorContext;
  struct THDStorage *view;
  // Additional fields
  int node_id;
  int device_id; // unused at the moment
} THDStorage;

THD_API ptrdiff_t THDStorage_(size)(const THDStorage*);
THD_API size_t THDStorage_(elementSize)(void);

/* slow access -- checks everything */
THD_API void THDStorage_(set)(THDStorage*, ptrdiff_t, real);
THD_API real THDStorage_(get)(const THDStorage*, ptrdiff_t);

THD_API THDStorage* THDStorage_(new)(void);
THD_API THDStorage* THDStorage_(newWithSize)(ptrdiff_t size);
THD_API THDStorage* THDStorage_(newWithSize1)(real);
THD_API THDStorage* THDStorage_(newWithSize2)(real, real);
THD_API THDStorage* THDStorage_(newWithSize3)(real, real, real);
THD_API THDStorage* THDStorage_(newWithSize4)(real, real, real, real);
THD_API THDStorage* THDStorage_(newWithMapping)(const char *filename, ptrdiff_t size, int flags);

/* takes ownership of data */
//THD_API THDStorage* THDStorage_(newWithAllocator)(ptrdiff_t size,
                                               //THDAllocator* allocator,
                                               //void *allocatorContext);
//THD_API THDStorage* THDStorage_(newWithDataAndAllocator)(
    //real* data, ptrdiff_t size, THDAllocator* allocator, void *allocatorContext);

/* should not differ with API */
THD_API void THDStorage_(setFlag)(THDStorage *storage, const char flag);
THD_API void THDStorage_(clearFlag)(THDStorage *storage, const char flag);
THD_API void THDStorage_(retain)(THDStorage *storage);
THD_API void THDStorage_(swap)(THDStorage *storage1, THDStorage *storage2);

/* might differ with other API (like CUDA) */
THD_API void THDStorage_(free)(THDStorage *storage);
THD_API void THDStorage_(resize)(THDStorage *storage, ptrdiff_t size);
THD_API void THDStorage_(fill)(THDStorage *storage, real value);

#endif

