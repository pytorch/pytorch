#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDStorage.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

static THDStorage* THDStorage_(_alloc)() {
  THDStorage* new_storage = new THDStorage();
  std::memset(reinterpret_cast<void*>(new_storage), 0, sizeof(new_storage));
  new_storage->refcount = 1;
  new_storage->storage_id = THDState::s_nextId++;
  new_storage->node_id = THDState::s_current_worker;
  new_storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return new_storage;
}

THDStorage* THDStorage_(new)() {
  THDStorage* storage = THDStorage_(_alloc)();
  Type type = type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageConstruct,
      type,
      storage
    ),
    THDState::s_current_worker
  );
  return storage;
}

void THDStorage_(resize)(THDStorage *storage, ptrdiff_t size)
{
  if(!(storage->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  storage->size = size;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageResize,
      storage
    ),
    THDState::s_current_worker
  );
}

void THDStorage_(free)(THDStorage *storage)
{
  if(!storage || !(storage->flag & TH_STORAGE_REFCOUNTED)) return;

  if (THAtomicDecrementRef(&storage->refcount)) {
    masterCommandChannel->sendMessage(
      packMessage(
        Functions::storageFree,
        storage
      ),
      THDState::s_current_worker
    );

    if(storage->flag & TH_STORAGE_VIEW)
      THDStorage_(free)(storage->view);
    delete storage;
  }
}

void THDStorage_(retain)(THDStorage *storage) {
  if(storage && (storage->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&storage->refcount);
}

ptrdiff_t THDStorage_(size)(const THDStorage* storage) {
  return storage->size;
}

THDStorage* THDStorage_(newWithSize)(ptrdiff_t size) {
  Type type = type_traits<real>::type;
  THDStorage *storage = THDStorage_(_alloc)();
  storage->size = size;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageConstructWithSize,
      type,
      storage,
      size
    ),
    THDState::s_current_worker
  );
  return storage;
}

void THDStorage_(set)(THDStorage* storage, ptrdiff_t offset, real value) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageSet,
      storage,
      offset,
      value
    ),
    THDState::s_current_worker
  );
}

real THDStorage_(get)(const THDStorage* storage, ptrdiff_t offset) {
  THError("get not supported yet");
  return 0;
}

#endif
