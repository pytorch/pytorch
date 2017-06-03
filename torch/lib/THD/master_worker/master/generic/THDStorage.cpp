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

ptrdiff_t THDStorage_(size)(const THDStorage* storage) {
  return storage->size;
}

size_t THDStorage_(elementSize)(void) {
  return sizeof(real);
}

THDStorage* THDStorage_(new)() {
  THDStorage* storage = THDStorage_(_alloc)();
  thpp::Type type = thpp::type_traits<real>::type;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageNew,
      type,
      storage
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
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageGet,
      storage,
      offset,
      thpp::type_traits<real>::type
    ),
    THDState::s_current_worker
  );
  return receiveValueFromWorker<real>(storage->node_id);
}

THDStorage* THDStorage_(newWithSize)(ptrdiff_t size) {
  thpp::Type type = thpp::type_traits<real>::type;
  THDStorage *storage = THDStorage_(_alloc)();
  storage->size = size;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageNewWithSize,
      type,
      storage,
      size
    ),
    THDState::s_current_worker
  );
  return storage;
}

THDStorage* THDStorage_(newWithSize1)(real value) {
  thpp::Type type = thpp::type_traits<real>::type;
  THDStorage *storage = THDStorage_(_alloc)();
  storage->size = 1;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageNewWithSize1,
      type,
      storage,
      value
    ),
    THDState::s_current_worker
  );
  return storage;
}

THDStorage* THDStorage_(newWithSize2)(real value1, real value2) {
  thpp::Type type = thpp::type_traits<real>::type;
  THDStorage *storage = THDStorage_(_alloc)();
  storage->size = 2;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageNewWithSize1,
      type,
      storage,
      value1,
      value2
    ),
    THDState::s_current_worker
  );
  return storage;
}

THDStorage* THDStorage_(newWithSize3)(real value1, real value2, real value3) {
  thpp::Type type = thpp::type_traits<real>::type;
  THDStorage *storage = THDStorage_(_alloc)();
  storage->size = 3;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageNewWithSize1,
      type,
      storage,
      value1,
      value2,
      value3
    ),
    THDState::s_current_worker
  );
  return storage;
}

THDStorage* THDStorage_(newWithSize4)(real value1, real value2, real value3, real value4) {
  thpp::Type type = thpp::type_traits<real>::type;
  THDStorage *storage = THDStorage_(_alloc)();
  storage->size = 4;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageNewWithSize1,
      type,
      storage,
      value1,
      value2,
      value3,
      value4
    ),
    THDState::s_current_worker
  );
  return storage;
}

void THDStorage_(setFlag)(THDStorage *storage, const char flag) {
  storage->flag |= flag;
}

void THDStorage_(clearFlag)(THDStorage *storage, const char flag) {
  storage->flag &= ~flag;
}

void THDStorage_(retain)(THDStorage *storage) {
  if (storage && (storage->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&storage->refcount);
}

void THDStorage_(swap)(THDStorage *storage1, THDStorage *storage2) {
  THDStorage dummy = *storage1;
  *storage1 = *storage2;
  *storage2 = dummy;
}

void THDStorage_(free)(THDStorage *storage) {
  if (!storage || !(storage->flag & TH_STORAGE_REFCOUNTED)) return;

  if (THAtomicDecrementRef(&storage->refcount)) {
    masterCommandChannel->sendMessage(
      packMessage(
        Functions::storageFree,
        storage
      ),
      THDState::s_current_worker
    );

    if (storage->flag & TH_STORAGE_VIEW)
      THDStorage_(free)(storage->view);
    delete storage;
  }
}

void THDStorage_(resize)(THDStorage *storage, ptrdiff_t size) {
  if (!(storage->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");
  if (size < storage->size)
    return;

  storage->size = size;
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageResize,
      storage,
      size
    ),
    THDState::s_current_worker
  );
}

void THDStorage_(fill)(THDStorage *storage, real value) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::storageFill,
      storage,
      value
    ),
    THDState::s_current_worker
  );
}

#endif
