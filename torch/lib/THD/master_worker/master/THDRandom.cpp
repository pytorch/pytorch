#include "THD.h"
#include "THDRandom.h"

#include "Utils.hpp"
#include "State.hpp"
#include "master_worker/common/RPC.hpp"
#include "master_worker/common/Functions.hpp"
#include "master_worker/master/Master.hpp"

#include <THPP/Traits.hpp>

#include <cstring>

using namespace thd;
using namespace rpc;
using namespace master;

static THDGenerator* THDGenerator_newUnseeded() {
  THDGenerator *new_generator = new THDGenerator();
  std::memset(reinterpret_cast<void*>(new_generator), 0, sizeof(new_generator));
  new_generator->generator_id = THDState::s_nextId++;
  return new_generator;
}

THDGenerator* THDGenerator_new() {
  THDGenerator *generator = THDGenerator_newUnseeded();
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorConstruct, generator),
    THDState::s_current_worker
  );
  THDRandom_seed(generator);
  return generator;
}

THDGenerator* THDGenerator_copy(THDGenerator *self, THDGenerator *from) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorCopy, self, from),
    THDState::s_current_worker
  );

  return self;
}

void THDGenerator_free(THDGenerator *self) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorFree, self),
    THDState::s_current_worker
  );

  delete self;
}

unsigned long THDRandom_seed(THDGenerator *_generator) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorSeed, _generator),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<unsigned long>(THDState::s_current_worker);
}

void THDRandom_manualSeed(THDGenerator *_generator, unsigned long the_seed_) {
  THDGenerator *blank = THDGenerator_newUnseeded();
  THDGenerator_copy(_generator, blank);
  THDGenerator_free(blank);

  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorManualSeed, _generator, the_seed_),
    THDState::s_current_worker
  );
}
