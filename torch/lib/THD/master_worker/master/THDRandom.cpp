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
  new_generator->generator_id = THDState::s_nextId++;
  return new_generator;
}

THDGenerator* THDGenerator_new() {
  THDGenerator *generator = THDGenerator_newUnseeded();
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorNew, generator),
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

uint64_t THDRandom_seed(THDGenerator *_generator) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorSeed, _generator),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<uint64_t>(THDState::s_current_worker);
}

void THDRandom_manualSeed(THDGenerator *_generator, uint64_t the_seed_) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::generatorManualSeed, _generator, the_seed_),
    THDState::s_current_worker
  );
}
