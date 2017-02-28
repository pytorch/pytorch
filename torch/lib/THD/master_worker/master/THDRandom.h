#pragma once

#include <TH/TH.h>
#include "../../THD.h"

typedef struct THDGenerator {
  // Additional fields
  unsigned long long generator_id;
} THDGenerator;

/* Manipulate THDGenerator objects */
THD_API THDGenerator * THDGenerator_new(void);
THD_API THDGenerator * THDGenerator_copy(THDGenerator *self, THDGenerator *from);
THD_API void THDGenerator_free(THDGenerator *gen);

/* Initializes the random number generator from /dev/urandom (or on Windows
platforms with the current time (granularity: seconds)) and returns the seed. */
THD_API unsigned long THDRandom_seed(THDGenerator *_generator);

/* Initializes the random number generator with the given long "the_seed_". */
THD_API void THDRandom_manualSeed(THDGenerator *_generator, unsigned long the_seed_);
