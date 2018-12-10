#ifndef _IDEEP_PIN_SINGLETONS_HPP_
#define _IDEEP_PIN_SINGLETONS_HPP_

#include "ideep.hpp"

namespace ideep {
/// Put these in only one library
engine &engine::cpu_engine() {
  static engine cpu_engine;
  return cpu_engine;
}

#if defined(PROFILE_ENABLE)
namespace instruments {
const __itt_domain *domain::ideep() {
  static auto *g_dm = __itt_domain_create("ideep.interface");
  return g_dm;
}

const __itt_domain *domain::mkldnn() {
  static auto *g_dm = __itt_domain_create("ideep.mkldnn");
  return g_dm;
}

const __itt_domain *domain::keygen() {
  static auto *g_dm = __itt_domain_create("ideep.keygen");
  return g_dm;
}

const __itt_domain *domain::fetch() {
  static auto *g_dm = __itt_domain_create("ideep.fetch");
  return g_dm;
}
}
#endif

}

#endif
