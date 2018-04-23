// TODO: I'm pretty sure Constness can be done with C++ templates, ala
// std::is_const, but no time to work it out...
#ifndef _MSC_VER
  #define GENERIC_IF(Constness, FullKind, x, Kind) \
    auto && __match_key = x; \
    switch(__match_key->kind()) { \
      case FullKind: { \
        auto * value = static_cast<Constness ::torch::jit::Kind*>(__match_key); (void) value;
  #define GENERIC_ELSEIF(Constness, FullKind, Kind) \
    } break; \
      case FullKind: { \
        auto * value = static_cast<Constness ::torch::jit::Kind*>(__match_key); (void) value;
  #define GENERIC_ELSE() \
    } break; \
      default: {
  #define GENERIC_END() \
      } break; \
    };
#else
  #define GENERIC_IF(Constness, FullKind, x, Kind) \
    auto && __match_key = x; \
    if(__match_key->kind() == FullKind) { \
      auto * value = static_cast<Constness ::torch::jit::Kind*>(__match_key);
  #define GENERIC_ELSEIF(Constness, FullKind, Kind) \
    } else if(__match_key->kind() == FullKind) { \
      auto * value = static_cast<Constness ::torch::jit::Kind*>(__match_key);
  #define GENERIC_ELSE() \
    } else {
  #define GENERIC_END() \
    }
#endif