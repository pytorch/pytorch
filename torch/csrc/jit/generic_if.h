// TODO: I'm pretty sure Constness can be done with C++ templates, ala
// std::is_const, but no time to work it out...
#if (_MSC_VER <= 1900)
  #define _MSC_14
  #define _USE_IF_STATEMENTS
#endif

#ifndef _USE_IF_STATEMENTS
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