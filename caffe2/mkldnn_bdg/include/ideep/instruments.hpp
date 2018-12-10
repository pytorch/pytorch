#ifndef _ITTNOTIFY_HPP_
#define _ITTNOTIFY_HPP_

#if defined(PROFILE_ENABLE)
#include <ittnotify.h>
#include <string>

namespace ideep {
namespace instruments {
/// Global notification control API
/// Overhead is very large, use with causion
//
class ittnotify {
public:
  static void pause() {
    __itt_pause();
  }

  static void resume() {
    __itt_resume();
  }

  static void detach() {
    __itt_detach();
  }
};

class domain {
public:
  static const __itt_domain *ideep();/* {
    static auto *g_dm = __itt_domain_create("ideep.interface");
    return g_dm;
  } */

  static const __itt_domain *mkldnn();/* {
    static auto *g_dm = __itt_domain_create("ideep.mkldnn");
    return g_dm;
  } */

  static const __itt_domain *keygen();/* {
    static const auto *g_dm = __itt_domain_create("ideep.keygen");
    return g_dm;
  } */

  static const __itt_domain *fetch();/* {
    static const auto *g_dm = __itt_domain_create("ideep.fetch");
    return g_dm;
  } */
};

class frame_impl {
public:
  explicit frame_impl(const __itt_domain *dm) : dm_(dm) {
    id_ = __itt_id_make(this, 0);
    __itt_id_create(dm, id_);
  }

  ~frame_impl() {
    if (armed)
      end();
    __itt_id_destroy(dm_, id_);
  }

  void begin() {
    __itt_frame_begin_v3(dm_, &id_);
    armed = true;
  }

  void end() {
    __itt_frame_end_v3(dm_, &id_);
    armed = false;
  }

private:
  __itt_id id_;
  const __itt_domain *dm_;
  bool armed;
};

class frame {
public:
  static std::unique_ptr<frame_impl> mark_start(const __itt_domain *dm) {
    return std::unique_ptr<frame_impl>(new frame_impl(dm));
  }

  static void mark_end(std::unique_ptr<frame_impl> fr) {
    fr->end();
  }
};
}
}
#else
#define __itt_frame_begin_v3(x, y)
#define __itt_frame_end_v3(x, y)
#endif
#endif
