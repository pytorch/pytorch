/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


#ifndef IDEEP_ALLOCATOR_HPP
#define IDEEP_ALLOCATOR_HPP

#include <mutex>
#include <list>
#include <sstream>

namespace ideep {

#ifdef _TENSOR_MEM_ALIGNMENT_
#define SYS_MEMORY_ALIGNMENT _TENSOR_MEM_ALIGNMENT_
#else
#define SYS_MEMORY_ALIGNMENT 4096
#endif

namespace utils {

class allocator {
public:
  constexpr static size_t tensor_memalignment = SYS_MEMORY_ALIGNMENT;
  allocator() = default;

  template<class computation_t = void>
  static char *malloc(size_t size) {
    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, tensor_memalignment);
    int rc = ((ptr)? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, tensor_memalignment, size);
#endif /* _WIN32 */
    return (rc == 0) ? (char*)ptr : nullptr;
  }

  template<class computation_t = void>
  static void free(void *p) {
#ifdef _WIN32
    _aligned_free((void*)p);
#else
    ::free((void*)p);
#endif /* _WIN32 */
  }

  template<class computation_t = void>
  struct byte {
  public:
    static void *operator new(size_t sz) {
      return (void *)malloc<computation_t>(sz);
    }

    static void *operator new[](size_t sz) {
      return (void *)malloc<computation_t>(sz);
    }

    static void operator delete(void *p) { free<computation_t>(p); }
    static void operator delete[](void *p) { free<computation_t>(p); }

  private:
    char q;
  };
};

// Default SA implementation (by computation)
class scratch_allocator {
public:
  #define GET_PTR(t, p, offset) \
      (reinterpret_cast<t*>(reinterpret_cast<size_t>(p) + \
      static_cast<size_t>(offset)))

  static bool is_enabled() {
    static bool enabled = true;
    static bool checked = false;

    // Set by first run. Could not be adjusted dynamically.
    if (!checked) {
      char *env = getenv("DISABLE_MEM_CACHE_OPT");
      if (env && *env != '0')
        enabled = false;
      checked = true;
    }
    return enabled;
  }

  class mpool {
  public:
    mpool() : alloc_size_(0), free_size_(0),
        alignment_(SYS_MEMORY_ALIGNMENT), seq_(0) {}

    ~mpool() {
      std::lock_guard<std::mutex> lock(mutex_);
      for (int i = 0; i < MAX_ENTRY; ++i) {
        std::list<header_t *>& l = free_hashline_[i];
        for (auto& h: l) {
          ::free(h);
        }
      }
    }

    void *malloc(size_t size) {
      std::lock_guard<std::mutex> lock(mutex_);
      void *ptr;
      int idx = to_index(size);

      if (!free_hashline_[idx].empty()) {
        header_t *head = nullptr;
        std::list<header_t *> &list = free_hashline_[idx];
        typename std::list<header_t *>::iterator it;
        for(it = list.begin(); it != list.end(); ++it) {
          if((*it)->size_ == size) {
            head = *it;
            break;
          }
        }
        if (head) {
          list.erase(it);
          void *ptr = static_cast<void *>(head);
          free_size_ -= size;
          return GET_PTR(void, ptr, alignment_);
        }
      }

      // No cached memory
      size_t len = size + alignment_;
#if defined(WIN32)
      ptr = _aligned_malloc(size, alignment_);
#else
      int rc = ::posix_memalign(&ptr, alignment_, len);
      if (rc != 0)
        throw std::invalid_argument("Out of memory");
#endif
      header_t *head = static_cast<header_t *>(ptr);
      head->size_ = size;
      head->seq_ = seq_++;
      alloc_size_ += size;
      return GET_PTR(void, ptr, alignment_);
    }

    void free(void *ptr) {
      std::lock_guard<std::mutex> lock(mutex_);
      header_t *head = GET_PTR(header_t, ptr, -alignment_);
      int idx = to_index(head->size_);
      free_hashline_[idx].push_back(head);
      free_size_ += head->size_;
    }

  private:
    inline int to_index(size_t size) {
      std::ostringstream os;
      os << std::hex << "L" << size << "_";
      size_t hash = std::hash<std::string>{}(os.str());
      return hash % MAX_ENTRY;
    }

    typedef struct {
      size_t size_;
      int seq_;
    } header_t;

    static constexpr int MAX_ENTRY = 512;

    size_t alloc_size_;
    size_t free_size_;
    const size_t alignment_;
    std::list<header_t *> free_hashline_[MAX_ENTRY];
    std::mutex mutex_;
    int seq_;
  };

  scratch_allocator() = default;

  template<class computation_t = void>
  static inline mpool *get_mpool(void) {
    static std::shared_ptr<mpool> mpool_(new mpool());
    return mpool_.get();
  }

  template<class computation_t = void>
  static char *malloc(size_t size) {
    if (!is_enabled())
      return static_cast<char *>(allocator::malloc(size));
    else
      return static_cast<char *>(get_mpool<computation_t>()->malloc(size));
  }

  template<class computation_t = void>
  static void free(void *p) {
    if (!is_enabled())
      allocator::free(p);
    else
      get_mpool<computation_t>()->free(p);
  }

  template<class computation_t = void>
  struct byte {
  public:
    static void *operator new(size_t sz) {
      return (void *)malloc<computation_t>(sz);
    }

    static void *operator new[](size_t sz) {
      return (void *)malloc<computation_t>(sz);
    }

    static void operator delete(void *p) { free<computation_t>(p); }
    static void operator delete[](void *p) {
      free<computation_t>(p);
    }

  private:
    char q;
  };
};
}
}

#define SCRATCH_ALLOCATOR(computation_t) \
    ideep::utils::scratch_allocator, ideep::computation_t

#endif
