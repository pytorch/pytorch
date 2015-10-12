// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_PROTOBUF_ARENA_H__
#define GOOGLE_PROTOBUF_ARENA_H__

#include <limits>
#ifdef max
#undef max  // Visual Studio defines this macro
#endif
#if __cplusplus >= 201103L
#include <google/protobuf/stubs/type_traits.h>
#endif
#if defined(_MSC_VER) && !_HAS_EXCEPTIONS
// Work around bugs in MSVC <typeinfo> header when _HAS_EXCEPTIONS=0.
#include <exception>
#include <typeinfo>
namespace std {
using type_info = ::type_info;
}
#else
#include <typeinfo>
#endif

#include <google/protobuf/stubs/atomic_sequence_num.h>
#include <google/protobuf/stubs/atomicops.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/mutex.h>
#include <google/protobuf/stubs/type_traits.h>

namespace google {
namespace protobuf {

class Arena;       // defined below
class Message;     // message.h

namespace internal {
class ArenaString; // arenastring.h
class LazyField;   // lazy_field.h

template<typename Type>
class GenericTypeHandler; // repeated_field.h

// Templated cleanup methods.
template<typename T> void arena_destruct_object(void* object) {
  reinterpret_cast<T*>(object)->~T();
}
template<typename T> void arena_delete_object(void* object) {
  delete reinterpret_cast<T*>(object);
}
inline void arena_free(void* object, size_t size) {
  free(object);
}

}  // namespace internal

// ArenaOptions provides optional additional parameters to arena construction
// that control its block-allocation behavior.
struct ArenaOptions {
  // This defines the size of the first block requested from the system malloc.
  // Subsequent block sizes will increase in a geometric series up to a maximum.
  size_t start_block_size;

  // This defines the maximum block size requested from system malloc (unless an
  // individual arena allocation request occurs with a size larger than this
  // maximum). Requested block sizes increase up to this value, then remain
  // here.
  size_t max_block_size;

  // An initial block of memory for the arena to use, or NULL for none. If
  // provided, the block must live at least as long as the arena itself. The
  // creator of the Arena retains ownership of the block after the Arena is
  // destroyed.
  char* initial_block;

  // The size of the initial block, if provided.
  size_t initial_block_size;

  // A function pointer to an alloc method that returns memory blocks of size
  // requested. By default, it contains a ptr to the malloc function.
  //
  // NOTE: block_alloc and dealloc functions are expected to behave like
  // malloc and free, including Asan poisoning.
  void* (*block_alloc)(size_t);
  // A function pointer to a dealloc method that takes ownership of the blocks
  // from the arena. By default, it contains a ptr to a wrapper function that
  // calls free.
  void (*block_dealloc)(void*, size_t);

  // Hooks for adding external functionality such as user-specific metrics
  // collection, specific debugging abilities, etc.
  // Init hook may return a pointer to a cookie to be stored in the arena.
  // reset and destruction hooks will then be called with the same cookie
  // pointer. This allows us to save an external object per arena instance and
  // use it on the other hooks (Note: It is just as legal for init to return
  // NULL and not use the cookie feature).
  // on_arena_reset and on_arena_destruction also receive the space used in
  // the arena just before the reset.
  void* (*on_arena_init)(Arena* arena);
  void (*on_arena_reset)(Arena* arena, void* cookie, uint64 space_used);
  void (*on_arena_destruction)(Arena* arena, void* cookie, uint64 space_used);

  // type_info is promised to be static - its lifetime extends to
  // match program's lifetime (It is given by typeid operator).
  // Note: typeid(void) will be passed as allocated_type every time we
  // intentionally want to avoid monitoring an allocation. (i.e. internal
  // allocations for managing the arena)
  void (*on_arena_allocation)(const std::type_info* allocated_type,
      uint64 alloc_size, void* cookie);

  ArenaOptions()
      : start_block_size(kDefaultStartBlockSize),
        max_block_size(kDefaultMaxBlockSize),
        initial_block(NULL),
        initial_block_size(0),
        block_alloc(&malloc),
        block_dealloc(&internal::arena_free),
        on_arena_init(NULL),
        on_arena_reset(NULL),
        on_arena_destruction(NULL),
        on_arena_allocation(NULL) {}

 private:
  // Constants define default starting block size and max block size for
  // arena allocator behavior -- see descriptions above.
  static const size_t kDefaultStartBlockSize = 256;
  static const size_t kDefaultMaxBlockSize   = 8192;
};

// Support for non-RTTI environments. (The metrics hooks API uses type
// information.)
#ifndef GOOGLE_PROTOBUF_NO_RTTI
#define RTTI_TYPE_ID(type) (&typeid(type))
#else
#define RTTI_TYPE_ID(type) (NULL)
#endif

// Arena allocator. Arena allocation replaces ordinary (heap-based) allocation
// with new/delete, and improves performance by aggregating allocations into
// larger blocks and freeing allocations all at once. Protocol messages are
// allocated on an arena by using Arena::CreateMessage<T>(Arena*), below, and
// are automatically freed when the arena is destroyed.
//
// This is a thread-safe implementation: multiple threads may allocate from the
// arena concurrently. Destruction is not thread-safe and the destructing
// thread must synchronize with users of the arena first.
//
// An arena provides two allocation interfaces: CreateMessage<T>, which works
// for arena-enabled proto2 message types as well as other types that satisfy
// the appropriate protocol (described below), and Create<T>, which works for
// any arbitrary type T. CreateMessage<T> is better when the type T supports it,
// because this interface (i) passes the arena pointer to the created object so
// that its sub-objects and internal allocations can use the arena too, and (ii)
// elides the object's destructor call when possible. Create<T> does not place
// any special requirements on the type T, and will invoke the object's
// destructor when the arena is destroyed.
//
// The arena message allocation protocol, required by CreateMessage<T>, is as
// follows:
//
// - The type T must have (at least) two constructors: a constructor with no
//   arguments, called when a T is allocated on the heap; and a constructor with
//   a google::protobuf::Arena* argument, called when a T is allocated on an arena. If the
//   second constructor is called with a NULL arena pointer, it must be
//   equivalent to invoking the first (no-argument) constructor.
//
// - The type T must have a particular type trait: a nested type
//   |InternalArenaConstructable_|. This is usually a typedef to |void|. If no
//   such type trait exists, then the instantiation CreateMessage<T> will fail
//   to compile.
//
// - The type T *may* have the type trait |DestructorSkippable_|. If this type
//   trait is present in the type, then its destructor will not be called if and
//   only if it was passed a non-NULL arena pointer. If this type trait is not
//   present on the type, then its destructor is always called when the
//   containing arena is destroyed.
//
// - One- and two-user-argument forms of CreateMessage<T>() also exist that
//   forward these constructor arguments to T's constructor: for example,
//   CreateMessage<T>(Arena*, arg1, arg2) forwards to a constructor T(Arena*,
//   arg1, arg2).
//
// This protocol is implemented by all arena-enabled proto2 message classes as
// well as RepeatedPtrField.
class LIBPROTOBUF_EXPORT Arena {
 public:
  // Arena constructor taking custom options. See ArenaOptions below for
  // descriptions of the options available.
  explicit Arena(const ArenaOptions& options) : options_(options) {
    Init();
  }

  // Default constructor with sensible default options, tuned for average
  // use-cases.
  Arena() {
    Init();
  }

  // Destructor deletes all owned heap allocated objects, and destructs objects
  // that have non-trivial destructors, except for proto2 message objects whose
  // destructors can be skipped. Also, frees all blocks except the initial block
  // if it was passed in.
  ~Arena();

  // API to create proto2 message objects on the arena. If the arena passed in
  // is NULL, then a heap allocated object is returned. Type T must be a message
  // defined in a .proto file with cc_enable_arenas set to true, otherwise a
  // compilation error will occur.
  //
  // RepeatedField and RepeatedPtrField may also be instantiated directly on an
  // arena with this method.
  //
  // This function also accepts any type T that satisfies the arena message
  // allocation protocol, documented above.
  template <typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* CreateMessage(::google::protobuf::Arena* arena) {
    if (arena == NULL) {
      return new T;
    } else {
      return arena->CreateMessageInternal<T>(static_cast<T*>(0));
    }
  }

  // One-argument form of CreateMessage. This is useful for constructing objects
  // that implement the arena message construction protocol described above but
  // take additional constructor arguments.
  template <typename T, typename Arg> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* CreateMessage(::google::protobuf::Arena* arena, const Arg& arg) {
    if (arena == NULL) {
      return new T(NULL, arg);
    } else {
      return arena->CreateMessageInternal<T>(static_cast<T*>(0),
                                             arg);
    }
  }

  // Two-argument form of CreateMessage. This is useful for constructing objects
  // that implement the arena message construction protocol described above but
  // take additional constructor arguments.
  template <typename T, typename Arg1, typename Arg2> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* CreateMessage(::google::protobuf::Arena* arena,
                          const Arg1& arg1,
                          const Arg2& arg2) {
    if (arena == NULL) {
      return new T(NULL, arg1, arg2);
    } else {
      return arena->CreateMessageInternal<T>(static_cast<T*>(0),
                                             arg1, arg2);
    }
  }

  // API to create any objects on the arena. Note that only the object will
  // be created on the arena; the underlying ptrs (in case of a proto2 message)
  // will be still heap allocated. Proto messages should usually be allocated
  // with CreateMessage<T>() instead.
  //
  // Note that even if T satisfies the arena message construction protocol
  // (InternalArenaConstructable_ trait and optional DestructorSkippable_
  // trait), as described above, this function does not follow the protocol;
  // instead, it treats T as a black-box type, just as if it did not have these
  // traits. Specifically, T's constructor arguments will always be only those
  // passed to Create<T>() -- no additional arena pointer is implicitly added.
  // Furthermore, the destructor will always be called at arena destruction time
  // (unless the destructor is trivial). Hence, from T's point of view, it is as
  // if the object were allocated on the heap (except that the underlying memory
  // is obtained from the arena).
  template <typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* Create(::google::protobuf::Arena* arena) {
    if (arena == NULL) {
      return new T();
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value);
    }
  }

  // Version of the above with one constructor argument for the created object.
  template <typename T, typename Arg> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* Create(::google::protobuf::Arena* arena, const Arg& arg) {
    if (arena == NULL) {
      return new T(arg);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg);
    }
  }

  // Version of the above with two constructor arguments for the created object.
  template <typename T, typename Arg1, typename Arg2> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* Create(::google::protobuf::Arena* arena, const Arg1& arg1, const Arg2& arg2) {
    if (arena == NULL) {
      return new T(arg1, arg2);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg1, arg2);
    }
  }

  // Version of the above with three constructor arguments for the created
  // object.
  template <typename T, typename Arg1, typename Arg2, typename Arg3>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE static T* Create(::google::protobuf::Arena* arena,
                                           const Arg1& arg1, const Arg2& arg2,
                                           const Arg3& arg3) {
    if (arena == NULL) {
      return new T(arg1, arg2, arg3);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg1, arg2, arg3);
    }
  }

  // Version of the above with four constructor arguments for the created
  // object.
  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE static T* Create(::google::protobuf::Arena* arena,
                                           const Arg1& arg1, const Arg2& arg2,
                                           const Arg3& arg3, const Arg4& arg4) {
    if (arena == NULL) {
      return new T(arg1, arg2, arg3, arg4);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg1, arg2, arg3, arg4);
    }
  }

  // Version of the above with five constructor arguments for the created
  // object.
  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE static T* Create(::google::protobuf::Arena* arena,
                                           const Arg1& arg1, const Arg2& arg2,
                                           const Arg3& arg3, const Arg4& arg4,
                                           const Arg5& arg5) {
    if (arena == NULL) {
      return new T(arg1, arg2, arg3, arg4, arg5);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg1, arg2, arg3, arg4, arg5);
    }
  }

  // Version of the above with six constructor arguments for the created
  // object.
  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5, typename Arg6>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE static T* Create(::google::protobuf::Arena* arena,
                                           const Arg1& arg1, const Arg2& arg2,
                                           const Arg3& arg3, const Arg4& arg4,
                                           const Arg5& arg5, const Arg6& arg6) {
    if (arena == NULL) {
      return new T(arg1, arg2, arg3, arg4, arg5, arg6);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg1, arg2, arg3, arg4, arg5, arg6);
    }
  }

  // Version of the above with seven constructor arguments for the created
  // object.
  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE static T* Create(::google::protobuf::Arena* arena,
                                           const Arg1& arg1, const Arg2& arg2,
                                           const Arg3& arg3, const Arg4& arg4,
                                           const Arg5& arg5, const Arg6& arg6,
                                           const Arg7& arg7) {
    if (arena == NULL) {
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    } else {
      return arena->CreateInternal<T>(google::protobuf::internal::has_trivial_destructor<T>::value,
                                      arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
  }

  // Version of the above with eight constructor arguments for the created
  // object.
  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5, typename Arg6, typename Arg7,
            typename Arg8>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE static T* Create(::google::protobuf::Arena* arena,
                                           const Arg1& arg1, const Arg2& arg2,
                                           const Arg3& arg3, const Arg4& arg4,
                                           const Arg5& arg5, const Arg6& arg6,
                                           const Arg7& arg7, const Arg8& arg8) {
    if (arena == NULL) {
      return new T(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    } else {
      return arena->CreateInternal<T>(
          google::protobuf::internal::has_trivial_destructor<T>::value,
          arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }
  }

  // Create an array of object type T on the arena *without* invoking the
  // constructor of T. If `arena` is null, then the return value should be freed
  // with `delete[] x;` (or `::operator delete[](x);`).
  // To ensure safe uses, this function checks at compile time
  // (when compiled as C++11) that T is trivially default-constructible and
  // trivially destructible.
  template <typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* CreateArray(::google::protobuf::Arena* arena, size_t num_elements) {
    GOOGLE_CHECK_LE(num_elements,
             std::numeric_limits<size_t>::max() / sizeof(T))
        << "Requested size is too large to fit into size_t.";
    if (arena == NULL) {
      return static_cast<T*>(::operator new[](num_elements * sizeof(T)));
    } else {
      return arena->CreateInternalRawArray<T>(num_elements);
    }
  }

  // Returns the total space used by the arena, which is the sums of the sizes
  // of the underlying blocks. The total space used may not include the new
  // blocks that are allocated by this arena from other threads concurrently
  // with the call to this method.
  GOOGLE_ATTRIBUTE_NOINLINE uint64 SpaceAllocated() const;
  // As above, but does not include any free space in underlying blocks.
  GOOGLE_ATTRIBUTE_NOINLINE uint64 SpaceUsed() const;

  // Frees all storage allocated by this arena after calling destructors
  // registered with OwnDestructor() and freeing objects registered with Own().
  // Any objects allocated on this arena are unusable after this call. It also
  // returns the total space used by the arena which is the sums of the sizes
  // of the allocated blocks. This method is not thread-safe.
  GOOGLE_ATTRIBUTE_NOINLINE uint64 Reset();

  // Adds |object| to a list of heap-allocated objects to be freed with |delete|
  // when the arena is destroyed or reset.
  template <typename T> GOOGLE_ATTRIBUTE_NOINLINE
  void Own(T* object) {
    OwnInternal(object, google::protobuf::internal::is_convertible<T*, ::google::protobuf::Message*>());
  }

  // Adds |object| to a list of objects whose destructors will be manually
  // called when the arena is destroyed or reset. This differs from Own() in
  // that it does not free the underlying memory with |delete|; hence, it is
  // normally only used for objects that are placement-newed into
  // arena-allocated memory.
  template <typename T> GOOGLE_ATTRIBUTE_NOINLINE
  void OwnDestructor(T* object) {
    if (object != NULL) {
      AddListNode(object, &internal::arena_destruct_object<T>);
    }
  }

  // Adds a custom member function on an object to the list of destructors that
  // will be manually called when the arena is destroyed or reset. This differs
  // from OwnDestructor() in that any member function may be specified, not only
  // the class destructor.
  GOOGLE_ATTRIBUTE_NOINLINE void OwnCustomDestructor(void* object,
                                              void (*destruct)(void*)) {
    AddListNode(object, destruct);
  }

  // Retrieves the arena associated with |value| if |value| is an arena-capable
  // message, or NULL otherwise. This differs from value->GetArena() in that the
  // latter is a virtual call, while this method is a templated call that
  // resolves at compile-time.
  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static ::google::protobuf::Arena* GetArena(const T* value) {
    return GetArenaInternal(value, static_cast<T*>(0));
  }

  // Helper typetrait that indicates support for arenas in a type T at compile
  // time. This is public only to allow construction of higher-level templated
  // utilities. is_arena_constructable<T>::value is an instance of
  // google::protobuf::internal::true_type if the message type T has arena support enabled, and
  // google::protobuf::internal::false_type otherwise.
  //
  // This is inside Arena because only Arena has the friend relationships
  // necessary to see the underlying generated code traits.
  template<typename T>
  struct is_arena_constructable {
    template<typename U>
    static char ArenaConstructable(
        const typename U::InternalArenaConstructable_*);
    template<typename U>
    static double ArenaConstructable(...);

    // This will resolve to either google::protobuf::internal::true_type or google::protobuf::internal::false_type.
    typedef google::protobuf::internal::integral_constant<bool,
              sizeof(ArenaConstructable<const T>(static_cast<const T*>(0))) ==
              sizeof(char)> type;
    static const type value;
  };

 private:
  // Blocks are variable length malloc-ed objects.  The following structure
  // describes the common header for all blocks.
  struct Block {
    void* owner;   // &ThreadCache of thread that owns this block, or
                   // &this->owner if not yet owned by a thread.
    Block* next;   // Next block in arena (may have different owner)
    // ((char*) &block) + pos is next available byte. It is always
    // aligned at a multiple of 8 bytes.
    size_t pos;
    size_t size;  // total size of the block.
    GOOGLE_ATTRIBUTE_ALWAYS_INLINE size_t avail() const { return size - pos; }
    // data follows
  };

  template<typename Type> friend class ::google::protobuf::internal::GenericTypeHandler;
  friend class MockArena;              // For unit-testing.
  friend class internal::ArenaString;  // For AllocateAligned.
  friend class internal::LazyField;    // For CreateMaybeMessage.

  struct ThreadCache {
    // The ThreadCache is considered valid as long as this matches the
    // lifecycle_id of the arena being used.
    int64 last_lifecycle_id_seen;
    Block* last_block_used_;
  };

  static const size_t kHeaderSize = sizeof(Block);
  static google::protobuf::internal::SequenceNumber lifecycle_id_generator_;
#if defined(GOOGLE_PROTOBUF_NO_THREADLOCAL)
  // Android ndk does not support GOOGLE_THREAD_LOCAL keyword so we use a custom thread
  // local storage class we implemented.
  // iOS also does not support the GOOGLE_THREAD_LOCAL keyword.
  static ThreadCache& thread_cache();
#elif defined(PROTOBUF_USE_DLLS)
  // Thread local variables cannot be exposed through DLL interface but we can
  // wrap them in static functions.
  static ThreadCache& thread_cache();
#else
  static GOOGLE_THREAD_LOCAL ThreadCache thread_cache_;
  static ThreadCache& thread_cache() { return thread_cache_; }
#endif

  // SFINAE for skipping addition to delete list for a message type when created
  // with CreateMessage. This is mainly to skip proto2/proto1 message objects
  // with cc_enable_arenas=true from being part of the delete list. Also, note,
  // compiler will optimize out the branch in CreateInternal<T>.
  template<typename T>
  static inline bool SkipDeleteList(typename T::DestructorSkippable_*) {
    return true;
  }

  // For message objects that don't have the DestructorSkippable_ trait, we
  // always add to the delete list.
  template<typename T>
  static inline bool SkipDeleteList(...) {
    return google::protobuf::internal::has_trivial_destructor<T>::value;
  }

  // Helper typetrait that indicates whether the desctructor of type T should be
  // called when arena is destroyed at compile time. This is only to allow
  // construction of higher-level templated utilities.
  // is_destructor_skippable<T>::value is an instance of google::protobuf::internal::true_type if the
  // destructor of the message type T should not be called when arena is
  // destroyed or google::protobuf::internal::has_trivial_destructor<T>::value == true, and
  // google::protobuf::internal::false_type otherwise.
  //
  // This is inside Arena because only Arena has the friend relationships
  // necessary to see the underlying generated code traits.
  template<typename T>
  struct is_destructor_skippable {
    template<typename U>
    static char DestructorSkippable(
        const typename U::DestructorSkippable_*);
    template<typename U>
    static double DestructorSkippable(...);

    // The raw_skippable_value const bool variable is separated from the typedef
    // line below as a work-around of an NVCC 7.0 (and earlier) compiler bug.
    static const bool raw_skippable_value =
          sizeof(DestructorSkippable<const T>(static_cast<const T*>(0))) ==
          sizeof(char) || google::protobuf::internal::has_trivial_destructor<T>::value == true;
    // This will resolve to either google::protobuf::internal::true_type or google::protobuf::internal::false_type.
    typedef google::protobuf::internal::integral_constant<bool, raw_skippable_value> type;
    static const type value;
  };


  // CreateMessage<T> requires that T supports arenas, but this private method
  // works whether or not T supports arenas. These are not exposed to user code
  // as it can cause confusing API usages, and end up having double free in
  // user code. These are used only internally from LazyField and Repeated
  // fields, since they are designed to work in all mode combinations.
  template<typename Msg> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static Msg* CreateMaybeMessage(
      Arena* arena, typename Msg::InternalArenaConstructable_*) {
    return CreateMessage<Msg>(arena);
  }

  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static T* CreateMaybeMessage(Arena* arena, ...) {
    return Create<T>(arena);
  }

  // Just allocate the required size for the given type assuming the
  // type has a trivial constructor.
  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateInternalRawArray(size_t num_elements) {
    GOOGLE_CHECK_LE(num_elements,
             std::numeric_limits<size_t>::max() / sizeof(T))
        << "Requested size is too large to fit into size_t.";
    return static_cast<T*>(
        AllocateAligned(RTTI_TYPE_ID(T), sizeof(T) * num_elements));
  }

  template <typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateInternal(bool skip_explicit_ownership) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T))) T();
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateInternal(bool skip_explicit_ownership, const Arg& arg) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T))) T(arg);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateInternal(
      bool skip_explicit_ownership, const Arg1& arg1, const Arg2& arg2) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T))) T(arg1, arg2);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE T* CreateInternal(bool skip_explicit_ownership,
                                            const Arg1& arg1,
                                            const Arg2& arg2,
                                            const Arg3& arg3) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T)))
        T(arg1, arg2, arg3);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE T* CreateInternal(bool skip_explicit_ownership,
                                            const Arg1& arg1,
                                            const Arg2& arg2,
                                            const Arg3& arg3,
                                            const Arg4& arg4) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T)))
        T(arg1, arg2, arg3, arg4);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE T* CreateInternal(bool skip_explicit_ownership,
                                            const Arg1& arg1,
                                            const Arg2& arg2,
                                            const Arg3& arg3,
                                            const Arg4& arg4,
                                            const Arg5& arg5) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T)))
        T(arg1, arg2, arg3, arg4, arg5);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5, typename Arg6>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE T* CreateInternal(bool skip_explicit_ownership,
                                            const Arg1& arg1,
                                            const Arg2& arg2,
                                            const Arg3& arg3,
                                            const Arg4& arg4,
                                            const Arg5& arg5,
                                            const Arg6& arg6) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T)))
        T(arg1, arg2, arg3, arg4, arg5, arg6);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE T* CreateInternal(bool skip_explicit_ownership,
                                            const Arg1& arg1,
                                            const Arg2& arg2,
                                            const Arg3& arg3,
                                            const Arg4& arg4,
                                            const Arg5& arg5,
                                            const Arg6& arg6,
                                            const Arg7& arg7) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T)))
        T(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T, typename Arg1, typename Arg2, typename Arg3,
            typename Arg4, typename Arg5, typename Arg6, typename Arg7,
            typename Arg8>
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE T* CreateInternal(bool skip_explicit_ownership,
                                            const Arg1& arg1,
                                            const Arg2& arg2,
                                            const Arg3& arg3,
                                            const Arg4& arg4,
                                            const Arg5& arg5,
                                            const Arg6& arg6,
                                            const Arg7& arg7,
                                            const Arg8& arg8) {
    T* t = new (AllocateAligned(RTTI_TYPE_ID(T), sizeof(T)))
        T(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    if (!skip_explicit_ownership) {
      AddListNode(t, &internal::arena_destruct_object<T>);
    }
    return t;
  }

  template <typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateMessageInternal(typename T::InternalArenaConstructable_*) {
    return CreateInternal<T, Arena*>(SkipDeleteList<T>(static_cast<T*>(0)),
                                     this);
  }

  template <typename T, typename Arg> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateMessageInternal(typename T::InternalArenaConstructable_*,
                           const Arg& arg) {
    return CreateInternal<T, Arena*>(SkipDeleteList<T>(static_cast<T*>(0)),
                                     this, arg);
  }

  template <typename T, typename Arg1, typename Arg2> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  T* CreateMessageInternal(typename T::InternalArenaConstructable_*,
                           const Arg1& arg1, const Arg2& arg2) {
    return CreateInternal<T, Arena*>(SkipDeleteList<T>(static_cast<T*>(0)),
                                     this, arg1, arg2);
  }

  // CreateInArenaStorage is used to implement map field. Without it,
  // google::protobuf::Map need to call generated message's protected arena constructor,
  // which needs to declare google::protobuf::Map as friend of generated message.
  template <typename T>
  static void CreateInArenaStorage(T* ptr, Arena* arena) {
    CreateInArenaStorageInternal(ptr, arena, is_arena_constructable<T>::value);
    RegisterDestructorInternal(ptr, arena, is_destructor_skippable<T>::value);
  }

  template <typename T>
  static void CreateInArenaStorageInternal(
      T* ptr, Arena* arena, google::protobuf::internal::true_type) {
    new (ptr) T(arena);
  }
  template <typename T>
  static void CreateInArenaStorageInternal(
      T* ptr, Arena* arena, google::protobuf::internal::false_type) {
    new (ptr) T;
  }

  template <typename T>
  static void RegisterDestructorInternal(
      T* ptr, Arena* arena, google::protobuf::internal::true_type) {}
  template <typename T>
  static void RegisterDestructorInternal(
      T* ptr, Arena* arena, google::protobuf::internal::false_type) {
    arena->OwnDestructor(ptr);
  }

  // These implement Own(), which registers an object for deletion (destructor
  // call and operator delete()). The second parameter has type 'true_type' if T
  // is a subtype of ::google::protobuf::Message and 'false_type' otherwise. Collapsing
  // all template instantiations to one for generic Message reduces code size,
  // using the virtual destructor instead.
  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  void OwnInternal(T* object, google::protobuf::internal::true_type) {
    if (object != NULL) {
      AddListNode(object, &internal::arena_delete_object< ::google::protobuf::Message >);
    }
  }
  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  void OwnInternal(T* object, google::protobuf::internal::false_type) {
    if (object != NULL) {
      AddListNode(object, &internal::arena_delete_object<T>);
    }
  }

  // Implementation for GetArena(). Only message objects with
  // InternalArenaConstructable_ tags can be associated with an arena, and such
  // objects must implement a GetArenaNoVirtual() method.
  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static ::google::protobuf::Arena* GetArenaInternal(
      const T* value, typename T::InternalArenaConstructable_*) {
    return value->GetArenaNoVirtual();
  }

  template<typename T> GOOGLE_ATTRIBUTE_ALWAYS_INLINE
  static ::google::protobuf::Arena* GetArenaInternal(const T* value, ...) {
    return NULL;
  }

  // Allocate and also optionally call on_arena_allocation callback with the
  // allocated type info when the hooks are in place in ArenaOptions and
  // the cookie is not null.
  void* AllocateAligned(const std::type_info* allocated, size_t n);

  // Allocate an internal allocation, avoiding optional typed monitoring.
  GOOGLE_ATTRIBUTE_ALWAYS_INLINE void* AllocateAligned(size_t n) {
    return AllocateAligned(NULL, n);
  }

  void Init();

  // Free all blocks and return the total space used which is the sums of sizes
  // of the all the allocated blocks.
  uint64 FreeBlocks();

  // Add object pointer and cleanup function pointer to the list.
  // TODO(rohananil, cfallin): We could pass in a sub-arena into this method
  // to avoid polluting blocks of this arena with list nodes. This would help in
  // mixed mode (where many protobufs have cc_enable_arenas=false), and is an
  // alternative to a chunked linked-list, but with extra overhead of *next.
  void AddListNode(void* elem, void (*cleanup)(void*));
  // Delete or Destruct all objects owned by the arena.
  void CleanupList();
  uint64 ResetInternal();

  inline void SetThreadCacheBlock(Block* block) {
    thread_cache().last_block_used_ = block;
    thread_cache().last_lifecycle_id_seen = lifecycle_id_;
  }

  int64 lifecycle_id_;  // Unique for each arena. Changes on Reset().

  google::protobuf::internal::AtomicWord blocks_;  // Head of linked list of all allocated blocks
  google::protobuf::internal::AtomicWord hint_;    // Fast thread-local block access

  // Node contains the ptr of the object to be cleaned up and the associated
  // cleanup function ptr.
  struct Node {
    void* elem;              // Pointer to the object to be cleaned up.
    void (*cleanup)(void*);  // Function pointer to the destructor or deleter.
    Node* next;              // Next node in the list.
  };

  google::protobuf::internal::AtomicWord cleanup_list_;  // Head of a linked list of nodes containing object
                             // ptrs and cleanup methods.

  bool owns_first_block_;    // Indicates that arena owns the first block
  Mutex blocks_lock_;

  void AddBlock(Block* b);
  // Access must be synchronized, either by blocks_lock_ or by being called from
  // Init()/Reset().
  void AddBlockInternal(Block* b);
  void* SlowAlloc(size_t n);
  Block* FindBlock(void* me);
  Block* NewBlock(void* me, Block* my_last_block, size_t n,
                  size_t start_block_size, size_t max_block_size);
  static void* AllocFromBlock(Block* b, size_t n);
  template <typename Key, typename T>
  friend class Map;

  // The arena may save a cookie it receives from the external on_init hook
  // and then use it when calling the on_reset and on_destruction hooks.
  void* hooks_cookie_;

  ArenaOptions options_;

  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(Arena);
};

// Defined above for supporting environments without RTTI.
#undef RTTI_TYPE_ID

template<typename T>
const typename Arena::is_arena_constructable<T>::type
    Arena::is_arena_constructable<T>::value =
        typename Arena::is_arena_constructable<T>::type();

template<typename T>
const typename Arena::is_destructor_skippable<T>::type
    Arena::is_destructor_skippable<T>::value =
        typename Arena::is_destructor_skippable<T>::type();

}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_ARENA_H__
