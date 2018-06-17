/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef _TBB_tls_H
#define _TBB_tls_H

#if USE_PTHREAD
#include <pthread.h>
#else /* assume USE_WINTHREAD */
#include "tbb/machine/windows_api.h"
#endif

namespace tbb {

namespace internal {

typedef void (*tls_dtor_t)(void*);

//! Basic cross-platform wrapper class for TLS operations.
template <typename T>
class basic_tls {
#if USE_PTHREAD
    typedef pthread_key_t tls_key_t;
public:
    int  create( tls_dtor_t dtor = NULL ) {
        return pthread_key_create(&my_key, dtor);
    }
    int  destroy()      { return pthread_key_delete(my_key); }
    void set( T value ) { pthread_setspecific(my_key, (void*)value); }
    T    get()          { return (T)pthread_getspecific(my_key); }
#else /* USE_WINTHREAD */
    typedef DWORD tls_key_t;
public:
#if !__TBB_WIN8UI_SUPPORT
    int create() {
        tls_key_t tmp = TlsAlloc();
        if( tmp==TLS_OUT_OF_INDEXES )
            return TLS_OUT_OF_INDEXES;
        my_key = tmp;
        return 0;
    }
    int  destroy()      { TlsFree(my_key); my_key=0; return 0; }
    void set( T value ) { TlsSetValue(my_key, (LPVOID)value); }
    T    get()          { return (T)TlsGetValue(my_key); }
#else /*!__TBB_WIN8UI_SUPPORT*/
    int create() {
        tls_key_t tmp = FlsAlloc(NULL);
        if( tmp== (DWORD)0xFFFFFFFF )
            return (DWORD)0xFFFFFFFF;
        my_key = tmp;
        return 0;
    }
    int  destroy()      { FlsFree(my_key); my_key=0; return 0; }
    void set( T value ) { FlsSetValue(my_key, (LPVOID)value); }
    T    get()          { return (T)FlsGetValue(my_key); }
#endif /* !__TBB_WIN8UI_SUPPORT */
#endif /* USE_WINTHREAD */
private:
    tls_key_t my_key;
};

//! More advanced TLS support template class.
/** It supports RAII and to some extent mimic __declspec(thread) variables. */
template <typename T>
class tls : public basic_tls<T> {
    typedef basic_tls<T> base;
public:
    tls()  { base::create();  }
    ~tls() { base::destroy(); }
    T operator=(T value) { base::set(value); return value; }
    operator T() { return base::get(); }
};

template <typename T>
class tls<T*> : basic_tls<T*> {
    typedef basic_tls<T*> base;
    static void internal_dtor(void* ptr) {
        if (ptr) delete (T*)ptr;
    }
    T* internal_get() {
        T* result = base::get();
        if (!result) {
            result = new T;
            base::set(result);
        }
        return result;
    }
public:
    tls()  {
#if USE_PTHREAD
        base::create( internal_dtor );
#else
        base::create();
#endif
    }
    ~tls() { base::destroy(); }
    T* operator=(T* value) { base::set(value); return value; }
    operator T*()   { return  internal_get(); }
    T* operator->() { return  internal_get(); }
    T& operator*()  { return *internal_get(); }
};

} // namespace internal

} // namespace tbb

#endif /* _TBB_tls_H */
