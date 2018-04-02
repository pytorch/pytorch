#ifndef THC_THREAD_LOCAL_INC
#define THC_THREAD_LOCAL_INC

#ifdef _WIN32
#include <intsafe.h>
typedef DWORD THCThreadLocal;
#else
#include <pthread.h>
typedef pthread_key_t THCThreadLocal;
#endif

THCThreadLocal THCThreadLocal_alloc(void);
void THCThreadLocal_free(THCThreadLocal local);
void* THCThreadLocal_get(THCThreadLocal local);
void THCThreadLocal_set(THCThreadLocal local, void* value);

#endif // THC_THREAD_LOCAL_INC
