#ifndef SLANG_COM_PTR_H
#define SLANG_COM_PTR_H

#include "slang-com-helper.h"

#include <assert.h>
#include <cstddef>

namespace Slang
{

/*! \brief ComPtr is a simple smart pointer that manages types which implement COM based interfaces.
\details A class that implements a COM, must derive from the IUnknown interface or a type that
matches it's layout exactly (such as ISlangUnknown). Trying to use this template with a class that
doesn't follow these rules, will lead to undefined behavior. This is a 'strong' pointer type, and
will AddRef when a non null pointer is set and Release when the pointer leaves scope. Using 'detach'
allows a pointer to be removed from the management of the ComPtr. To set the smart pointer to null,
there is the method setNull, or alternatively just assign SLANG_NULL/nullptr.

One edge case using the template is that sometimes you want access as a pointer to a pointer.
Sometimes this is to write into the smart pointer, other times to pass as an array. To handle these
different behaviors there are the methods readRef and writeRef, which are used instead of the &
(ref) operator. For example

\code
Void doSomething(ID3D12Resource** resources, IndexT numResources);
// ...
ComPtr<ID3D12Resource> resources[3];
doSomething(resources[0].readRef(), SLANG_COUNT_OF(resource));
\endcode

A more common scenario writing to the pointer

\code
IUnknown* unk = ...;

ComPtr<ID3D12Resource> resource;
Result res = unk->QueryInterface(resource.writeRef());
\endcode
*/

// Enum to force initializing as an attach (without adding a reference)
enum InitAttach
{
    INIT_ATTACH
};

template<class T>
class ComPtr
{
public:
    typedef T Type;
    typedef ComPtr ThisType;
    typedef ISlangUnknown* Ptr;

    /// Constructors
    /// Default Ctor. Sets to nullptr
    SLANG_FORCE_INLINE ComPtr()
        : m_ptr(nullptr)
    {
    }
    SLANG_FORCE_INLINE ComPtr(std::nullptr_t)
        : m_ptr(nullptr)
    {
    }
    /// Sets, and ref counts.
    SLANG_FORCE_INLINE explicit ComPtr(T* ptr)
        : m_ptr(ptr)
    {
        if (ptr)
            ((Ptr)ptr)->addRef();
    }
    /// The copy ctor
    SLANG_FORCE_INLINE ComPtr(const ThisType& rhs)
        : m_ptr(rhs.m_ptr)
    {
        if (m_ptr)
            ((Ptr)m_ptr)->addRef();
    }

    /// Ctor without adding to ref count.
    SLANG_FORCE_INLINE explicit ComPtr(InitAttach, T* ptr)
        : m_ptr(ptr)
    {
    }
    /// Ctor without adding to ref count
    SLANG_FORCE_INLINE ComPtr(InitAttach, const ThisType& rhs)
        : m_ptr(rhs.m_ptr)
    {
    }

#ifdef SLANG_HAS_MOVE_SEMANTICS
    /// Move Ctor
    SLANG_FORCE_INLINE ComPtr(ThisType&& rhs)
        : m_ptr(rhs.m_ptr)
    {
        rhs.m_ptr = nullptr;
    }
    /// Move assign
    SLANG_FORCE_INLINE ComPtr& operator=(ThisType&& rhs)
    {
        T* swap = m_ptr;
        m_ptr = rhs.m_ptr;
        rhs.m_ptr = swap;
        return *this;
    }
#endif

    /// Destructor releases the pointer, assuming it is set
    SLANG_FORCE_INLINE ~ComPtr()
    {
        if (m_ptr)
            ((Ptr)m_ptr)->release();
    }

    // !!! Operators !!!

    /// Returns the dumb pointer
    SLANG_FORCE_INLINE operator T*() const { return m_ptr; }

    SLANG_FORCE_INLINE T& operator*() { return *m_ptr; }
    /// For making method invocations through the smart pointer work through the dumb pointer
    SLANG_FORCE_INLINE T* operator->() const { return m_ptr; }

    /// Assign
    SLANG_FORCE_INLINE const ThisType& operator=(const ThisType& rhs);
    /// Assign from dumb ptr
    SLANG_FORCE_INLINE T* operator=(T* in);

    /// Get the pointer and don't ref
    SLANG_FORCE_INLINE T* get() const { return m_ptr; }
    /// Release a contained nullptr pointer if set
    SLANG_FORCE_INLINE void setNull();

    /// Detach
    SLANG_FORCE_INLINE T* detach()
    {
        T* ptr = m_ptr;
        m_ptr = nullptr;
        return ptr;
    }
    /// Set to a pointer without changing the ref count
    SLANG_FORCE_INLINE void attach(T* in) { m_ptr = in; }

    /// Get ready for writing (nulls contents)
    SLANG_FORCE_INLINE T** writeRef()
    {
        setNull();
        return &m_ptr;
    }
    /// Get for read access
    SLANG_FORCE_INLINE T* const* readRef() const { return &m_ptr; }

    /// Swap
    void swap(ThisType& rhs);

protected:
    /// Gets the address of the dumb pointer.
    // Disabled: use writeRef and readRef to get a reference based on usage.
#ifndef SLANG_COM_PTR_ENABLE_REF_OPERATOR
    SLANG_FORCE_INLINE T** operator&() = delete;
#endif

    T* m_ptr;
};

//----------------------------------------------------------------------------
template<typename T>
void ComPtr<T>::setNull()
{
    if (m_ptr)
    {
        ((Ptr)m_ptr)->release();
        m_ptr = nullptr;
    }
}
//----------------------------------------------------------------------------
template<typename T>
const ComPtr<T>& ComPtr<T>::operator=(const ThisType& rhs)
{
    if (rhs.m_ptr)
        ((Ptr)rhs.m_ptr)->addRef();
    if (m_ptr)
        ((Ptr)m_ptr)->release();
    m_ptr = rhs.m_ptr;
    return *this;
}
//----------------------------------------------------------------------------
template<typename T>
T* ComPtr<T>::operator=(T* ptr)
{
    if (ptr)
        ((Ptr)ptr)->addRef();
    if (m_ptr)
        ((Ptr)m_ptr)->release();
    m_ptr = ptr;
    return m_ptr;
}
//----------------------------------------------------------------------------
template<typename T>
void ComPtr<T>::swap(ThisType& rhs)
{
    T* tmp = m_ptr;
    m_ptr = rhs.m_ptr;
    rhs.m_ptr = tmp;
}

} // namespace Slang

#endif // SLANG_COM_PTR_H
