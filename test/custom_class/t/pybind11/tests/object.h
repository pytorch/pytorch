#if !defined(__OBJECT_H)
#define __OBJECT_H

#include <atomic>
#include "constructor_stats.h"

/// Reference counted object base class
class Object {
public:
    /// Default constructor
    Object() { print_default_created(this); }

    /// Copy constructor
    Object(const Object &) : m_refCount(0) { print_copy_created(this); }

    /// Return the current reference count
    int getRefCount() const { return m_refCount; };

    /// Increase the object's reference count by one
    void incRef() const { ++m_refCount; }

    /** \brief Decrease the reference count of
     * the object and possibly deallocate it.
     *
     * The object will automatically be deallocated once
     * the reference count reaches zero.
     */
    void decRef(bool dealloc = true) const {
        --m_refCount;
        if (m_refCount == 0 && dealloc)
            delete this;
        else if (m_refCount < 0)
            throw std::runtime_error("Internal error: reference count < 0!");
    }

    virtual std::string toString() const = 0;
protected:
    /** \brief Virtual protected deconstructor.
     * (Will only be called by \ref ref)
     */
    virtual ~Object() { print_destroyed(this); }
private:
    mutable std::atomic<int> m_refCount { 0 };
};

// Tag class used to track constructions of ref objects.  When we track constructors, below, we
// track and print out the actual class (e.g. ref<MyObject>), and *also* add a fake tracker for
// ref_tag.  This lets us check that the total number of ref<Anything> constructors/destructors is
// correct without having to check each individual ref<Whatever> type individually.
class ref_tag {};

/**
 * \brief Reference counting helper
 *
 * The \a ref refeference template is a simple wrapper to store a
 * pointer to an object. It takes care of increasing and decreasing
 * the reference count of the object. When the last reference goes
 * out of scope, the associated object will be deallocated.
 *
 * \ingroup libcore
 */
template <typename T> class ref {
public:
    /// Create a nullptr reference
    ref() : m_ptr(nullptr) { print_default_created(this); track_default_created((ref_tag*) this); }

    /// Construct a reference from a pointer
    ref(T *ptr) : m_ptr(ptr) {
        if (m_ptr) ((Object *) m_ptr)->incRef();

        print_created(this, "from pointer", m_ptr); track_created((ref_tag*) this, "from pointer");

    }

    /// Copy constructor
    ref(const ref &r) : m_ptr(r.m_ptr) {
        if (m_ptr)
            ((Object *) m_ptr)->incRef();

        print_copy_created(this, "with pointer", m_ptr); track_copy_created((ref_tag*) this);
    }

    /// Move constructor
    ref(ref &&r) : m_ptr(r.m_ptr) {
        r.m_ptr = nullptr;

        print_move_created(this, "with pointer", m_ptr); track_move_created((ref_tag*) this);
    }

    /// Destroy this reference
    ~ref() {
        if (m_ptr)
            ((Object *) m_ptr)->decRef();

        print_destroyed(this); track_destroyed((ref_tag*) this);
    }

    /// Move another reference into the current one
    ref& operator=(ref&& r) {
        print_move_assigned(this, "pointer", r.m_ptr); track_move_assigned((ref_tag*) this);

        if (*this == r)
            return *this;
        if (m_ptr)
            ((Object *) m_ptr)->decRef();
        m_ptr = r.m_ptr;
        r.m_ptr = nullptr;
        return *this;
    }

    /// Overwrite this reference with another reference
    ref& operator=(const ref& r) {
        print_copy_assigned(this, "pointer", r.m_ptr); track_copy_assigned((ref_tag*) this);

        if (m_ptr == r.m_ptr)
            return *this;
        if (m_ptr)
            ((Object *) m_ptr)->decRef();
        m_ptr = r.m_ptr;
        if (m_ptr)
            ((Object *) m_ptr)->incRef();
        return *this;
    }

    /// Overwrite this reference with a pointer to another object
    ref& operator=(T *ptr) {
        print_values(this, "assigned pointer"); track_values((ref_tag*) this, "assigned pointer");

        if (m_ptr == ptr)
            return *this;
        if (m_ptr)
            ((Object *) m_ptr)->decRef();
        m_ptr = ptr;
        if (m_ptr)
            ((Object *) m_ptr)->incRef();
        return *this;
    }

    /// Compare this reference with another reference
    bool operator==(const ref &r) const { return m_ptr == r.m_ptr; }

    /// Compare this reference with another reference
    bool operator!=(const ref &r) const { return m_ptr != r.m_ptr; }

    /// Compare this reference with a pointer
    bool operator==(const T* ptr) const { return m_ptr == ptr; }

    /// Compare this reference with a pointer
    bool operator!=(const T* ptr) const { return m_ptr != ptr; }

    /// Access the object referenced by this reference
    T* operator->() { return m_ptr; }

    /// Access the object referenced by this reference
    const T* operator->() const { return m_ptr; }

    /// Return a C++ reference to the referenced object
    T& operator*() { return *m_ptr; }

    /// Return a const C++ reference to the referenced object
    const T& operator*() const { return *m_ptr; }

    /// Return a pointer to the referenced object
    operator T* () { return m_ptr; }

    /// Return a const pointer to the referenced object
    T* get_ptr() { return m_ptr; }

    /// Return a pointer to the referenced object
    const T* get_ptr() const { return m_ptr; }
private:
    T *m_ptr;
};

#endif /* __OBJECT_H */
