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

#ifndef _TBB_intrusive_list_H
#define _TBB_intrusive_list_H

#include "tbb/tbb_stddef.h"

namespace tbb {
namespace internal {

//! Data structure to be inherited by the types that can form intrusive lists.
/** Intrusive list is formed by means of the member_intrusive_list<T> template class.
    Note that type T must derive from intrusive_list_node either publicly or
    declare instantiation member_intrusive_list<T> as a friend.
    This class implements a limited subset of std::list interface. **/
struct intrusive_list_node {
    intrusive_list_node *my_prev_node,
                        *my_next_node;
#if TBB_USE_ASSERT
    intrusive_list_node () { my_prev_node = my_next_node = this; }
#endif /* TBB_USE_ASSERT */
};

//! List of element of type T, where T is derived from intrusive_list_node
/** The class is not thread safe. **/
template <class List, class T>
class intrusive_list_base {
    //! Pointer to the head node
    intrusive_list_node my_head;

    //! Number of list elements
    size_t my_size;

    static intrusive_list_node& node ( T& item ) { return List::node(item); }

    static T& item ( intrusive_list_node* node ) { return List::item(node); }

    template<class Iterator>
    class iterator_impl {
        Iterator& self () { return *static_cast<Iterator*>(this); }

        //! Node the iterator points to at the moment
        intrusive_list_node *my_pos;

    protected:
        iterator_impl (intrusive_list_node* pos )
            :  my_pos(pos)
        {}

        T& item () const {
            return intrusive_list_base::item(my_pos);
        }

    public:
        iterator_impl () :  my_pos(NULL) {}

        Iterator& operator = ( const Iterator& it ) {
            return my_pos = it.my_pos;
        }

        Iterator& operator = ( const T& val ) {
            return my_pos = &node(val);
        }

        bool operator == ( const Iterator& it ) const {
            return my_pos == it.my_pos;
        }

        bool operator != ( const Iterator& it ) const {
            return my_pos != it.my_pos;
        }

        Iterator& operator++ () {
            my_pos = my_pos->my_next_node;
            return self();
        }

        Iterator& operator-- () {
            my_pos = my_pos->my_prev_node;
            return self();
        }

        Iterator operator++ ( int ) {
            Iterator result = self();
            ++(*this);
            return result;
        }

        Iterator operator-- ( int ) {
            Iterator result = self();
            --(*this);
            return result;
        }
    }; // intrusive_list_base::iterator_impl

    void assert_ok () const {
        __TBB_ASSERT( (my_head.my_prev_node == &my_head && !my_size) ||
                      (my_head.my_next_node != &my_head && my_size >0), "intrusive_list_base corrupted" );
#if TBB_USE_ASSERT >= 2
        size_t i = 0;
        for ( intrusive_list_node *n = my_head.my_next_node; n != &my_head; n = n->my_next_node )
            ++i;
        __TBB_ASSERT( my_size == i, "Wrong size" );
#endif /* TBB_USE_ASSERT >= 2 */
    }

public:
    class iterator : public iterator_impl<iterator> {
        template <class U, class V> friend class intrusive_list_base;
    public:
        iterator (intrusive_list_node* pos )
            : iterator_impl<iterator>(pos )
        {}
        iterator () {}

        T* operator-> () const { return &this->item(); }

        T& operator* () const { return this->item(); }
    }; // class iterator

    class const_iterator : public iterator_impl<const_iterator> {
        template <class U, class V> friend class intrusive_list_base;
    public:
        const_iterator (const intrusive_list_node* pos )
            : iterator_impl<const_iterator>(const_cast<intrusive_list_node*>(pos) )
        {}
        const_iterator () {}

        const T* operator-> () const { return &this->item(); }

        const T& operator* () const { return this->item(); }
    }; // class iterator

    intrusive_list_base () : my_size(0) {
        my_head.my_prev_node = &my_head;
        my_head.my_next_node = &my_head;
    }

    bool empty () const { return my_head.my_next_node == &my_head; }

    size_t size () const { return my_size; }

    iterator begin () { return iterator(my_head.my_next_node); }

    iterator end () { return iterator(&my_head); }

    const_iterator begin () const { return const_iterator(my_head.my_next_node); }

    const_iterator end () const { return const_iterator(&my_head); }

    void push_front ( T& val ) {
        __TBB_ASSERT( node(val).my_prev_node == &node(val) && node(val).my_next_node == &node(val),
                    "Object with intrusive list node can be part of only one intrusive list simultaneously" );
        // An object can be part of only one intrusive list at the given moment via the given node member
        node(val).my_prev_node = &my_head;
        node(val).my_next_node = my_head.my_next_node;
        my_head.my_next_node->my_prev_node = &node(val);
        my_head.my_next_node = &node(val);
        ++my_size;
        assert_ok();
    }

    void remove( T& val ) {
        __TBB_ASSERT( node(val).my_prev_node != &node(val) && node(val).my_next_node != &node(val), "Element to remove is not in the list" );
        __TBB_ASSERT( node(val).my_prev_node->my_next_node == &node(val) && node(val).my_next_node->my_prev_node == &node(val), "Element to remove is not in the list" );
        --my_size;
        node(val).my_next_node->my_prev_node = node(val).my_prev_node;
        node(val).my_prev_node->my_next_node = node(val).my_next_node;
#if TBB_USE_ASSERT
        node(val).my_prev_node = node(val).my_next_node = &node(val);
#endif
        assert_ok();
    }

    iterator erase ( iterator it ) {
        T& val = *it;
        ++it;
        remove( val );
        return it;
    }

}; // intrusive_list_base


//! Double linked list of items of type T containing a member of type intrusive_list_node.
/** NodePtr is a member pointer to the node data field. Class U is either T or
    a base class of T containing the node member. Default values exist for the sake
    of a partial specialization working with inheritance case.

    The list does not have ownership of its items. Its purpose is to avoid dynamic
    memory allocation when forming lists of existing objects.

    The class is not thread safe. **/
template <class T, class U, intrusive_list_node U::*NodePtr>
class memptr_intrusive_list : public intrusive_list_base<memptr_intrusive_list<T, U, NodePtr>, T>
{
    friend class intrusive_list_base<memptr_intrusive_list<T, U, NodePtr>, T>;

    static intrusive_list_node& node ( T& val ) { return val.*NodePtr; }

    static T& item ( intrusive_list_node* node ) {
        // Cannot use __TBB_offsetof (and consequently __TBB_get_object_ref) macro
        // with *NodePtr argument because gcc refuses to interpret pasted "->" and "*"
        // as member pointer dereferencing operator, and explicit usage of ## in
        // __TBB_offsetof implementation breaks operations with normal member names.
        return *reinterpret_cast<T*>((char*)node - ((ptrdiff_t)&(reinterpret_cast<T*>(0x1000)->*NodePtr) - 0x1000));
    }
}; // intrusive_list<T, U, NodePtr>

//! Double linked list of items of type T that is derived from intrusive_list_node class.
/** The list does not have ownership of its items. Its purpose is to avoid dynamic
    memory allocation when forming lists of existing objects.

    The class is not thread safe. **/
template <class T>
class intrusive_list : public intrusive_list_base<intrusive_list<T>, T>
{
    friend class intrusive_list_base<intrusive_list<T>, T>;

    static intrusive_list_node& node ( T& val ) { return val; }

    static T& item ( intrusive_list_node* node ) { return *static_cast<T*>(node); }
}; // intrusive_list<T>

} // namespace internal
} // namespace tbb

#endif /* _TBB_intrusive_list_H */
