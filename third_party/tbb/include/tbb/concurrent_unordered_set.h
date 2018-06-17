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

/* Container implementations in this header are based on PPL implementations
   provided by Microsoft. */

#ifndef __TBB_concurrent_unordered_set_H
#define __TBB_concurrent_unordered_set_H

#include "internal/_concurrent_unordered_impl.h"

namespace tbb
{

namespace interface5 {

// Template class for hash set traits
template<typename Key, typename Hash_compare, typename Allocator, bool Allow_multimapping>
class concurrent_unordered_set_traits
{
protected:
    typedef Key value_type;
    typedef Key key_type;
    typedef Hash_compare hash_compare;
    typedef typename Allocator::template rebind<value_type>::other allocator_type;
    enum { allow_multimapping = Allow_multimapping };

    concurrent_unordered_set_traits() : my_hash_compare() {}
    concurrent_unordered_set_traits(const hash_compare& hc) : my_hash_compare(hc) {}

    static const Key& get_key(const value_type& value) {
        return value;
    }

    hash_compare my_hash_compare; // the comparator predicate for keys
};

template <typename Key, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>, typename Allocator = tbb::tbb_allocator<Key> >
class concurrent_unordered_set : public internal::concurrent_unordered_base< concurrent_unordered_set_traits<Key, internal::hash_compare<Key, Hasher, Key_equality>, Allocator, false> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_set_traits<Key, hash_compare, Allocator, false> traits_type;
    typedef internal::concurrent_unordered_base< traits_type > base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef Key mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;

    // Construction/destruction/copying
    explicit concurrent_unordered_set(size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {}

    explicit concurrent_unordered_set(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {}

    template <typename Iterator>
    concurrent_unordered_set(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_set(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        this->insert(il.begin(),il.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
#if !__TBB_IMPLICIT_MOVE_PRESENT
    concurrent_unordered_set(const concurrent_unordered_set& table)
        : base_type(table)
    {}

    concurrent_unordered_set& operator=(const concurrent_unordered_set& table)
    {
        return static_cast<concurrent_unordered_set&>(base_type::operator=(table));
    }

    concurrent_unordered_set(concurrent_unordered_set&& table)
        : base_type(std::move(table))
    {}

    concurrent_unordered_set& operator=(concurrent_unordered_set&& table)
    {
        return static_cast<concurrent_unordered_set&>(base_type::operator=(std::move(table)));
    }
#endif //!__TBB_IMPLICIT_MOVE_PRESENT

    concurrent_unordered_set(concurrent_unordered_set&& table, const Allocator& a)
        : base_type(std::move(table), a)
    {}
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    concurrent_unordered_set(const concurrent_unordered_set& table, const Allocator& a)
        : base_type(table, a)
    {}

};

template <typename Key, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>,
         typename Allocator = tbb::tbb_allocator<Key> >
class concurrent_unordered_multiset :
    public internal::concurrent_unordered_base< concurrent_unordered_set_traits<Key,
    internal::hash_compare<Key, Hasher, Key_equality>, Allocator, true> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_set_traits<Key, hash_compare, Allocator, true> traits_type;
    typedef internal::concurrent_unordered_base< traits_type > base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef Key mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;

    // Construction/destruction/copying
    explicit concurrent_unordered_multiset(size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {}

    explicit concurrent_unordered_multiset(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {}

    template <typename Iterator>
    concurrent_unordered_multiset(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& _Hasher = hasher(), const key_equal& _Key_equality = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(_Hasher, _Key_equality), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_multiset(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number, const hasher& a_hasher = hasher(),
        const key_equal& a_keyeq = key_equal(), const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        this->insert(il.begin(),il.end());
    }
#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
#if !__TBB_IMPLICIT_MOVE_PRESENT
    concurrent_unordered_multiset(const concurrent_unordered_multiset& table)
        : base_type(table)
    {}

    concurrent_unordered_multiset& operator=(const concurrent_unordered_multiset& table)
    {
        return static_cast<concurrent_unordered_multiset&>(base_type::operator=(table));
    }

    concurrent_unordered_multiset(concurrent_unordered_multiset&& table)
        : base_type(std::move(table))
    {}

    concurrent_unordered_multiset& operator=(concurrent_unordered_multiset&& table)
    {
        return static_cast<concurrent_unordered_multiset&>(base_type::operator=(std::move(table)));
    }
#endif //!__TBB_IMPLICIT_MOVE_PRESENT

    concurrent_unordered_multiset(concurrent_unordered_multiset&& table, const Allocator& a)
        : base_type(std::move(table), a)
    {
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT

    concurrent_unordered_multiset(const concurrent_unordered_multiset& table, const Allocator& a)
        : base_type(table, a)
    {}
};
} // namespace interface5

using interface5::concurrent_unordered_set;
using interface5::concurrent_unordered_multiset;

} // namespace tbb

#endif// __TBB_concurrent_unordered_set_H
