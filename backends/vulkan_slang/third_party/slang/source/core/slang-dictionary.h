#ifndef SLANG_CORE_DICTIONARY_H
#define SLANG_CORE_DICTIONARY_H

#include "slang-common.h"
#include "slang-exception.h"
#include "slang-hash.h"
#include "slang-linked-list.h"
#include "slang-list.h"
#include "slang-math.h"
#include "slang-uint-set.h"

#include <ankerl/unordered_dense.h>
#include <initializer_list>

namespace Slang
{
template<typename TKey, typename TValue>
class KeyValuePair
{
public:
    TKey key;
    TValue value;
    KeyValuePair() {}
    KeyValuePair(const TKey& inKey, const TValue& inValue)
    {
        key = inKey;
        value = inValue;
    }
    KeyValuePair(TKey&& inKey, TValue&& inValue)
    {
        key = _Move(inKey);
        value = _Move(inValue);
    }
    KeyValuePair(TKey&& inKey, const TValue& inValue)
    {
        key = _Move(inKey);
        value = inValue;
    }
    KeyValuePair(const KeyValuePair<TKey, TValue>& that)
    {
        key = that.key;
        value = that.value;
    }
    KeyValuePair(KeyValuePair<TKey, TValue>&& that) { operator=(_Move(that)); }
    KeyValuePair& operator=(KeyValuePair<TKey, TValue>&& that)
    {
        key = _Move(that.key);
        value = _Move(that.value);
        return *this;
    }
    KeyValuePair& operator=(const KeyValuePair<TKey, TValue>& that)
    {
        key = that.key;
        value = that.value;
        return *this;
    }
    HashCode getHashCode() const
    {
        return combineHash(Slang::getHashCode(key), Slang::getHashCode(value));
    }
    bool operator==(const KeyValuePair<TKey, TValue>& that) const
    {
        return (key == that.key) && (value == that.value);
    }
};

template<typename TKey, typename TValue>
inline KeyValuePair<TKey, TValue> KVPair(const TKey& k, const TValue& v)
{
    return KeyValuePair<TKey, TValue>(k, v);
}

namespace KeyValueDetail
{

template<typename KEY, typename VALUE>
SLANG_FORCE_INLINE const KEY* getKey(const std::pair<KEY, VALUE>* in)
{
    return &in->first;
}
template<typename KEY, typename VALUE>
SLANG_FORCE_INLINE const KEY* getKey(const KeyValuePair<KEY, VALUE>* in)
{
    return &in->key;
}

template<typename KEY, typename VALUE>
SLANG_FORCE_INLINE const VALUE* getValue(const std::pair<KEY, VALUE>* in)
{
    return &in->second;
}
template<typename KEY, typename VALUE>
SLANG_FORCE_INLINE const VALUE* getValue(const KeyValuePair<KEY, VALUE>* in)
{
    return &in->value;
}

} // namespace KeyValueDetail

const float kMaxLoadFactor = 0.7f;

template<
    typename TKey,
    typename TValue,
    typename Hash = Slang::Hash<TKey>,
    typename KeyEqual = std::equal_to<TKey>>
class Dictionary
{
    using InnerMap = ankerl::unordered_dense::map<TKey, TValue, Hash, KeyEqual>;
    using ThisType = Dictionary<TKey, TValue, Hash, KeyEqual>;
    InnerMap map;

public:
    Dictionary() = default;
    Dictionary(const Dictionary&) = default;
    Dictionary(Dictionary&&) = default;
    ThisType& operator=(const ThisType&) = default;
    ThisType& operator=(ThisType&&) = default;
    Dictionary(std::initializer_list<typename InnerMap::value_type> inits)
        : map(std::move(inits))
    {
    }

    //
    // Types
    //
    using Iterator = typename InnerMap::iterator;
    using ConstIterator = typename InnerMap::const_iterator;
    using KeyType = TKey;
    using ValueType = TValue;

    //
    // Iterators
    //

    auto begin() { return map.begin(); }
    auto begin() const { return map.begin(); }
    auto end() { return map.end(); }
    auto end() const { return map.end(); }

    //
    // Modifiers
    //

    // Removes all values from the map
    void clear() { map.clear(); }

    // Erases the value at the specified key if it exists
    void remove(const TKey& key) { map.erase(key); }

    // Removes all values satifying the predicate:
    // bool predicate(pair<Key, Value>)
    template<typename Predicate>
    void removeIf(Predicate&& predicate)
    {
        auto it = begin();
        while (it != end())
        {
            if (predicate(*it))
            {
                it = map.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    // Reserves enough space for the specified number of values
    void reserve(Index size) { map.reserve(std::size_t(size)); };

    // Swap with another map
    void swapWith(ThisType& rhs) { std::swap(*this, rhs); }

    //
    // Query capacity
    //

    std::size_t getCount() const { return map.size(); }

    //
    // Lookup
    //

    // Returns true if the map contains an equivalent key
    template<typename K>
    bool containsKey(const K& k) const
    {
        return map.contains(k);
    }

    // Returns a valid pointer to the requested element, or nullptr if it
    // doesn't exist
    template<typename K>
    const TValue* tryGetValue(const K& key) const
    {
        auto i = map.find(key);
        return i == map.end() ? nullptr : &(i->second);
    }
    // Returns a valid pointer to the requested element, or nullptr if it
    // doesn't exist
    template<typename K>
    TValue* tryGetValue(const K& key)
    {
        auto i = map.find(key);
        return i == map.end() ? nullptr : std::addressof(i->second);
    }

    // Returns true and copies the element into 'value' if present.
    // Otherwise returns false and value unmodified.
    template<typename K>
    bool tryGetValue(const K& key, TValue& value) const
    {
        auto i = map.find(key);
        if (i == map.end())
            return false;
        value = i->second;
        return true;
    }

    // Returns a const reference to the value at the given key. Asserts if
    // the value doesn't exist
    const TValue& getValue(const TKey& key) const
    {
        if (const auto x = tryGetValue(key))
            return *x;
        SLANG_ASSERT_FAILURE("The key does not exist in dictionary.");
    }

    // Returns a reference to the value at the given key. Asserts if the
    // value doesn't exist
    TValue& getValue(const TKey& key)
    {
        if (const auto x = tryGetValue(key))
            return *x;
        SLANG_ASSERT_FAILURE("The key does not exist in dictionary.");
    }

    //
    // Combined Lookup and Insertion
    //

    // Tries to insert the given element, if a value was already present at
    // the given key then returns a pointer to that element instead.
    // Returns nullptr if insertion was successful.
    TValue* tryGetValueOrAdd(const typename InnerMap::value_type& kvPair)
    {
        const auto& [iterator, inserted] = map.insert(kvPair);
        return inserted ? nullptr : std::addressof(iterator->second);
    }
    // Tries to insert the given element, if a value was already present at
    // the given key then returns a pointer to that element instead.
    // Returns nullptr if insertion was successful.
    TValue* tryGetValueOrAdd(typename InnerMap::value_type&& kvPair)
    {
        const auto& [iterator, inserted] = map.insert(std::move(kvPair));
        return inserted ? nullptr : std::addressof(iterator->second);
    }
    // Tries to insert the given element, if a value was already present at
    // the given key then returns a pointer to that element instead.
    // Returns nullptr if insertion was successful.
    TValue* tryGetValueOrAdd(const TKey& key, const TValue& value)
    {
        return tryGetValueOrAdd({key, value});
    }

    // Inserts the given value if it doesn't exist already
    // Return a reference to the (possibly new) value in the map
    TValue& getOrAddValue(const TKey& key, const TValue& defaultValue)
    {
        auto [iterator, inserted] = map.insert({key, defaultValue});
        return iterator->second;
    }

    // Returns a reference to the value at the specified key, default
    // initializing it if it doesn't already exist
    TValue& operator[](const TKey& key) { return map[key]; }
    // Returns a reference to the value at the specified key, default
    // initializing it if it doesn't already exist
    TValue& operator[](TKey&& key) { return map[std::move(key)]; }

    //
    // Insertion
    //

    // Returns true if the value was inserted, returns false if the map
    // already has a value associated with this key
    bool addIfNotExists(typename InnerMap::value_type&& kvPair)
    {
        return !tryGetValueOrAdd(std::move(kvPair));
    }
    // Returns true if the value was inserted, returns false if the map
    // already has a value associated with this key
    bool addIfNotExists(const typename InnerMap::value_type& kvPair)
    {
        return !tryGetValueOrAdd(kvPair);
    }
    // Returns true if the value was inserted, returns false if the map
    // already has a value associated with this key
    bool addIfNotExists(const TKey& k, const TValue& v) { return addIfNotExists({k, v}); }
    // Returns true if the value was inserted, returns false if the map
    // already has a value associated with this key
    bool addIfNotExists(TKey&& k, TValue&& v)
    {
        return addIfNotExists({std::move(k), std::move(v)});
    }

    // Asserts if the key already exists in the dictionary
    void add(typename InnerMap::value_type&& kvPair)
    {
        if (!addIfNotExists(std::move(kvPair)))
            SLANG_ASSERT_FAILURE("The key already exists in Dictionary.");
    }
    // Asserts if the key already exists in the dictionary
    void add(const typename InnerMap::value_type& kvPair)
    {
        if (!addIfNotExists(kvPair))
            SLANG_ASSERT_FAILURE("The key already exists in Dictionary.");
    }
    // Asserts if the key already exists in the dictionary
    void add(const TKey& key, const TValue& value) { add({key, value}); }
    // Asserts if the key already exists in the dictionary
    void add(TKey&& key, TValue&& value) { add({std::move(key), std::move(value)}); }

    // Inserts into the dictionary or assigns if the key already exists
    void set(const TKey& key, const TValue& value) { map.insert_or_assign(key, value); }
};

/* We may want to rename this, as strictly speaking _Caps names are reserved */
class _DummyClass
{
};

template<typename T, typename DictionaryType>
class HashSetBase
{
protected:
    DictionaryType dict;

private:
    template<typename... Args>
    void init(const T& v, Args... args)
    {
        add(v);
        init(args...);
    }

public:
    HashSetBase() {}
    template<typename Arg, typename... Args>
    HashSetBase(Arg arg, Args... args)
    {
        init(arg, args...);
    }
    HashSetBase(const HashSetBase& set) { operator=(set); }
    HashSetBase(HashSetBase&& set) { operator=(_Move(set)); }
    HashSetBase& operator=(const HashSetBase& set)
    {
        dict = set.dict;
        return *this;
    }
    HashSetBase& operator=(HashSetBase&& set)
    {
        dict = _Move(set.dict);
        return *this;
    }

public:
    class Iterator
    {
    private:
        typename DictionaryType::ConstIterator iter;

    public:
        Iterator() = default;
        const T& operator*() const { return *KeyValueDetail::getKey(std::addressof(*iter)); }
        const T* operator->() const { return KeyValueDetail::getKey(std::addressof(*iter)); }

        Iterator& operator++()
        {
            ++iter;
            return *this;
        }
        Iterator operator++(int)
        {
            Iterator rs = *this;
            operator++();
            return rs;
        }
        bool operator!=(const Iterator& that) const { return iter != that.iter; }
        bool operator==(const Iterator& that) const { return iter == that.iter; }
        Iterator(const typename DictionaryType::ConstIterator& _iter) { this->iter = _iter; }
    };
    Iterator begin() const { return Iterator(dict.begin()); }
    Iterator end() const { return Iterator(dict.end()); }

public:
    auto getCount() const { return dict.getCount(); }
    void clear() { dict.clear(); }
    bool add(const T& obj) { return dict.addIfNotExists(obj, _DummyClass()); }
    bool add(T&& obj) { return dict.addIfNotExists(_Move(obj), _DummyClass()); }
    void remove(const T& obj) { dict.remove(obj); }
    bool contains(const T& obj) const { return dict.containsKey(obj); }
};
template<typename T>
class HashSet : public HashSetBase<T, Dictionary<T, _DummyClass>>
{
};

template<typename TKey, typename TValue>
class OrderedDictionary
{
    friend class Iterator;
    friend class ItemProxy;

private:
    inline int getProbeOffset(int /*probeIdx*/) const
    {
        // quadratic probing
        return 1;
    }

private:
    int m_bucketCountMinusOne;
    int m_count;
    UIntSet m_marks;

    LinkedList<KeyValuePair<TKey, TValue>> m_kvPairs;
    LinkedNode<KeyValuePair<TKey, TValue>>** m_hashMap;
    void deallocateAll()
    {
        if (m_hashMap)
            delete[] m_hashMap;
        m_hashMap = nullptr;
        m_kvPairs.clear();
    }
    inline bool isDeleted(int pos) const { return m_marks.contains((pos << 1) + 1); }
    inline bool isEmpty(int pos) const { return !m_marks.contains((pos << 1)); }
    inline void setDeleted(int pos, bool val)
    {
        if (val)
            m_marks.add((pos << 1) + 1);
        else
            m_marks.remove((pos << 1) + 1);
    }
    inline void setEmpty(int pos, bool val)
    {
        if (val)
            m_marks.remove((pos << 1));
        else
            m_marks.add((pos << 1));
    }
    struct FindPositionResult
    {
        int objectPosition;
        int insertionPosition;
        FindPositionResult()
        {
            objectPosition = -1;
            insertionPosition = -1;
        }
        FindPositionResult(int objPos, int insertPos)
        {
            objectPosition = objPos;
            insertionPosition = insertPos;
        }
    };
    template<typename T>
    inline int getHashPos(T& key) const
    {
        const unsigned int hash = (unsigned int)getHashCode(key);
        return ((unsigned int)(hash * 2654435761)) % m_bucketCountMinusOne;
    }
    template<typename T>
    FindPositionResult findPosition(const T& key) const
    {
        int hashPos = getHashPos((T&)key);
        int insertPos = -1;
        int numProbes = 0;
        while (numProbes <= m_bucketCountMinusOne)
        {
            if (isEmpty(hashPos))
            {
                if (insertPos == -1)
                    return FindPositionResult(-1, hashPos);
                else
                    return FindPositionResult(-1, insertPos);
            }
            else if (isDeleted(hashPos))
            {
                if (insertPos == -1)
                    insertPos = hashPos;
            }
            else if (m_hashMap[hashPos]->value.key == key)
            {
                return FindPositionResult(hashPos, -1);
            }
            numProbes++;
            hashPos = (hashPos + getProbeOffset(numProbes)) & m_bucketCountMinusOne;
        }
        if (insertPos != -1)
            return FindPositionResult(-1, insertPos);
        SLANG_ASSERT_FAILURE(
            "Hash map is full. This indicates an error in Key::Equal or Key::GetHashCode.");
    }
    TValue& _insert(KeyValuePair<TKey, TValue>&& kvPair, int pos)
    {
        auto node = m_kvPairs.addLast();
        node->value = _Move(kvPair);
        m_hashMap[pos] = node;
        setEmpty(pos, false);
        setDeleted(pos, false);
        return node->value.value;
    }
    void maybeRehash()
    {
        if (m_bucketCountMinusOne == -1 || m_count / (float)m_bucketCountMinusOne >= kMaxLoadFactor)
        {
            int newSize = (m_bucketCountMinusOne + 1) * 2;
            if (newSize == 0)
            {
                newSize = 128;
            }
            OrderedDictionary<TKey, TValue> newDict;
            newDict.m_bucketCountMinusOne = newSize - 1;
            newDict.m_hashMap = new LinkedNode<KeyValuePair<TKey, TValue>>*[newSize];
            newDict.m_marks.resizeAndClear(newSize * 2);
            if (m_hashMap)
            {
                for (auto& kvPair : *this)
                {
                    newDict.add(_Move(kvPair));
                }
            }
            *this = _Move(newDict);
        }
    }

    bool addIfNotExists(KeyValuePair<TKey, TValue>&& kvPair)
    {
        maybeRehash();
        auto pos = findPosition(kvPair.key);
        if (pos.objectPosition != -1)
            return false;
        else if (pos.insertionPosition != -1)
        {
            m_count++;
            _insert(_Move(kvPair), pos.insertionPosition);
            return true;
        }
        else
            SLANG_ASSERT_FAILURE(
                "Inconsistent find result returned. This is a bug in Dictionary implementation.");
    }
    void add(KeyValuePair<TKey, TValue>&& kvPair)
    {
        if (!addIfNotExists(_Move(kvPair)))
            SLANG_ASSERT_FAILURE("The key already exists in Dictionary.");
    }
    TValue& set(KeyValuePair<TKey, TValue>&& kvPair)
    {
        maybeRehash();
        auto pos = findPosition(kvPair.key);
        if (pos.objectPosition != -1)
        {
            m_hashMap[pos.objectPosition]->removeAndDelete();
            return _insert(_Move(kvPair), pos.objectPosition);
        }
        else if (pos.insertionPosition != -1)
        {
            m_count++;
            return _insert(_Move(kvPair), pos.insertionPosition);
        }
        else
            SLANG_ASSERT_FAILURE(
                "Inconsistent find result returned. This is a bug in Dictionary implementation.");
    }

public:
    using Iterator = typename LinkedList<KeyValuePair<TKey, TValue>>::Iterator;
    using ConstIterator = typename LinkedList<KeyValuePair<TKey, TValue>>::ConstIterator;

    Iterator begin() { return m_kvPairs.begin(); }
    Iterator end() { return m_kvPairs.end(); }
    ConstIterator begin() const { return m_kvPairs.begin(); }
    ConstIterator end() const { return m_kvPairs.end(); }

public:
    void add(const TKey& key, const TValue& value) { add(KeyValuePair<TKey, TValue>(key, value)); }
    void add(TKey&& key, TValue&& value)
    {
        add(KeyValuePair<TKey, TValue>(_Move(key), _Move(value)));
    }
    bool addIfNotExists(const TKey& key, const TValue& value)
    {
        return addIfNotExists(KeyValuePair<TKey, TValue>(key, value));
    }
    bool addIfNotExists(TKey&& key, TValue&& value)
    {
        return addIfNotExists(KeyValuePair<TKey, TValue>(_Move(key), _Move(value)));
    }
    void remove(const TKey& key)
    {
        if (m_count > 0)
        {
            auto pos = findPosition(key);
            if (pos.objectPosition != -1)
            {
                m_kvPairs.removeAndDelete(m_hashMap[pos.objectPosition]);
                m_hashMap[pos.objectPosition] = 0;
                setDeleted(pos.objectPosition, true);
                m_count--;
            }
        }
    }
    void clear()
    {
        m_count = 0;
        m_kvPairs.clear();
        m_marks.clear();
    }
    template<typename T>
    bool containsKey(const T& key) const
    {
        if (m_bucketCountMinusOne == -1)
            return false;
        auto pos = findPosition(key);
        return pos.objectPosition != -1;
    }
    template<typename T>
    TValue* tryGetValue(const T& key) const
    {
        if (m_bucketCountMinusOne == -1)
            return nullptr;
        auto pos = findPosition(key);
        if (pos.objectPosition != -1)
        {
            return &(m_hashMap[pos.objectPosition]->value.value);
        }
        return nullptr;
    }
    template<typename T>
    bool tryGetValue(const T& key, TValue& value) const
    {
        if (m_bucketCountMinusOne == -1)
            return false;
        auto pos = findPosition(key);
        if (pos.objectPosition != -1)
        {
            value = m_hashMap[pos.objectPosition]->value.value;
            return true;
        }
        return false;
    }
    class ItemProxy
    {
    private:
        const OrderedDictionary<TKey, TValue>* dict;
        TKey key;

    public:
        ItemProxy(const TKey& _key, const OrderedDictionary<TKey, TValue>* _dict)
        {
            this->dict = _dict;
            this->key = _key;
        }
        ItemProxy(TKey&& _key, const OrderedDictionary<TKey, TValue>* _dict)
        {
            this->dict = _dict;
            this->key = _Move(_key);
        }
        TValue& getValue() const
        {
            auto pos = dict->findPosition(key);
            if (pos.objectPosition != -1)
            {
                return dict->m_hashMap[pos.objectPosition]->value.value;
            }
            else
            {
                SLANG_ASSERT_FAILURE("The key does not exists in dictionary.");
            }
        }
        inline TValue& operator()() const { return getValue(); }
        operator TValue&() const { return getValue(); }
        TValue& operator=(const TValue& val)
        {
            return ((OrderedDictionary<TKey, TValue>*)dict)
                ->set(KeyValuePair<TKey, TValue>(_Move(key), val));
        }
        TValue& operator=(TValue&& val)
        {
            return ((OrderedDictionary<TKey, TValue>*)dict)
                ->set(KeyValuePair<TKey, TValue>(_Move(key), _Move(val)));
        }
    };
    ItemProxy operator[](const TKey& key) const { return ItemProxy(key, this); }
    ItemProxy operator[](TKey&& key) const { return ItemProxy(_Move(key), this); }

    int getCount() const { return m_count; }
    KeyValuePair<TKey, TValue>& getFirst() const { return m_kvPairs.getFirst(); }
    KeyValuePair<TKey, TValue>& getLast() const { return m_kvPairs.getLast(); }

private:
    template<typename... Args>
    void init(const KeyValuePair<TKey, TValue>& kvPair, Args... args)
    {
        add(kvPair);
        init(args...);
    }

public:
    OrderedDictionary()
    {
        m_bucketCountMinusOne = -1;
        m_count = 0;
        m_hashMap = 0;
    }
    template<typename Arg, typename... Args>
    OrderedDictionary(Arg arg, Args... args)
    {
        init(arg, args...);
    }
    OrderedDictionary(const OrderedDictionary<TKey, TValue>& other)
        : m_bucketCountMinusOne(-1), m_count(0), m_hashMap(0)
    {
        *this = other;
    }
    OrderedDictionary(OrderedDictionary<TKey, TValue>&& other)
        : m_bucketCountMinusOne(-1), m_count(0), m_hashMap(0)
    {
        *this = (_Move(other));
    }
    OrderedDictionary<TKey, TValue>& operator=(const OrderedDictionary<TKey, TValue>& other)
    {
        if (this == &other)
            return *this;
        clear();
        for (auto& item : other)
            add(item.key, item.value);
        return *this;
    }
    OrderedDictionary<TKey, TValue>& operator=(OrderedDictionary<TKey, TValue>&& other)
    {
        if (this == &other)
            return *this;
        deallocateAll();
        m_bucketCountMinusOne = other.m_bucketCountMinusOne;
        m_count = other.m_count;
        m_hashMap = other.m_hashMap;
        m_marks = _Move(other.m_marks);
        other.m_hashMap = 0;
        other.m_count = 0;
        other.m_bucketCountMinusOne = -1;
        m_kvPairs = _Move(other.m_kvPairs);
        return *this;
    }
    ~OrderedDictionary() { deallocateAll(); }
};

template<typename T>
class OrderedHashSet : public HashSetBase<T, OrderedDictionary<T, _DummyClass>>
{
public:
    T& getLast() { return this->dict.getLast().key; }
    void removeLast() { this->remove(getLast()); }
};
} // namespace Slang

#endif
