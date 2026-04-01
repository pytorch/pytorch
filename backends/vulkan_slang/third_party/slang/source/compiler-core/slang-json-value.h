// slang-json-value.h
#ifndef SLANG_JSON_VALUE_H
#define SLANG_JSON_VALUE_H

#include "../core/slang-basic.h"
#include "../core/slang-rtti-info.h"
#include "slang-diagnostic-sink.h"
#include "slang-json-parser.h"
#include "slang-source-loc.h"

namespace Slang
{

typedef uint32_t JSONKey;

struct JSONValue
{
    enum class Kind
    {
        Invalid,

        Null,

        Bool,
        String,
        Integer,
        Float,

        Array,
        Object,

        CountOf,
    };

    enum class Type
    {
        Invalid,

        True,
        False,
        Null,

        StringLexeme,
        IntegerLexeme,
        FloatLexeme,

        IntegerValue,
        FloatValue,
        StringValue,

        StringRepresentation,

        Array,
        Object,

        CountOf,
    };

    static bool isLexeme(Type type)
    {
        return Index(type) >= Index(Type::StringLexeme) && Index(type) <= Index(Type::FloatLexeme);
    }

    static JSONValue makeInt(int64_t inValue, SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = Type::IntegerValue;
        value.loc = loc;
        value.intValue = inValue;
        return value;
    }
    static JSONValue makeFloat(double inValue, SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = Type::FloatValue;
        value.loc = loc;
        value.floatValue = inValue;
        return value;
    }
    static JSONValue makeNull(SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = Type::Null;
        value.loc = loc;
        return value;
    }
    static JSONValue makeBool(bool inValue, SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = (inValue ? Type::True : Type::False);
        value.loc = loc;
        return value;
    }

    static JSONValue makeLexeme(Type type, SourceLoc loc, Index length)
    {
        SLANG_ASSERT(isLexeme(type));
        JSONValue value;
        value.type = type;
        value.loc = loc;
        value.length = length;
        return value;
    }

    static JSONValue makeEmptyArray(SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = Type::Array;
        value.loc = loc;
        value.rangeIndex = 0;
        return value;
    }
    static JSONValue makeEmptyObject(SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = Type::Object;
        value.loc = loc;
        value.rangeIndex = 0;
        return value;
    }

    static JSONValue makeInvalid(SourceLoc loc = SourceLoc())
    {
        JSONValue value;
        value.type = Type::Invalid;
        value.loc = loc;
        return value;
    }
    // The following functions only work if the value is stored directly NOT as a lexeme. Use the
    // methods on the container to access values if it is potentially stored as a lexeme

    /// As a boolean value
    bool asBool() const;
    /// As an integer value
    int64_t asInteger() const;
    /// As a float value
    double asFloat() const;

    /// True if this is a object like
    bool isObjectLike() const { return Index(type) >= Index(Type::Array); }

    /// True if this appears to be a valid value
    bool isValid() const { return type != JSONValue::Type::Invalid; }

    /// True if needs destroy
    bool needsDestroy() const { return isObjectLike() && rangeIndex != 0; }

    /// Get the kind
    SLANG_FORCE_INLINE Kind getKind() const { return getKindForType(type); }

    void reset()
    {
        type = Type::Invalid;
        loc = SourceLoc();
    }

    /// Given a type return the associated kind
    static Kind getKindForType(Type type) { return g_typeToKind[Index(type)]; }

    Type type = Type::Invalid; ///< The type of value
    SourceLoc loc;             ///< The (optional) location in source of this value.

    union
    {
        Index rangeIndex;                ///< Used for Array/Object
        Index length;                    ///< Length in bytes if it is a 'Lexeme'
        double floatValue;               ///< Float value
        int64_t intValue;                ///< Integer value
        JSONKey stringKey;               ///< The pool key if it's a string
        StringRepresentation* stringRep; ///< Only ever used on a 'PersistentJSONValue'
    };

    static const Kind g_typeToKind[Index(Type::CountOf)];

    static const OtherRttiInfo g_rttiInfo;
};

template<>
struct GetRttiInfo<JSONValue>
{
    static const RttiInfo* get() { return &JSONValue::g_rttiInfo; }
};

struct JSONKeyValue
{
    /// True if it's valid
    bool isValid() const { return value.type != JSONValue::Type::Invalid; }

    void reset()
    {
        key = JSONKey(0);
        keyLoc = SourceLoc();
        value.reset();
    }

    JSONKey key;
    SourceLoc keyLoc;
    JSONValue value;

    static JSONKeyValue make(JSONKey inKey, JSONValue inValue, SourceLoc inKeyLoc = SourceLoc())
    {
        return JSONKeyValue{inKey, inKeyLoc, inValue};
    }

    static JSONKeyValue g_invalid;
};

class JSONContainer;

/* Is similar to JSONValue, but is designed to

* Only be able to hold 'Simple' types (ie not array/object)
* Does not reference/require JSONContainer.

Not requiring JSONContainer means it's useful to hold state when JSONContainer goes out of scope.
Care may need to be taken if sourceManager goes out of scope, sourceLocs may become invalid. This
is true of a regular JSONValue.

Care must also be taken because it is derived from JSONValue. It *can* be sliced and work correctly,
but *requires* that the PersistentJSONValue with same value to stay in scope in general. In practice
this is only an issue with StringRepresention type.
*/
class PersistentJSONValue : public JSONValue
{
public:
    typedef JSONValue Super;
    typedef PersistentJSONValue ThisType;

    /// If it's a string type this will always work
    String getString() const;
    UnownedStringSlice getSlice() const;

    /// Set to the value
    void set(const JSONValue& in, JSONContainer* container);
    /// Set directly to a string
    void set(const UnownedStringSlice& slice, SourceLoc loc);

    /// True if identical
    bool operator==(const ThisType& rhs) const;
    bool operator!=(const ThisType& rhs) const { return !(*this == rhs); }

    /// Assignable
    void operator=(const ThisType& rhs);

    PersistentJSONValue(const JSONValue& in, JSONContainer* container) { _init(in, container); }
    PersistentJSONValue(const JSONValue& in, JSONContainer* container, SourceLoc inLoc)
    {
        _init(in, container);
        loc = inLoc;
    }

    /// Copy Ctor
    PersistentJSONValue(const ThisType& rhs);
    /// Default Ctor (will be set to invalid)
    PersistentJSONValue() {}


    ~PersistentJSONValue()
    {
        if (type == Type::StringRepresentation && stringRep)
        {
            stringRep->releaseReference();
        }
    }

protected:
    /// Assumes this has no valid data
    void _init(const JSONValue& in, JSONContainer* container);
    void _init(const UnownedStringSlice& slice, SourceLoc loc);
};

class JSONContainer : public RefObject
{
public:
    /// Make a new array
    JSONValue createArray(const JSONValue* values, Index valuesCount, SourceLoc loc = SourceLoc());
    /// Make a new object
    JSONValue createObject(
        const JSONKeyValue* keyValues,
        Index keyValueCount,
        SourceLoc loc = SourceLoc());
    /// Make a string
    JSONValue createString(const UnownedStringSlice& slice, SourceLoc loc = SourceLoc());

    ConstArrayView<JSONValue> getArray(const JSONValue& in) const;
    ConstArrayView<JSONKeyValue> getObject(const JSONValue& in) const;

    ArrayView<JSONValue> getArray(const JSONValue& in);
    ArrayView<JSONKeyValue> getObject(const JSONValue& in);

    /// Add value to array.
    void addToArray(JSONValue& array, const JSONValue& value);

    /// Get the value at the index in the array
    JSONValue& getAt(const JSONValue& array, Index index);

    /// Returns the index of key in obj, or -1 if not found
    Index findObjectIndex(const JSONValue& obj, JSONKey key) const;
    /// Get the value in the object at key. Returns invalid if not found.
    JSONValue findObjectValue(const JSONValue& obj, JSONKey key) const;

    /// Returns the index
    Index findKeyGlobalIndex(const JSONValue& obj, JSONKey key);
    Index findKeyGlobalIndex(const JSONValue& obj, const UnownedStringSlice& slice);

    /// Set a key value for the obj
    void setKeyValue(
        JSONValue& obj,
        JSONKey key,
        const JSONValue& value,
        SourceLoc loc = SourceLoc());

    /// Returns true if found
    bool removeKey(JSONValue& obj, JSONKey key);
    bool removeKey(JSONValue& obj, const UnownedStringSlice& slice);

    /// As a boolean value
    bool asBool(const JSONValue& value);
    /// As an integer value
    int64_t asInteger(const JSONValue& value);
    /// As a float value
    double asFloat(const JSONValue& value);

    /// Returns string as a key
    JSONKey getStringKey(const JSONValue& in);

    /// Get as a string. The slice may used backing lexeme (ie will only last
    /// as long as the backing JSON text, or be decoded and be transitory).
    UnownedStringSlice getTransientString(const JSONValue& in);

    /// Get as a string. The contents will stay in scope as long as the container
    UnownedStringSlice getString(const JSONValue& in);

    /// Gets the lexeme
    UnownedStringSlice getLexeme(const JSONValue& in);

    /// Get a key for a name
    JSONKey getKey(const UnownedStringSlice& slice);
    /// Returns JSONKey(0) if not found
    JSONKey findKey(const UnownedStringSlice& slice) const;
    /// Get the string from the key
    UnownedStringSlice getStringFromKey(JSONKey key) const
    {
        return m_slicePool.getSlice(StringSlicePool::Handle(key));
    }

    /// True if they are the same value
    /// If object like type comparison is performed recursively.
    /// NOTE! That Float and Integer values do not compare & source locations are ignored.
    bool areEqual(const JSONValue& a, const JSONValue& b);
    bool areEqual(const JSONValue* a, const JSONValue* b, Index count);
    bool areEqual(const JSONKeyValue* a, const JSONKeyValue* b, Index count);

    bool areEqual(const JSONValue& a, const UnownedStringSlice& slice);

    /// Destroy value
    void destroy(JSONValue& value);
    /// Destroy recursively from value
    void destroyRecursively(JSONValue& value);

    /// Traverse a JSON hierarchy from value, outputting to the listener
    void traverseRecursively(const JSONValue& value, JSONListener* listener);

    /// Returns the source manager used.
    SourceManager* getSourceManager() const { return m_sourceManager; }
    /// Set the source manager
    void setSourceManager(SourceManager* sourceManger) { m_sourceManager = sourceManger; }

    /// Clears all the source locs. Useful if the sourceManager is no longer available, or has
    /// itself been reset. All JSONValues which were Lexeme based will become held in the container
    /// The source manager will set to nullptr
    void clearSourceManagerDependency(JSONValue* ioValues, Index count);

    /// Reset the state
    void reset();

    /// Return inValue as a regular value (ie not held as a lexeme)
    JSONValue asValue(const JSONValue& inValue);

    // Ctor
    JSONContainer(SourceManager* sourceManger);

    /// Returns true if all the keys are unique
    static bool areKeysUnique(const JSONKeyValue* keyValues, Index keyValueCount);

    /// Access the internal set of strings, removing anything from this
    /// will invalidate the container, so only do it immediately prior to
    /// destruction.
    StringSlicePool& getStringSlicePool() { return m_slicePool; };

protected:
    struct Range
    {
        // We want to record the underlying range, because we don't track JSONValue, and so we need
        // to know what the range applies to if we want to reorder, flatten etc.
        enum class Type
        {
            None,
            Destroyed,
            Object,
            Array,
        };

        /// Is active if it consuming some part of a value list (even if zero count)
        SLANG_FORCE_INLINE bool isActive() const { return Index(type) >= Index(Type::Object); }

        Type type;
        Index startIndex;
        Index count;
        Index capacity;
    };

    template<typename T>
    static void _add(Range& range, List<T>& list, const T& value);

    Index _addRange(Range::Type type, Index startIndex, Index count);
    void _removeKey(JSONValue& obj, Index globalIndex);
    /// Note does not destroy values in range.
    void _destroyRange(Index rangeIndex);

    static bool _sameKeyOrder(const JSONKeyValue* a, const JSONKeyValue* b, Index count);
    /// True if the values are equal
    bool _areEqualValues(const JSONKeyValue* a, const JSONKeyValue* b, Index count);
    /// True if the key and value are equal
    bool _areEqualOrderedKeys(const JSONKeyValue* a, const JSONKeyValue* b, Index count);

    void _clearSourceManagerDependency(JSONValue* ioValues, Index count);
    JSONValue _removeManagerDependency(const JSONValue& inValue);

    StringBuilder m_buf; ///< A temporary buffer used to hold unescaped strings

    SourceView* m_currentView = nullptr;
    SourceManager* m_sourceManager;

    StringSlicePool m_slicePool;
    List<Range> m_ranges;
    List<Index> m_freeRangeIndices;
    List<JSONValue> m_arrayValues;
    List<JSONKeyValue> m_objectValues;
};

class JSONBuilder : public JSONListener
{
public:
    typedef uint32_t Flags;
    struct Flag
    {
        enum Enum : Flags
        {
            ConvertLexemes = 0x01,
        };
    };


    virtual void startObject(SourceLoc loc) SLANG_OVERRIDE;
    virtual void endObject(SourceLoc loc) SLANG_OVERRIDE;
    virtual void startArray(SourceLoc loc) SLANG_OVERRIDE;
    virtual void endArray(SourceLoc loc) SLANG_OVERRIDE;
    virtual void addQuotedKey(const UnownedStringSlice& key, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addUnquotedKey(const UnownedStringSlice& key, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addLexemeValue(JSONTokenType type, const UnownedStringSlice& value, SourceLoc loc)
        SLANG_OVERRIDE;
    virtual void addIntegerValue(int64_t value, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addFloatValue(double value, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addBoolValue(bool value, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addStringValue(const UnownedStringSlice& string, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addNullValue(SourceLoc loc) SLANG_OVERRIDE;

    /// Reset the state
    void reset();

    /// Get the root value. Will be set after valid construction
    const JSONValue& getRootValue() const { return m_rootValue; }

    JSONBuilder(JSONContainer* container, Flags flags = 0);

protected:
    struct State
    {
        enum class Kind : uint8_t
        {
            Root,
            Object,
            Array,
        };
        void setKey(JSONKey key, SourceLoc loc)
        {
            m_key = key;
            m_keyLoc = loc;
        }
        void resetKey()
        {
            m_key = JSONKey(0);
            m_keyLoc = SourceLoc();
        }
        bool hasKey() const { return m_key != JSONKey(0); }

        Kind m_kind;
        Index m_startIndex;
        SourceLoc m_loc;
        JSONKey m_key;
        SourceLoc m_keyLoc;
    };

    void _popState();
    void _add(const JSONValue& value);

    Index _findKeyIndex(JSONKey key) const;

    Flags m_flags;

    List<JSONKeyValue> m_keyValues;
    List<JSONValue> m_values;
    List<State> m_stateStack;

    State m_state;

    JSONContainer* m_container;
    JSONValue m_rootValue;

    StringBuilder m_work;
};

} // namespace Slang

#endif
