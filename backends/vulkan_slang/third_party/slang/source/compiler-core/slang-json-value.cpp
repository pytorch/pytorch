// slang-json-value.cpp
#include "slang-json-value.h"

#include "../core/slang-string-escape-util.h"
#include "../core/slang-string-util.h"

namespace Slang
{

/* static */ const JSONValue::Kind JSONValue::g_typeToKind[] = {
    JSONValue::Kind::Invalid, // Invalid

    JSONValue::Kind::Bool, // True,
    JSONValue::Kind::Bool, // False
    JSONValue::Kind::Null, // Null,

    JSONValue::Kind::String,  // StringLexeme,
    JSONValue::Kind::Integer, // IntegerLexeme,
    JSONValue::Kind::Float,   // FloatLexeme,

    JSONValue::Kind::Integer, // IntegerValue,
    JSONValue::Kind::Float,   // FloatValue,
    JSONValue::Kind::String,  // StringValue,

    JSONValue::Kind::String, // StringRepresentation

    JSONValue::Kind::Array,  // Array,
    JSONValue::Kind::Object, // Object,
};

static bool _isDefault(const RttiInfo* type, const void* in)
{
    SLANG_UNUSED(type)
    const JSONValue& value = *(const JSONValue*)in;
    return value.getKind() == JSONValue::Kind::Invalid;
}

static OtherRttiInfo _getJSONValueRttiInfo()
{
    OtherRttiInfo info;
    info.init<JSONValue>(RttiInfo::Kind::Other);
    info.m_name = "JSONValue";
    info.m_isDefaultFunc = _isDefault;
    info.m_typeFuncs = GetRttiTypeFuncs<JSONValue>::getFuncs();
    return info;
}
/* static */ const OtherRttiInfo JSONValue::g_rttiInfo = _getJSONValueRttiInfo();

static JSONKeyValue _makeInvalidKeyValue()
{
    JSONKeyValue keyValue;
    keyValue.key = JSONKey(0);
    keyValue.value.type = JSONValue::Type::Invalid;
    return keyValue;
}

/* static */ JSONKeyValue g_invalid = _makeInvalidKeyValue();

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                             JSONValue

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

bool JSONValue::asBool() const
{
    switch (type)
    {
    case JSONValue::Type::True:
        return true;
    case JSONValue::Type::False:
    case JSONValue::Type::Null:
        {
            return false;
        }
    case JSONValue::Type::IntegerValue:
        return intValue != 0;
    case JSONValue::Type::FloatValue:
        return floatValue != 0;
    default:
        break;
    }

    if (isLexeme(type))
    {
        SLANG_ASSERT(!"Lexeme values can only be accessed through container");
    }
    else
    {
        SLANG_ASSERT(!"Not bool convertable");
    }

    return false;
}

int64_t JSONValue::asInteger() const
{
    switch (type)
    {
    case JSONValue::Type::True:
        return 1;
    case JSONValue::Type::False:
    case JSONValue::Type::Null:
        {
            return 0;
        }
    case JSONValue::Type::IntegerValue:
        return intValue;
    case JSONValue::Type::FloatValue:
        return int64_t(floatValue);
        break;
    }

    if (isLexeme(type))
    {
        SLANG_ASSERT(!"Lexeme values can only be accessed through container");
    }
    else
    {
        SLANG_ASSERT(!"Not int convertable");
    }

    return 0;
}

double JSONValue::asFloat() const
{
    switch (type)
    {
    case JSONValue::Type::True:
        return 1.0;
    case JSONValue::Type::False:
    case JSONValue::Type::Null:
        {
            return 0.0;
        }
    case JSONValue::Type::IntegerValue:
        return double(intValue);
    case JSONValue::Type::FloatValue:
        return floatValue;
    default:
        break;
    }

    if (isLexeme(type))
    {
        SLANG_ASSERT(!"Lexeme values can only be accessed through container");
    }
    else
    {
        SLANG_ASSERT(!"Not float convertable");
    }

    return 0;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                             PersistentJSONValue

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

PersistentJSONValue::PersistentJSONValue(const ThisType& rhs)
{
    *(JSONValue*)this = rhs;

    if (type == Type::StringRepresentation && stringRep)
    {
        stringRep->addReference();
    }
}

void PersistentJSONValue::operator=(const ThisType& rhs)
{
    if (this != &rhs)
    {
        if (rhs.type == Type::StringRepresentation && rhs.stringRep)
        {
            rhs.stringRep->addReference();
        }
        if (type == Type::StringRepresentation && stringRep)
        {
            stringRep->releaseReference();
        }
        *(JSONValue*)this = rhs;
    }
}

String PersistentJSONValue::getString() const
{
    if (type == Type::StringRepresentation)
    {
        return String(stringRep);
    }
    SLANG_ASSERT(!"Not a string type");
    return String();
}

UnownedStringSlice PersistentJSONValue::getSlice() const
{
    if (type == Type::StringRepresentation)
    {
        return StringRepresentation::asSlice(stringRep);
    }
    SLANG_ASSERT(!"Not a string type");
    return UnownedStringSlice();
}

void PersistentJSONValue::set(const UnownedStringSlice& slice, SourceLoc inLoc)
{
    StringRepresentation* oldRep =
        (type == JSONValue::Type::StringRepresentation) ? stringRep : nullptr;

    type = Type::StringRepresentation;
    loc = inLoc;

    StringRepresentation* newRep = nullptr;

    const auto sliceLength = slice.getLength();

    // If we have an oldRep that is unique and large enough reuse it
    if (sliceLength)
    {
        if (oldRep && oldRep->isUniquelyReferenced() && sliceLength <= oldRep->capacity)
        {
            oldRep->setContents(slice);
            newRep = oldRep;
            // We are reusing so make null so not freed
            oldRep = nullptr;
        }
        else
        {
            newRep = StringRepresentation::createWithReference(slice);
        }

        SLANG_ASSERT(newRep->debugGetReferenceCount() >= 1);
    }

    stringRep = newRep;

    if (oldRep)
    {
        oldRep->releaseReference();
    }
}

void PersistentJSONValue::_init(const UnownedStringSlice& slice, SourceLoc inLoc)
{
    loc = inLoc;
    type = Type::StringRepresentation;
    stringRep = StringRepresentation::createWithReference(slice);
}

bool PersistentJSONValue::operator==(const ThisType& rhs) const
{
    if (this == &rhs)
    {
        return true;
    }

    if (type != rhs.type || loc != rhs.loc)
    {
        return false;
    }

    switch (type)
    {
    case Type::Invalid:
    case Type::True:
    case Type::False:
    case Type::Null:
        {
            // The type is all that needs to be checked
            return true;
        }
    case Type::IntegerValue:
        return intValue == rhs.intValue;
    case Type::FloatValue:
        return floatValue == rhs.floatValue;
    case Type::StringRepresentation:
        {
            if (stringRep == rhs.stringRep)
            {
                return true;
            }
            auto thisSlice = StringRepresentation::asSlice(stringRep);
            auto rhsSlice = StringRepresentation::asSlice(rhs.stringRep);
            return thisSlice == rhsSlice;
        }
    default:
        break;
    }

    SLANG_ASSERT(!"Not valid Persistent type");
    return false;
}

void PersistentJSONValue::_init(const JSONValue& in, JSONContainer* container)
{
    // We are assuming this is invalid, so it can't be the same as in
    SLANG_ASSERT(&in != this);

    switch (in.type)
    {
    case Type::StringValue:
    case Type::StringLexeme:
        {
            if (!container)
            {
                SLANG_ASSERT(!"Requires container");
                return;
            }
            _init(container->getTransientString(in), in.loc);
            break;
        }
    case Type::StringRepresentation:
        {
            *(JSONValue*)this = in;
            if (stringRep)
            {
                stringRep->addReference();
            }
            break;
        }
    case Type::IntegerLexeme:
        {
            type = JSONValue::Type::IntegerValue;
            intValue = container->asInteger(in);
            loc = in.loc;
            break;
        }
    case Type::FloatLexeme:
        {
            type = JSONValue::Type::FloatValue;
            floatValue = container->asFloat(in);
            loc = in.loc;
            break;
        }
    case Type::Array:
    case Type::Object:
        {
            SLANG_ASSERT(!"Not a simple JSON type");
            break;
        }
    default:
        {
            *(JSONValue*)this = in;
            break;
        }
    }
}

void PersistentJSONValue::set(const JSONValue& in, JSONContainer* container)
{
    if (&in != this)
    {
        if (type == Type::StringRepresentation)
        {
            StringRepresentation* oldStringRep = stringRep;
            _init(in, container);
            if (oldStringRep)
            {
                oldStringRep->releaseReference();
            }
        }
        else
        {
            _init(in, container);
        }
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                             JSONContainer

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

JSONContainer::JSONContainer(SourceManager* sourceManager)
    : m_slicePool(StringSlicePool::Style::Default), m_sourceManager(sourceManager)
{
    // Index 0 is the empty array or object
    _addRange(Range::Type::None, 0, 0);
}

void JSONContainer::reset()
{
    m_slicePool.clear();

    m_freeRangeIndices.clear();
    m_arrayValues.clear();
    m_objectValues.clear();

    _addRange(Range::Type::None, 0, 0);

    m_currentView = nullptr;
}

/* static */ bool JSONContainer::areKeysUnique(const JSONKeyValue* keyValues, Index keyValueCount)
{
    for (Index i = 1; i < keyValueCount; ++i)
    {
        const JSONKey key = keyValues[i].key;

        for (Int j = 0; j < i - 1; j++)
        {
            if (keyValues[j].key == key)
            {
                return false;
            }
        }
    }

    return true;
}

Index JSONContainer::_addRange(Range::Type type, Index startIndex, Index count)
{
    if (m_freeRangeIndices.getCount() > 0)
    {
        const Index rangeIndex = m_freeRangeIndices.getLast();
        m_freeRangeIndices.removeLast();

        auto& range = m_ranges[rangeIndex];
        range.type = type;
        range.startIndex = startIndex;
        range.count = count;
        range.capacity = count;

        return rangeIndex;
    }
    else
    {
        Range range;
        range.type = type;
        range.startIndex = startIndex;
        range.count = count;
        range.capacity = count;

        m_ranges.add(range);
        return m_ranges.getCount() - 1;
    }
}

JSONValue JSONContainer::createArray(const JSONValue* values, Index valuesCount, SourceLoc loc)
{
    if (valuesCount <= 0)
    {
        return JSONValue::makeEmptyArray(loc);
    }

    JSONValue value;
    value.type = JSONValue::Type::Array;
    value.loc = loc;
    value.rangeIndex = _addRange(Range::Type::Array, m_arrayValues.getCount(), valuesCount);

    m_arrayValues.addRange(values, valuesCount);
    return value;
}

JSONValue JSONContainer::createObject(
    const JSONKeyValue* keyValues,
    Index keyValueCount,
    SourceLoc loc)
{
    if (keyValueCount <= 0)
    {
        return JSONValue::makeEmptyObject(loc);
    }

    JSONValue value;
    value.type = JSONValue::Type::Object;
    value.loc = loc;
    value.rangeIndex = _addRange(Range::Type::Object, m_objectValues.getCount(), keyValueCount);

    m_objectValues.addRange(keyValues, keyValueCount);
    return value;
}

JSONValue JSONContainer::createString(const UnownedStringSlice& slice, SourceLoc loc)
{
    JSONValue value;
    value.type = JSONValue::Type::StringValue;
    value.loc = loc;
    value.stringKey = getKey(slice);
    return value;
}

JSONKey JSONContainer::getKey(const UnownedStringSlice& slice)
{
    return JSONKey(m_slicePool.add(slice));
}

JSONKey JSONContainer::findKey(const UnownedStringSlice& slice) const
{
    const Index index = m_slicePool.findIndex(slice);
    return (index < 0) ? JSONKey(0) : JSONKey(index);
}

ConstArrayView<JSONValue> JSONContainer::getArray(const JSONValue& in) const
{
    SLANG_ASSERT(in.type == JSONValue::Type::Array);
    if (in.type != JSONValue::Type::Array || in.rangeIndex == 0)
    {
        return ConstArrayView<JSONValue>((const JSONValue*)nullptr, 0);
    }
    const Range& range = m_ranges[in.rangeIndex];
    return ConstArrayView<JSONValue>(m_arrayValues.getBuffer() + range.startIndex, range.count);
}

ConstArrayView<JSONKeyValue> JSONContainer::getObject(const JSONValue& in) const
{
    SLANG_ASSERT(in.type == JSONValue::Type::Object);
    if (in.type != JSONValue::Type::Object || in.rangeIndex == 0)
    {
        return ConstArrayView<JSONKeyValue>((const JSONKeyValue*)nullptr, 0);
    }

    const Range& range = m_ranges[in.rangeIndex];
    return ConstArrayView<JSONKeyValue>(m_objectValues.getBuffer() + range.startIndex, range.count);
}

ArrayView<JSONValue> JSONContainer::getArray(const JSONValue& in)
{
    SLANG_ASSERT(in.type == JSONValue::Type::Array);
    if (in.type != JSONValue::Type::Array || in.rangeIndex == 0)
    {
        return ArrayView<JSONValue>((JSONValue*)nullptr, 0);
    }
    const Range& range = m_ranges[in.rangeIndex];
    SLANG_ASSERT(
        range.startIndex <= m_arrayValues.getCount() &&
        range.startIndex + range.count <= m_arrayValues.getCount());

    return ArrayView<JSONValue>(m_arrayValues.getBuffer() + range.startIndex, range.count);
}

ArrayView<JSONKeyValue> JSONContainer::getObject(const JSONValue& in)
{
    SLANG_ASSERT(in.type == JSONValue::Type::Object);
    if (in.type != JSONValue::Type::Object || in.rangeIndex == 0)
    {
        return ArrayView<JSONKeyValue>((JSONKeyValue*)nullptr, 0);
    }

    const Range& range = m_ranges[in.rangeIndex];
    return ArrayView<JSONKeyValue>(m_objectValues.getBuffer() + range.startIndex, range.count);
}

UnownedStringSlice JSONContainer::getLexeme(const JSONValue& in)
{
    SLANG_ASSERT(JSONValue::isLexeme(in.type));
    if (!JSONValue::isLexeme(in.type))
    {
        return UnownedStringSlice();
    }

    if (!(m_currentView && m_currentView->getRange().contains(in.loc)))
    {
        m_currentView = m_sourceManager->findSourceView(in.loc);
        if (!m_currentView)
        {
            return UnownedStringSlice();
        }
    }

    const auto offset = m_currentView->getRange().getOffset(in.loc);
    SourceFile* sourceFile = m_currentView->getSourceFile();

    return UnownedStringSlice(sourceFile->getContent().begin() + offset, in.length);
}

UnownedStringSlice JSONContainer::getString(const JSONValue& in)
{
    switch (in.type)
    {
    case JSONValue::Type::StringValue:
        {
            return getStringFromKey(in.stringKey);
        }
    case JSONValue::Type::StringLexeme:
        {
            auto slice = getTransientString(in);
            auto handle = m_slicePool.add(slice);
            return m_slicePool.getSlice(handle);
        }
    case JSONValue::Type::StringRepresentation:
        {
            return StringRepresentation::asSlice(in.stringRep);
        }
    case JSONValue::Type::Null:
        {
            return UnownedStringSlice();
        }
    default:
        break;
    }

    SLANG_ASSERT(!"Not a string type");
    return UnownedStringSlice();
}

UnownedStringSlice JSONContainer::getTransientString(const JSONValue& in)
{
    switch (in.type)
    {
    case JSONValue::Type::StringRepresentation:
        {
            return StringRepresentation::asSlice(in.stringRep);
        }
    case JSONValue::Type::StringValue:
        {
            return getStringFromKey(in.stringKey);
        }
    case JSONValue::Type::StringLexeme:
        {
            StringEscapeHandler* handler =
                StringEscapeUtil::getHandler(StringEscapeUtil::Style::JSON);

            UnownedStringSlice lexeme = getLexeme(in);
            UnownedStringSlice unquoted = StringEscapeUtil::unquote(handler, lexeme);

            if (handler->isUnescapingNeeeded(unquoted))
            {
                m_buf.clear();
                handler->appendUnescaped(unquoted, m_buf);
                return m_buf.getUnownedSlice();
            }
            else
            {
                return unquoted;
            }
        }
    case JSONValue::Type::Null:
        {
            return UnownedStringSlice();
        }
    }

    SLANG_ASSERT(!"Not a string type");
    return UnownedStringSlice();
}

JSONKey JSONContainer::getStringKey(const JSONValue& in)
{
    return (in.type == JSONValue::Type::StringValue) ? in.stringKey
                                                     : getKey(getTransientString(in));
}

bool JSONContainer::asBool(const JSONValue& value)
{
    switch (value.type)
    {
    case JSONValue::Type::IntegerLexeme:
        return asInteger(value) != 0;
    case JSONValue::Type::FloatLexeme:
        return asFloat(value) != 0.0;
    default:
        return value.asBool();
    }
}

JSONValue JSONContainer::asValue(const JSONValue& inValue)
{
    JSONValue value = inValue;
    switch (value.type)
    {
    case JSONValue::Type::StringLexeme:
        {
            const UnownedStringSlice slice = getTransientString(inValue);
            value.stringKey = getKey(slice);
            value.type = JSONValue::Type::StringValue;
            break;
        }
    case JSONValue::Type::IntegerLexeme:
        {
            value.floatValue = value.asFloat();
            value.type = JSONValue::Type::IntegerValue;
            break;
        }
    case JSONValue::Type::FloatLexeme:
        {
            value.floatValue = value.asFloat();
            value.type = JSONValue::Type::FloatValue;
            break;
        }
    default:
        break;
    }

    return value;
}

void JSONContainer::_clearSourceManagerDependency(JSONValue* ioValues, Index count)
{
    for (Index i = 0; i < count; ++i)
    {
        auto& value = ioValues[i];
        value = asValue(value);
        value.loc = SourceLoc();
    }
}

void JSONContainer::clearSourceManagerDependency(JSONValue* ioValues, Index valuesCount)
{
    _clearSourceManagerDependency(ioValues, valuesCount);

    // We need to find ranges that are available
    for (auto& range : m_ranges)
    {
        switch (range.type)
        {
        case Range::Type::Array:
            {
                _clearSourceManagerDependency(
                    m_arrayValues.getBuffer() + range.startIndex,
                    range.count);
                break;
            }
        case Range::Type::Object:
            {
                const Index count = range.count;
                auto pairs = m_objectValues.getBuffer() + range.startIndex;

                for (Index i = 0; i < count; ++i)
                {
                    auto& pair = pairs[i];
                    pair.keyLoc = SourceLoc();
                    pair.value = asValue(pair.value);
                    pair.value.loc = SourceLoc();
                }
                break;
            }
        default:
            break;
        }
    }

    // Remove the source manager
    m_sourceManager = nullptr;
}

int64_t JSONContainer::asInteger(const JSONValue& value)
{
    switch (value.type)
    {
    case JSONValue::Type::IntegerLexeme:
        {
            UnownedStringSlice slice = getLexeme(value);
            int64_t intValue;
            if (SLANG_SUCCEEDED(StringUtil::parseInt64(slice, intValue)))
            {
                return intValue;
            }
            SLANG_ASSERT(!"Couldn't convert int");
            return 0;
        }
    case JSONValue::Type::FloatLexeme:
        return int64_t(asFloat(value));
    default:
        return value.asInteger();
    }
}

double JSONContainer::asFloat(const JSONValue& value)
{
    switch (value.type)
    {
    case JSONValue::Type::IntegerLexeme:
        return double(asInteger(value));
    case JSONValue::Type::FloatLexeme:
        {
            UnownedStringSlice slice = getLexeme(value);
            double floatValue;
            if (SLANG_SUCCEEDED(StringUtil::parseDouble(slice, floatValue)))
            {
                return floatValue;
            }
            SLANG_ASSERT(!"Couldn't convert double");
            return 0.0;
        }
    default:
        return value.asFloat();
    }
}

Index JSONContainer::findObjectIndex(const JSONValue& obj, JSONKey key) const
{
    auto pairs = getObject(obj);
    return pairs.findFirstIndex(
        [key](const JSONKeyValue& pair) -> bool { return pair.key == key; });
}

JSONValue JSONContainer::findObjectValue(const JSONValue& obj, JSONKey key) const
{
    auto pairs = getObject(obj);
    const Index index =
        pairs.findFirstIndex([key](const JSONKeyValue& pair) -> bool { return pair.key == key; });
    return (index >= 0) ? pairs[index].value : JSONValue::makeInvalid();
}

JSONValue& JSONContainer::getAt(const JSONValue& array, Index index)
{
    SLANG_ASSERT(array.type == JSONValue::Type::Array);
    const Range& range = m_ranges[array.rangeIndex];

    SLANG_ASSERT(index >= 0 && index < range.count);
    return m_arrayValues[range.startIndex + index];
}

void JSONContainer::addToArray(JSONValue& array, const JSONValue& value)
{
    SLANG_ASSERT(array.type == JSONValue::Type::Array);
    if (array.type == JSONValue::Type::Array)
    {
        // If it's empty
        if (array.rangeIndex == 0)
        {
            // We can just add to the end
            array.rangeIndex = _addRange(Range::Type::Array, m_arrayValues.getCount(), 1);
            m_arrayValues.add(value);
        }
        else
        {
            _add(m_ranges[array.rangeIndex], m_arrayValues, value);
        }
    }
}

Index JSONContainer::findKeyGlobalIndex(const JSONValue& obj, JSONKey key)
{
    SLANG_ASSERT(obj.type == JSONValue::Type::Object);
    if (obj.type != JSONValue::Type::Object)
    {
        return -1;
    }

    auto buf = m_objectValues.getBuffer();

    const Range& range = m_ranges[obj.rangeIndex];
    for (Index i = range.startIndex; i < range.startIndex + range.count; ++i)
    {
        if (buf[i].key == key)
        {
            return i;
        }
    }

    return -1;
}

Index JSONContainer::findKeyGlobalIndex(const JSONValue& obj, const UnownedStringSlice& slice)
{
    Index keyIndex = m_slicePool.findIndex(slice);
    if (keyIndex < 0)
    {
        return -1;
    }

    return findKeyGlobalIndex(obj, JSONKey(keyIndex));
}

void JSONContainer::_removeKey(JSONValue& obj, Index globalIndex)
{
    Range& range = m_ranges[obj.rangeIndex];
    const auto localIndex = globalIndex + range.startIndex;

    if (localIndex < range.count - 1)
    {
        auto localBuf = m_objectValues.getBuffer() + range.startIndex;
        ::memmove(
            (void*)(localBuf + localIndex),
            (void*)(localBuf + localIndex + 1),
            sizeof(*localBuf) * (range.count - (localIndex + 1)));
    }

    --range.count;
}

bool JSONContainer::removeKey(JSONValue& obj, JSONKey key)
{
    const Index globalIndex = findKeyGlobalIndex(obj, key);
    if (globalIndex >= 0)
    {
        _removeKey(obj, globalIndex);
        return true;
    }
    return false;
}

bool JSONContainer::removeKey(JSONValue& obj, const UnownedStringSlice& slice)
{
    const Index globalIndex = findKeyGlobalIndex(obj, slice);
    if (globalIndex >= 0)
    {
        _removeKey(obj, globalIndex);
        return true;
    }
    return false;
}

template<typename T>
/* static */ void JSONContainer::_add(Range& ioRange, List<T>& ioList, const T& value)
{
    // If we have capacity, we can add to the end
    if (ioRange.count < ioRange.capacity)
    {
        ioList[ioRange.startIndex + ioRange.count++] = value;
        return;
    }

    // If we are at the end, we can just add
    if (ioRange.startIndex + ioRange.capacity == ioList.getCount())
    {
        ioList.add(value);
        ioRange.capacity++;
        ioRange.count++;
        return;
    }

    // Okay we have no choice but to make new space at the end
    // So there's no place to add. We want to move to the end with an extra space.

    const Index newStartIndex = ioList.getCount();
    ioList.growToCount(newStartIndex + ioRange.count + 1);

    auto buffer = ioList.getBuffer();
    ::memmove(
        (void*)(buffer + newStartIndex),
        (void*)(buffer + ioRange.startIndex),
        sizeof(*buffer) * ioRange.count);

    buffer[newStartIndex + ioRange.count] = value;

    ioRange.startIndex = newStartIndex;
    ioRange.count++;
    ioRange.capacity++;
}


void JSONContainer::setKeyValue(JSONValue& obj, JSONKey key, const JSONValue& value, SourceLoc loc)
{
    SLANG_ASSERT(obj.type == JSONValue::Type::Object);
    if (obj.type != JSONValue::Type::Object)
    {
        return;
    }

    const JSONKeyValue keyValue{key, loc, value};
    if (obj.rangeIndex == 0)
    {
        // We need a new range and add to the end
        obj.rangeIndex = _addRange(Range::Type::Object, m_objectValues.getCount(), 1);
        m_objectValues.add(keyValue);
        return;
    }

    const Index globalIndex = findKeyGlobalIndex(obj, key);
    if (globalIndex >= 0)
    {
        auto& dst = m_objectValues[globalIndex];
        SLANG_ASSERT(dst.key == key);
        dst = keyValue;
        return;
    }

    Range& range = m_ranges[obj.rangeIndex];
    _add(range, m_objectValues, keyValue);
}

void JSONContainer::_destroyRange(Index rangeIndex)
{
    auto& range = m_ranges[rangeIndex];

    // If the range is at the end, shrink it
    switch (range.type)
    {
    case Range::Type::Array:
        {
            if (range.startIndex + range.capacity == m_arrayValues.getCount())
            {
                m_arrayValues.setCount(range.startIndex);
            }
            break;
        }
    case Range::Type::Object:
        {
            if (range.startIndex + range.capacity == m_objectValues.getCount())
            {
                m_objectValues.setCount(range.startIndex);
            }
            break;
        }
    default:
        break;
    }

    range.type = Range::Type::Destroyed;
    m_freeRangeIndices.add(rangeIndex);
}

void JSONContainer::destroy(JSONValue& value)
{
    if (value.needsDestroy())
    {
        _destroyRange(value.rangeIndex);
    }
    value.type = JSONValue::Type::Invalid;
}

void JSONContainer::destroyRecursively(JSONValue& inValue)
{
    if (!(inValue.needsDestroy() && m_ranges[inValue.rangeIndex].isActive()))
    {
        inValue.type = JSONValue::Type::Invalid;
        return;
    }

    inValue.type = JSONValue::Type::Invalid;

    List<Range> activeRanges;

    activeRanges.add(m_ranges[inValue.rangeIndex]);
    _destroyRange(inValue.rangeIndex);

    while (activeRanges.getCount())
    {
        const Range range = activeRanges.getLast();
        activeRanges.removeLast();

        auto type = range.type;
        const Index count = range.count;

        if (type == Range::Type::Array)
        {
            auto* buf = m_arrayValues.getBuffer() + range.startIndex;

            for (Index i = 0; i < count; ++i)
            {
                auto& value = buf[i];
                // If we have an active range, add to work list, and destroy
                if (value.needsDestroy() && m_ranges[value.rangeIndex].isActive())
                {
                    activeRanges.add(m_ranges[value.rangeIndex]);
                    _destroyRange(value.rangeIndex);
                }
                value.type = JSONValue::Type::Invalid;
            }
        }
        else
        {
            SLANG_ASSERT(type == Range::Type::Object);

            auto* buf = m_objectValues.getBuffer() + range.startIndex;

            for (Index i = 0; i < count; ++i)
            {
                auto& keyValue = buf[i];
                auto& value = keyValue.value;
                // We want to mark that it's in the list so that if we have a badly formed tree we
                // don't read
                if (value.needsDestroy() && m_ranges[value.rangeIndex].isActive())
                {
                    activeRanges.add(m_ranges[value.rangeIndex]);
                    _destroyRange(value.rangeIndex);
                }
                value.type = JSONValue::Type::Invalid;
            }
        }
    }
}

bool JSONContainer::areEqual(const JSONValue* a, const JSONValue* b, Index count)
{
    for (Index i = 0; i < count; ++i)
    {
        if (!areEqual(a[i], b[i]))
        {
            return false;
        }
    }

    return true;
}


/* static */ bool JSONContainer::_sameKeyOrder(
    const JSONKeyValue* a,
    const JSONKeyValue* b,
    Index count)
{
    for (Index i = 0; i < count; ++i)
    {
        if (a[i].key != b[i].key)
        {
            return false;
        }
    }
    return true;
}

bool JSONContainer::_areEqualOrderedKeys(const JSONKeyValue* a, const JSONKeyValue* b, Index count)
{
    for (Index i = 0; i < count; ++i)
    {
        const auto& curA = a[i];
        const auto& curB = b[i];

        if (curA.key != curB.key || !areEqual(curA.value, curB.value))
        {
            return false;
        }
    }
    return true;
}

bool JSONContainer::_areEqualValues(const JSONKeyValue* a, const JSONKeyValue* b, Index count)
{
    for (Index i = 0; i < count; ++i)
    {
        if (!areEqual(a[i].value, b[i].value))
        {
            return false;
        }
    }
    return true;
}

bool JSONContainer::areEqual(const JSONKeyValue* a, const JSONKeyValue* b, Index count)
{
    if (count == 0)
    {
        return true;
    }

    if (count == 1)
    {
        return _areEqualOrderedKeys(a, b, count);
    }
    else if (_sameKeyOrder(a, b, count))
    {
        return _areEqualValues(a, b, count);
    }
    else
    {
        // We need to compare with keys in the same order
        List<JSONKeyValue> sortedAs;
        sortedAs.addRange(a, count);

        List<JSONKeyValue> sortedBs;
        sortedBs.addRange(b, count);

        sortedAs.sort(
            [](const JSONKeyValue& a, const JSONKeyValue& b) -> bool { return a.key < b.key; });
        sortedBs.sort(
            [](const JSONKeyValue& a, const JSONKeyValue& b) -> bool { return a.key < b.key; });

        return _areEqualOrderedKeys(sortedAs.getBuffer(), sortedBs.getBuffer(), count);
    }
}

bool JSONContainer::areEqual(const JSONValue& a, const UnownedStringSlice& slice)
{
    return a.getKind() == JSONValue::Kind::String && getTransientString(a) == slice;
}

bool JSONContainer::areEqual(const JSONValue& a, const JSONValue& b)
{
    if (&a == &b)
    {
        return true;
    }

    if (a.type == b.type)
    {
        switch (a.type)
        {
        default:
        // Invalid are never equal
        case JSONValue::Type::Invalid:
            return false;
        case JSONValue::Type::True:
        case JSONValue::Type::False:
        case JSONValue::Type::Null:
            {
                return true;
            }
        case JSONValue::Type::IntegerLexeme:
            return asInteger(a) == asInteger(b);
        case JSONValue::Type::FloatLexeme:
            return asFloat(a) == asFloat(b);
        case JSONValue::Type::StringLexeme:
            {
                // If the lexemes are equal they are equal
                UnownedStringSlice lexemeA = getLexeme(a);
                UnownedStringSlice lexemeB = getLexeme(b);
                // Else we want to decode the string to be sure if they are equal.
                return lexemeA == lexemeB || getStringKey(a) == getStringKey(b);
            }
        case JSONValue::Type::IntegerValue:
            return a.intValue == b.intValue;
        case JSONValue::Type::FloatValue:
            return a.floatValue == b.floatValue;
        case JSONValue::Type::StringValue:
            return a.stringKey == b.stringKey;
        case JSONValue::Type::StringRepresentation:
            {
                return a.stringRep == b.stringRep || StringRepresentation::asSlice(a.stringRep) ==
                                                         StringRepresentation::asSlice(b.stringRep);
            }
        case JSONValue::Type::Array:
            {
                if (a.rangeIndex == b.rangeIndex)
                {
                    return true;
                }
                auto arrayA = getArray(a);
                auto arrayB = getArray(b);

                const Index count = arrayA.getCount();
                return (count == arrayB.getCount()) &&
                       areEqual(arrayA.getBuffer(), arrayB.getBuffer(), count);
            }
        case JSONValue::Type::Object:
            {
                if (a.rangeIndex == b.rangeIndex)
                {
                    return true;
                }
                const auto aValues = getObject(a);
                const auto bValues = getObject(b);

                const Index count = aValues.getCount();
                return (count == bValues.getCount()) &&
                       areEqual(aValues.getBuffer(), bValues.getBuffer(), count);
            }
        }
    }

    // If they are the same kind, and float/int/string we can convert to compare
    const JSONValue::Kind kind = a.getKind();
    if (kind == b.getKind())
    {
        switch (kind)
        {
        case JSONValue::Kind::String:
            return getStringKey(a) == getStringKey(b);
        case JSONValue::Kind::Integer:
            return asInteger(a) == asInteger(b);
        case JSONValue::Kind::Float:
            return asFloat(a) == asFloat(b);
        default:
            break;
        }
    }

    return false;
}

void JSONContainer::traverseRecursively(const JSONValue& value, JSONListener* listener)
{
    typedef JSONValue::Type Type;

    switch (value.type)
    {
    case Type::True:
        return listener->addBoolValue(true, value.loc);
    case Type::False:
        return listener->addBoolValue(false, value.loc);
    case Type::Null:
        return listener->addNullValue(value.loc);

    case Type::StringLexeme:
        return listener->addLexemeValue(JSONTokenType::StringLiteral, getLexeme(value), value.loc);
    case Type::IntegerLexeme:
        return listener->addLexemeValue(JSONTokenType::IntegerLiteral, getLexeme(value), value.loc);
    case Type::FloatLexeme:
        return listener->addLexemeValue(JSONTokenType::FloatLiteral, getLexeme(value), value.loc);

    case Type::IntegerValue:
        return listener->addIntegerValue(value.intValue, value.loc);
    case Type::FloatValue:
        return listener->addFloatValue(value.floatValue, value.loc);
    case Type::StringValue:
        {
            const auto slice = getStringFromKey(value.stringKey);
            return listener->addStringValue(slice, value.loc);
        }
    case Type::StringRepresentation:
        {
            return listener->addStringValue(getTransientString(value), value.loc);
        }
    case Type::Array:
        {
            listener->startArray(value.loc);

            const auto arr = getArray(value);

            for (const auto& arrayValue : arr)
            {
                traverseRecursively(arrayValue, listener);
            }

            listener->endArray(SourceLoc());
            break;
        }
    case Type::Object:
        {
            listener->startObject(value.loc);

            const auto obj = getObject(value);

            for (const auto& objKeyValue : obj)
            {
                // Emit the key
                const auto keyString = getStringFromKey(objKeyValue.key);
                listener->addUnquotedKey(keyString, objKeyValue.keyLoc);

                // Emit the value associated with the key
                traverseRecursively(objKeyValue.value, listener);
            }

            listener->endObject(SourceLoc());
            break;
        }
    default:
        {
            SLANG_ASSERT(!"Invalid type");
            return;
        }
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                          JSONBuilder

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

JSONBuilder::JSONBuilder(JSONContainer* container, Flags flags)
    : m_container(container), m_flags(flags)
{
    m_state.m_kind = State::Kind::Root;
    m_state.m_startIndex = 0;
    m_state.resetKey();

    m_rootValue.reset();
}

void JSONBuilder::reset()
{
    // Reset the state
    m_state.m_kind = State::Kind::Root;
    m_state.m_startIndex = 0;
    m_state.resetKey();

    // Clear the work values
    m_rootValue.reset();

    // Clear the lists
    m_stateStack.clear();
    m_values.clear();
    m_keyValues.clear();
}

void JSONBuilder::_popState()
{
    SLANG_ASSERT(m_stateStack.getCount() > 0);

    // Reset the end depending on typpe
    switch (m_state.m_kind)
    {
    case State::Kind::Array:
        {
            m_values.setCount(m_state.m_startIndex);
            break;
        }
    case State::Kind::Object:
        {
            m_keyValues.setCount(m_state.m_startIndex);
            break;
        }
    }

    // Pop from the stack
    m_state = m_stateStack.getLast();
    m_stateStack.removeLast();
}

Index JSONBuilder::_findKeyIndex(JSONKey key) const
{
    SLANG_ASSERT(m_state.m_kind == State::Kind::Object);
    const Index count = m_keyValues.getCount();
    for (Index i = m_state.m_startIndex; i < count; ++i)
    {
        auto& keyValue = m_keyValues[i];
        // If we find the key return it's index
        if (keyValue.key == key)
        {
            return i;
        }
    }
    return -1;
}

void JSONBuilder::_add(const JSONValue& value)
{
    SLANG_ASSERT(value.isValid());
    switch (m_state.m_kind)
    {
    case State::Kind::Root:
        {
            SLANG_ASSERT(!m_rootValue.isValid());
            m_rootValue = value;
            break;
        }
    case State::Kind::Array:
        {
            m_values.add(value);
            break;
        }
    case State::Kind::Object:
        {
            SLANG_ASSERT(m_state.hasKey());

            JSONKeyValue keyValue;
            keyValue.key = m_state.m_key;
            keyValue.keyLoc = m_state.m_keyLoc;
            keyValue.value = value;

            const Index index = _findKeyIndex(keyValue.key);
            if (index >= 0)
            {
                m_keyValues[index] = keyValue;
            }
            else
            {
                m_keyValues.add(keyValue);
            }

            m_state.resetKey();
            break;
        }
    }
}

void JSONBuilder::startObject(SourceLoc loc)
{
    m_stateStack.add(m_state);
    m_state.m_kind = State::Kind::Object;
    m_state.m_startIndex = m_keyValues.getCount();
    m_state.m_loc = loc;
    m_state.resetKey();
}

void JSONBuilder::endObject(SourceLoc loc)
{
    SLANG_UNUSED(loc);

    SLANG_ASSERT(m_state.m_kind == State::Kind::Object);

    const Index count = m_keyValues.getCount() - m_state.m_startIndex;
    const JSONValue value = m_container->createObject(
        m_keyValues.getBuffer() + m_state.m_startIndex,
        count,
        m_state.m_loc);

    // Pop current state
    _popState();
    // Add the value to the current state
    _add(value);
}

void JSONBuilder::startArray(SourceLoc loc)
{
    m_stateStack.add(m_state);
    m_state.m_kind = State::Kind::Array;
    m_state.m_startIndex = m_values.getCount();
    m_state.m_loc = loc;
    m_state.resetKey();
}

void JSONBuilder::endArray(SourceLoc loc)
{
    SLANG_UNUSED(loc);

    SLANG_ASSERT(m_state.m_kind == State::Kind::Array);

    const Index count = m_values.getCount() - m_state.m_startIndex;
    const JSONValue value =
        m_container->createArray(m_values.getBuffer() + m_state.m_startIndex, count, m_state.m_loc);

    // Pop current state
    _popState();
    // Add the value to the current state
    _add(value);
}

void JSONBuilder::addQuotedKey(const UnownedStringSlice& key, SourceLoc loc)
{
    // We need to decode
    m_work.clear();
    StringEscapeHandler* handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::JSON);
    StringEscapeUtil::appendUnquoted(handler, key, m_work);
    addUnquotedKey(m_work.getUnownedSlice(), loc);
}

void JSONBuilder::addUnquotedKey(const UnownedStringSlice& key, SourceLoc loc)
{
    SLANG_ASSERT(!m_state.hasKey());
    m_state.setKey(m_container->getKey(key), loc);
}

void JSONBuilder::addLexemeValue(JSONTokenType type, const UnownedStringSlice& value, SourceLoc loc)
{
    switch (type)
    {
    case JSONTokenType::True:
        return _add(JSONValue::makeBool(true, loc));
    case JSONTokenType::False:
        return _add(JSONValue::makeBool(false, loc));
    case JSONTokenType::Null:
        return _add(JSONValue::makeNull(loc));

    case JSONTokenType::IntegerLiteral:
        {
            if (m_flags & Flag::ConvertLexemes)
            {
                int64_t intValue = -1;
                auto res = StringUtil::parseInt64(value, intValue);
                SLANG_UNUSED(res);
                SLANG_ASSERT(SLANG_SUCCEEDED(res));
                _add(JSONValue::makeInt(intValue, loc));
            }
            else
            {
                SLANG_ASSERT(loc.isValid());
                _add(JSONValue::makeLexeme(JSONValue::Type::IntegerLexeme, loc, value.getLength()));
            }
            break;
        }
    case JSONTokenType::FloatLiteral:
        {
            if (m_flags & Flag::ConvertLexemes)
            {
                double floatValue = 0;
                auto res = StringUtil::parseDouble(value, floatValue);
                SLANG_UNUSED(res);
                SLANG_ASSERT(SLANG_SUCCEEDED(res));
                _add(JSONValue::makeFloat(floatValue, loc));
            }
            else
            {
                SLANG_ASSERT(loc.isValid());
                _add(JSONValue::makeLexeme(JSONValue::Type::FloatLexeme, loc, value.getLength()));
            }
            break;
        }
    case JSONTokenType::StringLiteral:
        {
            if (m_flags & Flag::ConvertLexemes)
            {
                auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::JSON);
                StringBuilder buf;
                StringEscapeUtil::appendUnquoted(handler, value, buf);

                _add(m_container->createString(buf.getUnownedSlice(), loc));
            }
            else
            {
                SLANG_ASSERT(loc.isValid());
                _add(JSONValue::makeLexeme(JSONValue::Type::StringLexeme, loc, value.getLength()));
            }
            break;
        }
    default:
        {
            SLANG_ASSERT(!"Unhandled type");
        }
    }
}

void JSONBuilder::addIntegerValue(int64_t value, SourceLoc loc)
{
    _add(JSONValue::makeInt(value, loc));
}

void JSONBuilder::addFloatValue(double value, SourceLoc loc)
{
    _add(JSONValue::makeFloat(value, loc));
}

void JSONBuilder::addBoolValue(bool value, SourceLoc loc)
{
    _add(JSONValue::makeBool(value, loc));
}

void JSONBuilder::addStringValue(const UnownedStringSlice& slice, SourceLoc loc)
{
    _add(m_container->createString(slice, loc));
}

void JSONBuilder::addNullValue(SourceLoc loc)
{
    _add(JSONValue::makeNull(loc));
}

} // namespace Slang
