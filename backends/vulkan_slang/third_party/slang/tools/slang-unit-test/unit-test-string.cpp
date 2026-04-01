// unit-test-path.cpp

#include "../../source/core/slang-string-util.h"
#include "unit-test/slang-unit-test.h"

// #include <math.h>

#include <sstream>

using namespace Slang;

static bool _areEqual(
    const List<UnownedStringSlice>& lines,
    const UnownedStringSlice* checkLines,
    Int checkLinesCount)
{
    if (checkLinesCount != lines.getCount())
    {
        return false;
    }

    for (Int i = 0; i < checkLinesCount; ++i)
    {
        if (lines[i] != checkLines[i])
        {
            return false;
        }
    }
    return true;
}

static bool _checkLines(
    const UnownedStringSlice& input,
    const UnownedStringSlice* checkLines,
    Int checkLinesCount)
{
    List<UnownedStringSlice> lines;
    StringUtil::calcLines(input, lines);
    return _areEqual(lines, checkLines, checkLinesCount);
}

static bool _checkLineParser(const UnownedStringSlice& input)
{
    UnownedStringSlice remaining(input), line;
    for (const auto parserLine : LineParser(input))
    {
        if (!StringUtil::extractLine(remaining, line) || line != parserLine)
        {
            return false;
        }
    }
    return StringUtil::extractLine(remaining, line) == false;
}

static void _append(double v, StringBuilder& buf)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.setf(std::ios::fixed, std::ios::floatfield);
    stream.precision(20);

    stream << std::scientific << v;

    buf << stream.str().c_str();
}

// Unit of least precision
static int64_t _calcULPDistance(double a, double b)
{
    // Save work if the floats are equal.
    // Also handles +0 == -0
    if (a == b)
    {
        return 0;
    }

    const int64_t max = int64_t((~uint64_t(0)) >> 1);

#if 0
    // Max distance for NaN
    if (isnan(a) || isnan(b))
    {
        return max;
    }

    // If one's infinite and they're not equal, max distance.
    if (isinf(a) || isinf(b))
    {
        return max;
    }
#endif

    int64_t ia, ib;
    memcpy(&ia, &a, sizeof(a));
    memcpy(&ib, &b, sizeof(b));

    // Don't compare differently-signed floats.
    if ((ia < 0) != (ib < 0))
    {
        return max;
    }

    // Return the absolute value of the distance in ULPs.
    int64_t distance = ia - ib;
    return distance < 0 ? -distance : distance;
}

static bool _areApproximatelyEqual(
    double a,
    double b,
    double fixedEpsilon = 1e-10,
    int ulpsEpsilon = 100)
{
    // Handle the near-zero case.
    const double difference = abs(a - b);
    if (difference <= fixedEpsilon)
    {
        return true;
    }

    return _calcULPDistance(a, b) <= ulpsEpsilon;
}

SLANG_UNIT_TEST(string)
{
    {
        UnownedStringSlice checkLines[] = {UnownedStringSlice::fromLiteral("")};
        SLANG_CHECK(_checkLines(
            UnownedStringSlice::fromLiteral(""),
            checkLines,
            SLANG_COUNT_OF(checkLines)));
    }
    {
        // Will emit no lines
        SLANG_CHECK(_checkLines(UnownedStringSlice(nullptr, nullptr), nullptr, 0));
    }
    {
        // Two lines - both empty
        UnownedStringSlice checkLines[] = {UnownedStringSlice(), UnownedStringSlice()};
        SLANG_CHECK(_checkLines(
            UnownedStringSlice::fromLiteral("\n"),
            checkLines,
            SLANG_COUNT_OF(checkLines)));
    }
    {
        UnownedStringSlice checkLines[] = {
            UnownedStringSlice::fromLiteral("Hello"),
            UnownedStringSlice::fromLiteral("World!")};
        SLANG_CHECK(_checkLines(
            UnownedStringSlice::fromLiteral("Hello\nWorld!"),
            checkLines,
            SLANG_COUNT_OF(checkLines)));
    }
    {
        UnownedStringSlice checkLines[] = {
            UnownedStringSlice::fromLiteral("Hello"),
            UnownedStringSlice::fromLiteral("World!"),
            UnownedStringSlice()};
        SLANG_CHECK(_checkLines(
            UnownedStringSlice::fromLiteral("Hello\n\rWorld!\n"),
            checkLines,
            SLANG_COUNT_OF(checkLines)));
    }

    {
        SLANG_CHECK(_checkLineParser(UnownedStringSlice::fromLiteral("Hello\n\rWorld!\n")));
        SLANG_CHECK(_checkLineParser(UnownedStringSlice::fromLiteral("\n")));
        SLANG_CHECK(_checkLineParser(UnownedStringSlice::fromLiteral("")));
    }
    {
        Int value;
        SLANG_CHECK(
            SLANG_SUCCEEDED(StringUtil::parseInt(UnownedStringSlice("-10"), value)) &&
            value == -10);
        SLANG_CHECK(
            SLANG_SUCCEEDED(StringUtil::parseInt(UnownedStringSlice("0"), value)) && value == 0);
        SLANG_CHECK(
            SLANG_SUCCEEDED(StringUtil::parseInt(UnownedStringSlice("-0"), value)) && value == 0);

        SLANG_CHECK(
            SLANG_SUCCEEDED(StringUtil::parseInt(UnownedStringSlice("13824"), value)) &&
            value == 13824);
        SLANG_CHECK(
            SLANG_SUCCEEDED(StringUtil::parseInt(UnownedStringSlice("-13824"), value)) &&
            value == -13824);
    }

    {
        UnownedStringSlice values[] = {
            UnownedStringSlice("hello"),
            UnownedStringSlice("world"),
            UnownedStringSlice("!")};
        ArrayView<UnownedStringSlice> valuesView(values, SLANG_COUNT_OF(values));

        List<UnownedStringSlice> checkValues;
        StringBuilder builder;

        {
            builder.clear();
            StringUtil::join(values, 0, ',', builder);
            SLANG_CHECK(builder == "");
        }

        {
            builder.clear();
            StringUtil::join(values, 1, ',', builder);
            SLANG_CHECK(builder == "hello");

            StringUtil::split(builder.getUnownedSlice(), ',', checkValues);
            SLANG_CHECK(checkValues.getArrayView() == ArrayView<UnownedStringSlice>(values, 1));
        }

        {
            builder.clear();
            StringUtil::join(values, 2, ',', builder);
            SLANG_CHECK(builder == "hello,world");

            StringUtil::split(builder.getUnownedSlice(), ',', checkValues);
            SLANG_CHECK(checkValues.getArrayView() == ArrayView<UnownedStringSlice>(values, 2));
        }

        {
            builder.clear();
            StringUtil::join(values, 3, UnownedStringSlice("ab"), builder);
            SLANG_CHECK(builder == "helloabworldab!");

            StringUtil::split(builder.getUnownedSlice(), UnownedStringSlice("ab"), checkValues);
            SLANG_CHECK(checkValues.getArrayView() == ArrayView<UnownedStringSlice>(values, 3));
        }
    }
    {

        List<double> values;
        values.add(0.0);
        values.add(-0.0);

        for (Index i = -300; i < 300; ++i)
        {
            double value = pow(10, i);

            values.add(value);
            values.add(-value);

            values.addRange(value / 3);
            values.addRange(-value / 3);
        }

        StringBuilder buf;

        for (auto value : values)
        {
            buf.clear();
            _append(value, buf);

            UnownedStringSlice slice = buf.getUnownedSlice();

            double parsedValue;
            SlangResult res = StringUtil::parseDouble(slice, parsedValue);

            auto ulpsParsed = _calcULPDistance(value, parsedValue);

            SLANG_CHECK(SLANG_SUCCEEDED(res));

            // Check that they are equal
            SLANG_CHECK(_areApproximatelyEqual(value, parsedValue));
        }
    }
    {
        List<int64_t> values;
        values.add(0);

        for (Index i = 0; i < 63; ++i)
        {
            auto value = int64_t(1) << i;

            values.add(value);
            values.add(-value);
        }

        StringBuilder buf;

        for (auto value : values)
        {
            buf.clear();
            buf << value;


            int64_t parsedValue;

            UnownedStringSlice slice = buf.getUnownedSlice();
            SlangResult res = StringUtil::parseInt64(slice, parsedValue);

            SLANG_CHECK(SLANG_SUCCEEDED(res));

            // Check that they are equal
            SLANG_CHECK(value == parsedValue);
        }
    }
}
