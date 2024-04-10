#pragma once
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <c10/util/bit_cast.h>
#include <c10/util/irange.h>
#include <gtest/gtest.h>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <complex>
#include <math.h>
#include <float.h>
#include <algorithm>

#if defined(CPU_CAPABILITY_AVX512)
#define CACHE_LINE 64
#else
#define CACHE_LINE 32
#endif

#if defined(__GNUC__)
#define CACHE_ALIGN __attribute__((aligned(CACHE_LINE)))
#define not_inline __attribute__((noinline))
#elif defined(_WIN32)
#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#define not_inline __declspec(noinline)
#else
CACHE_ALIGN #define
#define not_inline
#endif
#if defined(CPU_CAPABILITY_DEFAULT) || defined(_MSC_VER)
#define TEST_AGAINST_DEFAULT 1
#elif !defined(CPU_CAPABILITY_AVX512) && !defined(CPU_CAPABILITY_AVX2) && !defined(CPU_CAPABILITY_VSX) && !defined(CPU_CAPABILITY_ZVECTOR)
#define TEST_AGAINST_DEFAULT 1
#else
#undef TEST_AGAINST_DEFAULT
#endif
#undef NAME_INFO
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define NAME_INFO(name) TOSTRING(name) " " TOSTRING(__FILE__) ":" TOSTRING(__LINE__)

#define RESOLVE_OVERLOAD(...)                                  \
  [](auto&&... args) -> decltype(auto) {                       \
    return __VA_ARGS__(std::forward<decltype(args)>(args)...); \
  }

#if defined(CPU_CAPABILITY_ZVECTOR) || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_AVX2) || \
  defined(CPU_CAPABILITY_AVX512) && (defined(__GNUC__) || defined(__GNUG__))
#undef CHECK_DEQUANT_WITH_LOW_PRECISION
#define CHECK_WITH_FMA 1
#elif !defined(CPU_CAPABILITY_VSX) && !defined(CPU_CAPABILITY_AVX2)
#undef CHECK_DEQUANT_WITH_LOW_PRECISION
#undef CHECK_WITH_FMA
#else
#define CHECK_DEQUANT_WITH_LOW_PRECISION 1
#undef CHECK_WITH_FMA
#endif

template<typename T>
using Complex = typename c10::complex<T>;

template <typename T>
using VecType = typename at::vec::Vectorized<T>;

using vfloat = VecType<float>;
using vdouble = VecType<double>;
using vcomplex = VecType<Complex<float>>;
using vcomplexDbl = VecType<Complex<double>>;
using vlong = VecType<int64_t>;
using vint = VecType<int32_t>;
using vshort = VecType<int16_t>;
using vqint8 = VecType<c10::qint8>;
using vquint8 = VecType<c10::quint8>;
using vqint = VecType<c10::qint32>;
using vBFloat16 = VecType<c10::BFloat16>;
using vHalf = VecType<c10::Half>;

template <typename T>
using ValueType = typename T::value_type;

template <int N>
struct BitStr
{
    using type = uintmax_t;
};

template <>
struct BitStr<8>
{
    using type = uint64_t;
};

template <>
struct BitStr<4>
{
    using type = uint32_t;
};

template <>
struct BitStr<2>
{
    using type = uint16_t;
};

template <>
struct BitStr<1>
{
    using type = uint8_t;
};

template <typename T>
using BitType = typename BitStr<sizeof(T)>::type;

template<typename T>
struct VecTypeHelper {
    using holdType = typename T::value_type;
    using memStorageType = typename T::value_type;
    static constexpr int holdCount = T::size();
    static constexpr int unitStorageCount = 1;
};

template<>
struct VecTypeHelper<vcomplex> {
    using holdType = Complex<float>;
    using memStorageType = float;
    static constexpr int holdCount = vcomplex::size();
    static constexpr int unitStorageCount = 2;
};

template<>
struct VecTypeHelper<vcomplexDbl> {
    using holdType = Complex<double>;
    using memStorageType = double;
    static constexpr int holdCount = vcomplexDbl::size();
    static constexpr int unitStorageCount = 2;
};

template<>
struct VecTypeHelper<vqint8> {
    using holdType = c10::qint8;
    using memStorageType = typename c10::qint8::underlying;
    static constexpr int holdCount = vqint8::size();
    static constexpr int unitStorageCount = 1;
};

template<>
struct VecTypeHelper<vquint8> {
    using holdType = c10::quint8;
    using memStorageType = typename c10::quint8::underlying;
    static constexpr int holdCount = vquint8::size();
    static constexpr int unitStorageCount = 1;
};

template<>
struct VecTypeHelper<vqint> {
    using holdType = c10::qint32;
    using memStorageType = typename c10::qint32::underlying;
    static constexpr int holdCount = vqint::size();
    static constexpr int unitStorageCount = 1;
};

template<>
struct VecTypeHelper<vBFloat16> {
    using holdType = c10::BFloat16;
    using memStorageType = typename vBFloat16::value_type;
    static constexpr int holdCount = vBFloat16::size();
    static constexpr int unitStorageCount = 1;
};

template<>
struct VecTypeHelper<vHalf> {
    using holdType = c10::Half;
    using memStorageType = typename vHalf::value_type;
    static constexpr int holdCount = vHalf::size();
    static constexpr int unitStorageCount = 1;
};

template <typename T>
using UholdType = typename VecTypeHelper<T>::holdType;

template <typename T>
using UvalueType = typename VecTypeHelper<T>::memStorageType;

template <class T, size_t N>
constexpr size_t size(T(&)[N]) {
    return N;
}

template <typename Filter, typename T>
typename std::enable_if_t<std::is_same<Filter, std::nullptr_t>::value, void>
call_filter(Filter filter, T& val) {}

template <typename Filter, typename T>
typename std::enable_if_t< std::is_same<Filter, std::nullptr_t>::value, void>
call_filter(Filter filter, T& first, T& second) { }

template <typename Filter, typename T>
typename std::enable_if_t< std::is_same<Filter, std::nullptr_t>::value, void>
call_filter(Filter filter, T& first, T& second, T& third) {  }

template <typename Filter, typename T>
typename std::enable_if_t<
    !std::is_same<Filter, std::nullptr_t>::value, void>
    call_filter(Filter filter, T& val) {
    return filter(val);
}

template <typename Filter, typename T>
typename std::enable_if_t<
    !std::is_same<Filter, std::nullptr_t>::value, void>
    call_filter(Filter filter, T& first, T& second) {
    return filter(first, second);
}

template <typename Filter, typename T>
typename std::enable_if_t<
    !std::is_same<Filter, std::nullptr_t>::value, void>
    call_filter(Filter filter, T& first, T& second, T& third) {
    return filter(first, second, third);
}

template <typename T>
struct DomainRange {
    T start;  // start [
    T end;    // end is not included. one could use  nextafter for including his end case for tests
};

template <typename T>
struct CustomCheck {
    std::vector<UholdType<T>> Args;
    UholdType<T> expectedResult;
};

template <typename T>
struct CheckWithinDomains {
    // each argument takes domain Range
    std::vector<DomainRange<T>> ArgsDomain;
    // check with error tolerance
    bool CheckWithTolerance = false;
    T ToleranceError = (T)0;
};

template <typename T>
std::ostream& operator<<(std::ostream& stream, const CheckWithinDomains<T>& dmn) {
    stream << "Domain: ";
    if (dmn.ArgsDomain.size() > 0) {
        for (const DomainRange<T>& x : dmn.ArgsDomain) {
            if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
                stream << "\n{ " << static_cast<int>(x.start) << ", " << static_cast<int>(x.end) << " }";
            }
            else {
                stream << "\n{ " << x.start << ", " << x.end << " }";
            }
        }
    }
    else {
        stream << "default range";
    }
    if (dmn.CheckWithTolerance) {
        stream << "\nError tolerance: " << dmn.ToleranceError;
    }
    return stream;
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> check_both_nan(T x,
    T y) {
    return std::isnan(x) && std::isnan(y);
}

template <typename T>
std::enable_if_t<!std::is_floating_point<T>::value, bool> check_both_nan(T x,
    T y) {
    return false;
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> check_both_inf(T x,
    T y) {
    return std::isinf(x) && std::isinf(y);
}

template <typename T>
std::enable_if_t<!std::is_floating_point<T>::value, bool> check_both_inf(T x,
    T y) {
    return false;
}

template<typename T>
std::enable_if_t<!std::is_floating_point<T>::value, bool> check_both_big(T x, T y) {
    return false;
}

template<typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> check_both_big(T x, T y) {
    T cmax = std::is_same<T, float>::value ? static_cast<T>(1e+30) : static_cast<T>(1e+300);
    T cmin = std::is_same<T, float>::value ? static_cast<T>(-1e+30) : static_cast<T>(-1e+300);
    //only allow when one is inf
    bool x_inf = std::isinf(x);
    bool y_inf = std::isinf(y);
    bool px = x > 0;
    bool py = y > 0;
    return (px && x_inf && y >= cmax) || (py && y_inf && x >= cmax) ||
        (!px && x_inf && y <= cmin) || (!py && y_inf && x <= cmin);
}

template<class T> struct is_complex : std::false_type {};

template<class T> struct is_complex<Complex<T>> : std::true_type {};

template<typename T>
T safe_fpt_division(T f1, T f2)
{
    //code was taken from boost
    // Avoid overflow.
    if ((f2 < static_cast<T>(1)) && (f1 > f2 * std::numeric_limits<T>::max())) {
        return std::numeric_limits<T>::max();
    }
    // Avoid underflow.
    if ((f1 == static_cast<T>(0)) ||
        ((f2 > static_cast<T>(1)) && (f1 < f2 * std::numeric_limits<T>::min()))) {
        return static_cast<T>(0);
    }
    return f1 / f2;
}

template<class T>
std::enable_if_t<std::is_floating_point<T>::value, bool>
nearlyEqual(T a, T b, T tolerance) {
    if (check_both_nan<T>(a, b)) return true;
    if (check_both_big(a, b)) return true;
    T absA = std::abs(a);
    T absB = std::abs(b);
    T diff = std::abs(a - b);
    if (diff <= tolerance) {
        return true;
    }
    T d1 = safe_fpt_division<T>(diff, absB);
    T d2 = safe_fpt_division<T>(diff, absA);
    return (d1 <= tolerance) || (d2 <= tolerance);
}

template<class T>
std::enable_if_t<!std::is_floating_point<T>::value, bool>
nearlyEqual(T a, T b, T tolerance) {
    return a == b;
}

template <typename T>
T reciprocal(T x) {
    return 1 / x;
}

template <typename T>
T rsqrt(T x) {
    return 1 / std::sqrt(x);
}

template <typename T>
T frac(T x) {
  return x - std::trunc(x);
}

template <class T>
T maximum(const T& a, const T& b) {
    return (a > b) ? a : b;
}

template <class T>
T minimum(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template <class T>
T clamp(const T& a, const T& min, const T& max) {
    return a < min ? min : (a > max ? max : a);
}

template <class T>
T clamp_max(const T& a, const T& max) {
    return a > max ? max : a;
}

template <class T>
T clamp_min(const T& a, const T& min) {
    return a < min ? min : a;
}

template <class VT, size_t N>
void copy_interleave(VT(&vals)[N], VT(&interleaved)[N]) {
    static_assert(N % 2 == 0, "should be even");
    auto ptr1 = vals;
    auto ptr2 = vals + N / 2;
    for (size_t i = 0; i < N; i += 2) {
        interleaved[i] = *ptr1++;
        interleaved[i + 1] = *ptr2++;
    }
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> is_zero(T val) {
    return std::fpclassify(val) == FP_ZERO;
}

template <typename T>
std::enable_if_t<!std::is_floating_point<T>::value, bool> is_zero(T val) {
    return val == 0;
}

template <typename T>
void filter_clamp(T& f, T& s, T& t) {
    if (t < s) {
        std::swap(s, t);
    }
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, void> filter_fmod(T& a, T& b) {
    // This is to make sure fmod won't cause overflow when doing the div
    if (std::abs(b) < (T)1) {
      b = b < (T)0 ? (T)-1 : T(1);
    }
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, void> filter_fmadd(T& a, T& b, T& c) {
    // This is to setup a limit to make sure fmadd (a * b + c) won't overflow
    T max = std::sqrt(std::numeric_limits<T>::max()) / T(2.0);
    T min = ((T)0 - max);

    if (a > max) a = max;
    else if (a < min) a = min;

    if (b > max) b = max;
    else if (b < min) b = min;

    if (c > max) c = max;
    else if (c < min) c = min;
}

template <typename T>
void filter_zero(T& val) {
    val = is_zero(val) ? (T)1 : val;
}
template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, void> filter_zero(Complex<T>& val) {
    T rr = val.real();
    T ii = val.imag();
    rr = is_zero(rr) ? (T)1 : rr;
    ii = is_zero(ii) ? (T)1 : ii;
    val = Complex<T>(rr, ii);
}

template <typename T>
void filter_int_minimum(T& val) {
    if constexpr (!std::is_integral_v<T>) return;
    if (val == std::numeric_limits<T>::min()) {
        val = 0;
    }
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_add_overflow(T& a, T& b)
{
    //missing for complex
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_sub_overflow(T& a, T& b)
{
    //missing for complex
}

template <typename T>
std::enable_if_t < !is_complex<T>::value, void> filter_add_overflow(T& a, T& b) {
    if constexpr (std::is_integral_v<T> == false) return;
    T max = std::numeric_limits<T>::max();
    T min = std::numeric_limits<T>::min();
    // min <= (a +b) <= max;
    // min - b <= a  <= max - b
    if (b < 0) {
        if (a < min - b) {
            a = min - b;
        }
    }
    else {
        if (a > max - b) {
            a = max - b;
        }
    }
}

template <typename T>
std::enable_if_t < !is_complex<T>::value, void> filter_sub_overflow(T& a, T& b) {
    if constexpr (std::is_integral_v<T> == false) return;
    T max = std::numeric_limits<T>::max();
    T min = std::numeric_limits<T>::min();
    // min <= (a-b) <= max;
    // min + b <= a  <= max +b
    if (b < 0) {
        if (a > max + b) {
            a = max + b;
        }
    }
    else {
        if (a < min + b) {
            a = min + b;
        }
    }
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void>
filter_mult_overflow(T& val1, T& val2) {
    //missing
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void>
filter_div_ub(T& val1, T& val2) {
    //missing
    //at least consdier zero division
    auto ret = std::abs(val2);
    if (ret == 0) {
        val2 = T(1, 2);
    }
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, void>
filter_mult_overflow(T& val1, T& val2) {
    if constexpr (std::is_integral_v<T> == false) return;
    if (!is_zero(val2)) {
        T c = (std::numeric_limits<T>::max() - 1) / val2;
        if (std::abs(val1) >= c) {
            // correct first;
            val1 = c;
        }
    }  // is_zero
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, void>
filter_div_ub(T& val1, T& val2) {
    if (is_zero(val2)) {
        val2 = 1;
    }
    else if (std::is_integral<T>::value && val1 == std::numeric_limits<T>::min() && val2 == -1) {
        val2 = 1;
    }
}

struct TestSeed {
    TestSeed() : seed(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
    }
    TestSeed(uint64_t seed) : seed(seed) {
    }
    uint64_t getSeed() {
        return seed;
    }
    operator uint64_t () const {
        return seed;
    }

    TestSeed add(uint64_t index) {
        return TestSeed(seed + index);
    }
private:
    uint64_t seed;
};

template <typename T, bool is_floating_point = std::is_floating_point<T>::value, bool is_complex = is_complex<T>::value>
struct ValueGen
{
    std::uniform_int_distribution<int64_t> dis;
    std::mt19937 gen;
    ValueGen() : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed)
    {
    }
    ValueGen(T start, T stop, uint64_t seed = TestSeed())
    {
        gen = std::mt19937(seed);
        dis = std::uniform_int_distribution<int64_t>(start, stop);
    }
    T get()
    {
        return static_cast<T>(dis(gen));
    }
};

template <typename T>
struct ValueGen<T, true, false>
{
    std::mt19937 gen;
    std::normal_distribution<T> normal;
    std::uniform_int_distribution<int> roundChance;
    T _start;
    T _stop;
    bool use_sign_change = false;
    bool use_round = true;
    ValueGen() : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed)
    {
    }
    ValueGen(T start, T stop, uint64_t seed = TestSeed())
    {
        gen = std::mt19937(seed);
        T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);
        //make it  normal +-3sigma
        T divRange = static_cast<T>(6.0);
        T stdev = std::abs(stop / divRange - start / divRange);
        normal = std::normal_distribution<T>{ mean, stdev };
        // in real its hard to get rounded value
        // so we will force it by  uniform chance
        roundChance = std::uniform_int_distribution<int>(0, 5);
        _start = start;
        _stop = stop;
    }
    T get()
    {
        T a = normal(gen);
        //make rounded value ,too
        auto rChoice = roundChance(gen);
        if (rChoice == 1)
            a = std::round(a);
        if (a < _start)
            return nextafter(_start, _stop);
        if (a >= _stop)
            return nextafter(_stop, _start);
        return a;
    }
};

template <typename T>
struct ValueGen<Complex<T>, false, true>
{
    std::mt19937 gen;
    std::normal_distribution<T> normal;
    std::uniform_int_distribution<int> roundChance;
    T _start;
    T _stop;
    bool use_sign_change = false;
    bool use_round = true;
    ValueGen() : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max())
    {
    }
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), seed)
    {
    }
    ValueGen(T start, T stop, uint64_t seed = TestSeed())
    {
        gen = std::mt19937(seed);
        T mean = start * static_cast<T>(0.5) + stop * static_cast<T>(0.5);
        //make it  normal +-3sigma
        T divRange = static_cast<T>(6.0);
        T stdev = std::abs(stop / divRange - start / divRange);
        normal = std::normal_distribution<T>{ mean, stdev };
        // in real its hard to get rounded value
        // so we will force it by  uniform chance
        roundChance = std::uniform_int_distribution<int>(0, 5);
        _start = start;
        _stop = stop;
    }
    Complex<T> get()
    {
        T a = normal(gen);
        T b = normal(gen);
        //make rounded value ,too
        auto rChoice = roundChance(gen);
        rChoice = rChoice & 3;
        if (rChoice & 1)
            a = std::round(a);
        if (rChoice & 2)
            b = std::round(b);
        if (a < _start)
            a = nextafter(_start, _stop);
        else if (a >= _stop)
            a = nextafter(_stop, _start);
        if (b < _start)
            b = nextafter(_start, _stop);
        else if (b >= _stop)
            b = nextafter(_stop, _start);
        return Complex<T>(a, b);
    }
};

template<class T>
int getTrialCount(int test_trials, int domains_size) {
    int trialCount;
    int trial_default = 1;
    if (sizeof(T) <= 2) {
        //half coverage for byte
        trial_default = 128;
    }
    else {
        //2*65536
        trial_default = 2 * std::numeric_limits<uint16_t>::max();
    }
    trialCount = test_trials < 1 ? trial_default : test_trials;
    if (domains_size > 1) {
        trialCount = trialCount / domains_size;
        trialCount = trialCount < 1 ? 1 : trialCount;
    }
    return trialCount;
}

template <typename T, typename U = UvalueType<T>>
class TestCaseBuilder;

template <typename T, typename U = UvalueType<T>>
class TestingCase {
public:
    friend class TestCaseBuilder<T, U>;
    static TestCaseBuilder<T, U> getBuilder() { return TestCaseBuilder<T, U>{}; }
    bool checkSpecialValues() const {
        //this will be used to check nan, infs, and other special cases
        return specialCheck;
    }
    size_t getTrialCount() const { return trials; }
    bool isBitwise() const { return bitwise; }
    const std::vector<CheckWithinDomains<U>>& getDomains() const {
        return domains;
    }
    const std::vector<CustomCheck<T>>& getCustomChecks() const {
        return customCheck;
    }
    TestSeed getTestSeed() const {
        return testSeed;
    }
private:
    // if domains is empty we will test default
    std::vector<CheckWithinDomains<U>> domains;
    std::vector<CustomCheck<T>> customCheck;
    // its not used for now
    bool specialCheck = false;
    bool bitwise = false;  // test bitlevel
    size_t trials = 0;
    TestSeed testSeed;
};

template <typename T, typename U >
class TestCaseBuilder {
private:
    TestingCase<T, U> _case;
public:
    TestCaseBuilder<T, U>& set(bool bitwise, bool checkSpecialValues) {
        _case.bitwise = bitwise;
        _case.specialCheck = checkSpecialValues;
        return *this;
    }
    TestCaseBuilder<T, U>& setTestSeed(TestSeed seed) {
        _case.testSeed = seed;
        return *this;
    }
    TestCaseBuilder<T, U>& setTrialCount(size_t trial_count) {
        _case.trials = trial_count;
        return *this;
    }
    TestCaseBuilder<T, U>& addDomain(const CheckWithinDomains<U>& domainCheck) {
        _case.domains.emplace_back(domainCheck);
        return *this;
    }
    TestCaseBuilder<T, U>& addCustom(const CustomCheck<T>& customArgs) {
        _case.customCheck.emplace_back(customArgs);
        return *this;
    }
    TestCaseBuilder<T, U>& checkSpecialValues() {
        _case.specialCheck = true;
        return *this;
    }
    TestCaseBuilder<T, U>& compareBitwise() {
        _case.bitwise = true;
        return *this;
    }
    operator TestingCase<T, U> && () { return std::move(_case); }
};

template <typename T>
typename std::enable_if_t<!is_complex<T>::value&& std::is_unsigned<T>::value, T>
correctEpsilon(const T& eps)
{
    return eps;
}
template <typename T>
typename std::enable_if_t<!is_complex<T>::value && !std::is_unsigned<T>::value, T>
correctEpsilon(const T& eps)
{
    return std::abs(eps);
}
template <typename T>
typename std::enable_if_t<is_complex<Complex<T>>::value, T>
correctEpsilon(const Complex<T>& eps)
{
    return std::abs(eps);
}

template <typename T>
class AssertVectorized
{
public:
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual, const T& input0)
        : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), argSize(1)
    {
    }
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual, const T& input0, const T& input1)
        : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), arg1(input1), argSize(2)
    {
    }
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual, const T& input0, const T& input1, const T& input2)
        : additionalInfo(info), testSeed(seed), exp(expected), act(actual), arg0(input0), arg1(input1), arg2(input2), argSize(3)
    {
    }
    AssertVectorized(const std::string& info, TestSeed seed, const T& expected, const T& actual) : additionalInfo(info), testSeed(seed), exp(expected), act(actual)
    {
    }
    AssertVectorized(const std::string& info, const T& expected, const T& actual) : additionalInfo(info), exp(expected), act(actual), hasSeed(false)
    {
    }

    std::string getDetail(int index) const
    {
        using UVT = UvalueType<T>;
        std::stringstream stream;
        stream.precision(std::numeric_limits<UVT>::max_digits10);
        stream << "Failure Details:\n";
        stream << additionalInfo << "\n";
        if (hasSeed)
        {
            stream << "Test Seed to reproduce: " << testSeed << "\n";
        }
        if (argSize > 0)
        {
            stream << "Arguments:\n";
            stream << "#\t " << arg0 << "\n";
            if (argSize == 2)
            {
                stream << "#\t " << arg1 << "\n";
            }
            if (argSize == 3)
            {
                stream << "#\t " << arg2 << "\n";
            }
        }
        stream << "Expected:\n#\t" << exp << "\nActual:\n#\t" << act;
        stream << "\nFirst mismatch Index: " << index;
        return stream.str();
    }

    bool check(bool bitwise = false, bool checkWithTolerance = false, ValueType<T> toleranceEps = {}) const
    {
        using UVT = UvalueType<T>;
        using BVT = BitType<UVT>;
        UVT absErr = correctEpsilon(toleranceEps);
        constexpr int sizeX = VecTypeHelper<T>::holdCount * VecTypeHelper<T>::unitStorageCount;
        constexpr int unitStorageCount = VecTypeHelper<T>::unitStorageCount;
        CACHE_ALIGN UVT expArr[sizeX];
        CACHE_ALIGN UVT actArr[sizeX];
        exp.store(expArr);
        act.store(actArr);
        if (bitwise)
        {
            for (const auto i : c10::irange(sizeX)) {
                BVT b_exp = c10::bit_cast<BVT>(expArr[i]);
                BVT b_act = c10::bit_cast<BVT>(actArr[i]);
                EXPECT_EQ(b_exp, b_act) << getDetail(i / unitStorageCount);
                if (::testing::Test::HasFailure())
                    return true;
            }
        }
        else if (checkWithTolerance)
        {
            for (const auto i : c10::irange(sizeX)) {
                EXPECT_EQ(nearlyEqual<UVT>(expArr[i], actArr[i], absErr), true) << expArr[i] << "!=" << actArr[i] << "\n" << getDetail(i / unitStorageCount);
                if (::testing::Test::HasFailure())
                    return true;
            }
        }
        else
        {
            for (const auto i : c10::irange(sizeX)) {
                if constexpr (std::is_same_v<UVT, float>)
                {
                    if (!check_both_nan(expArr[i], actArr[i])) {
                        EXPECT_FLOAT_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                    }
                }
                else if constexpr (std::is_same_v<UVT, double>)
                {
                    if (!check_both_nan(expArr[i], actArr[i]))
                    {
                        EXPECT_DOUBLE_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                    }
                }
                else
                {
                    EXPECT_EQ(expArr[i], actArr[i]) << getDetail(i / unitStorageCount);
                }
                if (::testing::Test::HasFailure())
                    return true;
            }
        }
        return false;
    }

private:
    std::string additionalInfo;
    TestSeed testSeed;
    T exp;
    T act;
    T arg0;
    T arg1;
    T arg2;
    int argSize = 0;
    bool hasSeed = true;
};

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_unary(
    std::string testNameInfo,
    Op1 expectedFunction,
    Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
    using vec_type = T;
    using VT = ValueType<T>;
    using UVT = UvalueType<T>;
    constexpr int el_count = vec_type::size();
    CACHE_ALIGN VT vals[el_count];
    CACHE_ALIGN VT expected[el_count];
    bool bitwise = testCase.isBitwise();
    UVT default_start = std::is_floating_point<UVT>::value ? std::numeric_limits<UVT>::lowest() : std::numeric_limits<UVT>::min();
    UVT default_end = std::numeric_limits<UVT>::max();
    auto domains = testCase.getDomains();
    auto domains_size = domains.size();
    auto test_trials = testCase.getTrialCount();
    int trialCount = getTrialCount<UVT>(test_trials, domains_size);
    TestSeed seed = testCase.getTestSeed();
    uint64_t changeSeedBy = 0;
    for (const CheckWithinDomains<UVT>& dmn : domains) {
        size_t dmn_argc = dmn.ArgsDomain.size();
        UVT start = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        ValueGen<VT> generator(start, end, seed.add(changeSeedBy));
        for (C10_UNUSED const auto trial : c10::irange(trialCount)) {
            for (const auto k : c10::irange(el_count)) {
                vals[k] = generator.get();
                call_filter(filter, vals[k]);
                //map operator
                expected[k] = expectedFunction(vals[k]);
            }
            // test
            auto input = vec_type::loadu(vals);
            auto actual = actualFunction(input);
            auto vec_expected = vec_type::loadu(expected);
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input);
            if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;

        }// trial
        //inrease Seed
        changeSeedBy += 1;
    }
    for (auto& custom : testCase.getCustomChecks()) {
        auto args = custom.Args;
        if (args.size() > 0) {
            auto input = vec_type{ args[0] };
            auto actual = actualFunction(input);
            auto vec_expected = vec_type{ custom.expectedResult };
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input);
            if (vecAssert.check()) return;
        }
    }
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_binary(
    std::string testNameInfo,
    Op1 expectedFunction,
    Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
    using vec_type = T;
    using VT = ValueType<T>;
    using UVT = UvalueType<T>;
    constexpr int el_count = vec_type::size();
    CACHE_ALIGN VT vals0[el_count];
    CACHE_ALIGN VT vals1[el_count];
    CACHE_ALIGN VT expected[el_count];
    bool bitwise = testCase.isBitwise();
    UVT default_start = std::is_floating_point<UVT>::value ? std::numeric_limits<UVT>::lowest() : std::numeric_limits<UVT>::min();
    UVT default_end = std::numeric_limits<UVT>::max();
    auto domains = testCase.getDomains();
    auto domains_size = domains.size();
    auto test_trials = testCase.getTrialCount();
    int trialCount = getTrialCount<UVT>(test_trials, domains_size);
    TestSeed seed = testCase.getTestSeed();
    uint64_t changeSeedBy = 0;
    for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
        size_t dmn_argc = dmn.ArgsDomain.size();
        UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
        UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
        ValueGen<VT> generator0(start0, end0, seed.add(changeSeedBy));
        ValueGen<VT> generator1(start1, end1, seed.add(changeSeedBy + 1));
        for (C10_UNUSED const auto trial : c10::irange(trialCount)) {
            for (const auto k : c10::irange(el_count)) {
                vals0[k] = generator0.get();
                vals1[k] = generator1.get();
                call_filter(filter, vals0[k], vals1[k]);
                //map operator
                expected[k] = expectedFunction(vals0[k], vals1[k]);
            }
            // test
            auto input0 = vec_type::loadu(vals0);
            auto input1 = vec_type::loadu(vals1);
            auto actual = actualFunction(input0, input1);
            auto vec_expected = vec_type::loadu(expected);
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1);
            if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError))return;
        }// trial
        changeSeedBy += 1;
    }
    for (auto& custom : testCase.getCustomChecks()) {
        auto args = custom.Args;
        if (args.size() > 0) {
            auto input0 = vec_type{ args[0] };
            auto input1 = args.size() > 1 ? vec_type{ args[1] } : vec_type{ args[0] };
            auto actual = actualFunction(input0, input1);
            auto vec_expected = vec_type(custom.expectedResult);
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1);
            if (vecAssert.check()) return;
        }
    }
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_ternary(
    std::string testNameInfo,
    Op1 expectedFunction,
    Op2 actualFunction, const TestingCase<T>& testCase, Filter filter = {}) {
    using vec_type = T;
    using VT = ValueType<T>;
    using UVT = UvalueType<T>;
    constexpr int el_count = vec_type::size();
    CACHE_ALIGN VT vals0[el_count];
    CACHE_ALIGN VT vals1[el_count];
    CACHE_ALIGN VT vals2[el_count];
    CACHE_ALIGN VT expected[el_count];
    bool bitwise = testCase.isBitwise();
    UVT default_start = std::is_floating_point<UVT>::value ? std::numeric_limits<UVT>::lowest() : std::numeric_limits<UVT>::min();
    UVT default_end = std::numeric_limits<UVT>::max();
    auto domains = testCase.getDomains();
    auto domains_size = domains.size();
    auto test_trials = testCase.getTrialCount();
    int trialCount = getTrialCount<UVT>(test_trials, domains_size);
    TestSeed seed = testCase.getTestSeed();
    uint64_t changeSeedBy = 0;
    for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
        size_t dmn_argc = dmn.ArgsDomain.size();
        UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
        UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
        UVT start2 = dmn_argc > 2 ? dmn.ArgsDomain[2].start : default_start;
        UVT end2 = dmn_argc > 2 ? dmn.ArgsDomain[2].end : default_end;
        ValueGen<VT> generator0(start0, end0, seed.add(changeSeedBy));
        ValueGen<VT> generator1(start1, end1, seed.add(changeSeedBy + 1));
        ValueGen<VT> generator2(start2, end2, seed.add(changeSeedBy + 2));

        for (C10_UNUSED const auto trial : c10::irange(trialCount)) {
            for (const auto k : c10::irange(el_count)) {
                vals0[k] = generator0.get();
                vals1[k] = generator1.get();
                vals2[k] = generator2.get();
                call_filter(filter, vals0[k], vals1[k], vals2[k]);
                //map operator
                expected[k] = expectedFunction(vals0[k], vals1[k], vals2[k]);
            }
            // test
            auto input0 = vec_type::loadu(vals0);
            auto input1 = vec_type::loadu(vals1);
            auto input2 = vec_type::loadu(vals2);
            auto actual = actualFunction(input0, input1, input2);
            auto vec_expected = vec_type::loadu(expected);
            AssertVectorized<vec_type> vecAssert(testNameInfo, seed, vec_expected, actual, input0, input1, input2);
            if (vecAssert.check(bitwise, dmn.CheckWithTolerance, dmn.ToleranceError)) return;
        }// trial
        changeSeedBy += 1;
    }
}

template <typename T, typename Op>
T func_cmp(Op call, T v0, T v1) {
    using bit_rep = BitType<T>;
    constexpr bit_rep mask = std::numeric_limits<bit_rep>::max();
    bit_rep  ret = call(v0, v1) ? mask : 0;
    return c10::bit_cast<T>(ret);
}

struct PreventFma
{
    not_inline float sub(float a, float b)
    {
        return a - b;
    }
    not_inline double sub(double a, double b)
    {
        return a - b;
    }
    not_inline float add(float a, float b)
    {
        return a + b;
    }
    not_inline double add(double a, double b)
    {
        return a + b;
    }
};

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_log2(T x) {
    return std::log2(x);
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_log2(Complex<T> x) {
    T ret = std::log(x);
    T real = ret.real() / std::log(static_cast<T>(2));
    T imag = ret.imag() / std::log(static_cast<T>(2));
    return Complex<T>(real, imag);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_abs(T x) {
    return std::abs(x);
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_abs(Complex<T> x) {
#if defined(TEST_AGAINST_DEFAULT)
    return std::abs(x);
#else
    PreventFma noFma;
    T real = x.real();
    T imag = x.imag();
    T rr = real * real;
    T ii = imag * imag;
    T abs = std::sqrt(noFma.add(rr, ii));
    return Complex<T>(abs, 0);
#endif
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_multiply(T x, T y) {
    return x * y;
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_multiply(Complex<T> x, Complex<T> y) {
#if defined(TEST_AGAINST_DEFAULT)
    return x * y;
#else
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
    T x_real = x.real();
    T x_imag = x.imag();
    T y_real = y.real();
    T y_imag = y.imag();
#if defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR)
    //check multiplication considerin swap and fma
    T rr = x_real * y_real;
    T ii = x_imag * y_real;
    T neg_imag = -y_imag;
    rr = fma(x_imag, neg_imag, rr);
    ii = fma(x_real, y_imag, ii);
#else
    // replicate order
    PreventFma noFma;
    T ac = x_real * y_real;
    T bd = x_imag * y_imag;
    T ad = x_real * y_imag;
    T bc = x_imag * (-y_real);
    T rr = noFma.sub(ac, bd);
    T ii = noFma.sub(ad, bc);
#endif
    return Complex<T>(rr, ii);
#endif
}



template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_division(T x, T y) {
    return x / y;
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_division(Complex<T> x, Complex<T> y) {
#if defined(TEST_AGAINST_DEFAULT)
    return x / y;
#else /* defined(TEST_AGAINST_DEFAULT) */
    //re = (ac + bd)/abs_2()
    //im = (bc - ad)/abs_2()
    T x_real = x.real();
    T x_imag = x.imag();
    T y_real = y.real();
    T y_imag = y.imag();
    PreventFma noFma;
#if defined(CPU_CAPABILITY_ZVECTOR)
    T abs_c = std::abs(y_real);
    T abs_d = std::abs(y_imag);
    T scale = 1.0 / std::max(abs_c, abs_d);

    T a_sc = x_real * scale; // a/sc
    T b_sc = x_imag * scale; // b/sc
    T c_sc = y_real * scale; // c/sc
    T d_sc = y_imag * scale; // d/sc

    T ac_sc2 = a_sc * c_sc; // ac/sc^2
    T bd_sc2 = b_sc * d_sc; // bd/sc^2

    T neg_d_sc = -1.0 * d_sc; // -d/sc^2

    T neg_ad_sc2 = a_sc * neg_d_sc; // -ad/sc^2
    T bc_sc2 = b_sc * c_sc; // bc/sc^2

    T ac_bd_sc2 = noFma.add(ac_sc2, bd_sc2); // (ac+bd)/sc^2
    T bc_ad_sc2 = noFma.add(bc_sc2, neg_ad_sc2); // (bc-ad)/sc^2

    T c2_sc2 = c_sc * c_sc; // c^2/sc^2
    T d2_sc2 = d_sc * d_sc; // d^2/sc^2

    T c2_d2_sc2 = noFma.add(c2_sc2, d2_sc2); // (c^2+d^2)/sc^2

    T rr = ac_bd_sc2 / c2_d2_sc2; // (ac+bd)/(c^2+d^2)
    T ii = bc_ad_sc2 / c2_d2_sc2; // (bc-ad)/(c^2+d^2)

    return Complex<T>(rr, ii);
#else /* defined(CPU_CAPABILITY_ZVECTOR) */
#if defined(CPU_CAPABILITY_VSX)
    //check multiplication considerin swap and fma
    T rr = x_real * y_real;
    T ii = x_imag * y_real;
    T neg_imag = -y_imag;
    rr = fma(x_imag, y_imag, rr);
    ii = fma(x_real, neg_imag, ii);
    //b.abs_2
#else /* defined(CPU_CAPABILITY_VSX) */
    T ac = x_real * y_real;
    T bd = x_imag * y_imag;
    T ad = x_real * y_imag;
    T bc = x_imag * y_real;
    T rr = noFma.add(ac, bd);
    T ii = noFma.sub(bc, ad);
#endif /* defined(CPU_CAPABILITY_VSX) */
    //b.abs_2()
    T abs_rr = y_real * y_real;
    T abs_ii = y_imag * y_imag;
    T abs_2 = noFma.add(abs_rr, abs_ii);
    rr = rr / abs_2;
    ii = ii / abs_2;
    return Complex<T>(rr, ii);
#endif /* defined(CPU_CAPABILITY_ZVECTOR) */
#endif /* defined(TEST_AGAINST_DEFAULT) */
}


template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_fmadd(T a, T b, T c) {
    PreventFma noFma;
    T ab = a * b;
    return noFma.add(ab, c);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_sqrt(T x) {
    return std::sqrt(x);
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_sqrt(Complex<T> x) {
    return std::sqrt(x);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_asin(T x) {
    return std::asin(x);
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_asin(Complex<T> x) {
    return std::asin(x);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_acos(T x) {
    return std::acos(x);
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>> local_acos(Complex<T> x) {
    return std::acos(x);
}

template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_and(const T& val0, const T& val1) {
    using bit_rep = BitType<T>;
    bit_rep ret = c10::bit_cast<bit_rep>(val0) & c10::bit_cast<bit_rep>(val1);
    return c10::bit_cast<T> (ret);
}

template <typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
local_and(const Complex<T>& val0, const Complex<T>& val1)
{
    using bit_rep = BitType<T>;
    T real1 = val0.real();
    T imag1 = val0.imag();
    T real2 = val1.real();
    T imag2 = val1.imag();
    bit_rep real_ret = c10::bit_cast<bit_rep>(real1) & c10::bit_cast<bit_rep>(real2);
    bit_rep imag_ret = c10::bit_cast<bit_rep>(imag1) & c10::bit_cast<bit_rep>(imag2);
    return Complex<T>(c10::bit_cast<T>(real_ret), c10::bit_cast<T>(imag_ret));
}

template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_or(const T& val0, const T& val1) {
    using bit_rep = BitType<T>;
    bit_rep ret = c10::bit_cast<bit_rep>(val0) | c10::bit_cast<bit_rep>(val1);
    return c10::bit_cast<T> (ret);
}

template<typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
local_or(const Complex<T>& val0, const Complex<T>& val1) {
    using bit_rep = BitType<T>;
    T real1 = val0.real();
    T imag1 = val0.imag();
    T real2 = val1.real();
    T imag2 = val1.imag();
    bit_rep real_ret = c10::bit_cast<bit_rep>(real1) | c10::bit_cast<bit_rep>(real2);
    bit_rep imag_ret = c10::bit_cast<bit_rep>(imag1) | c10::bit_cast<bit_rep>(imag2);
    return Complex<T>(c10::bit_cast<T> (real_ret), c10::bit_cast<T>(imag_ret));
}

template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_xor(const T& val0, const T& val1) {
    using bit_rep = BitType<T>;
    bit_rep ret = c10::bit_cast<bit_rep>(val0) ^ c10::bit_cast<bit_rep>(val1);
    return c10::bit_cast<T> (ret);
}

template<typename T>
std::enable_if_t<is_complex<Complex<T>>::value, Complex<T>>
local_xor(const Complex<T>& val0, const Complex<T>& val1) {
    using bit_rep = BitType<T>;
    T real1 = val0.real();
    T imag1 = val0.imag();
    T real2 = val1.real();
    T imag2 = val1.imag();
    bit_rep real_ret = c10::bit_cast<bit_rep>(real1) ^ c10::bit_cast<bit_rep>(real2);
    bit_rep imag_ret = c10::bit_cast<bit_rep>(imag1) ^ c10::bit_cast<bit_rep>(imag2);
    return Complex<T>(c10::bit_cast<T> (real_ret), c10::bit_cast<T>(imag_ret));
}

template <typename T>
T quantize_val(float scale, int64_t zero_point, float value) {
    int64_t qvalue;
    constexpr int64_t qmin = std::numeric_limits<T>::min();
    constexpr int64_t qmax = std::numeric_limits<T>::max();
    float inv_scale = 1.0f / scale;
    qvalue = static_cast<int64_t>(zero_point + at::native::round_impl<float>(value * inv_scale));
    qvalue = std::max<int64_t>(qvalue, qmin);
    qvalue = std::min<int64_t>(qvalue, qmax);
    return static_cast<T>(qvalue);
}

template <typename T>
#if defined(TEST_AGAINST_DEFAULT)
T requantize_from_int(float multiplier, int32_t zero_point, int32_t src) {
    auto xx = static_cast<float>(src) * multiplier;
    double xx2 = nearbyint(xx);
    int32_t quantize_down = xx2 + zero_point;
#else
T requantize_from_int(float multiplier, int64_t zero_point, int64_t src) {
    int64_t quantize_down = static_cast<int64_t>(zero_point + std::lrintf(src * multiplier));
#endif
    constexpr int64_t min = std::numeric_limits<T>::min();
    constexpr int64_t max = std::numeric_limits<T>::max();
    auto ret = static_cast<T>(std::min<int64_t>(std::max<int64_t>(quantize_down, min), max));
    return ret;
}

template <typename T>
float dequantize_val(float scale, int64_t zero_point, T value) {
    //when negated scale is used as addition
#if defined(CHECK_WITH_FMA)
    float neg_p = -(zero_point * scale);
    float v = static_cast<float>(value);
    float ret = fma(v, scale, neg_p);
#else
    float ret = (static_cast<float>(value) - zero_point) * scale;
#endif
    return ret;
}

template<typename T>
T relu(const T & val, const T & zero_point) {
    return std::max(val, zero_point);
}

template<typename T>
T relu6(T val, T zero_point, T q_six) {
    return std::min<T>(std::max<T>(val, zero_point), q_six);
}

template<typename T>
int32_t widening_subtract(T val, T b) {
    return static_cast<int32_t>(val) - static_cast<int32_t>(b);
}

//default testing case
template<typename T>
T getDefaultTolerance() {
    return static_cast<T>(0.0);
}

template<>
float getDefaultTolerance() {
    return 5.e-5f;
}

template<>
double getDefaultTolerance() {
    return 1.e-9;
}

template<typename T>
TestingCase<T> createDefaultUnaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, size_t trials = 0) {
    using UVT = UvalueType<T>;
    TestingCase<T> testCase;
    if (!bitwise && std::is_floating_point<UVT>::value) {
        //for float types lets add manual ranges
        UVT tolerance = getDefaultTolerance<UVT>();
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
            .setTrialCount(trials)
            .setTestSeed(seed);
    }
    else {
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UVT>{})
            .setTrialCount(trials)
            .setTestSeed(seed);
    }
    return testCase;
}

template<typename T>
TestingCase<T> createDefaultBinaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, size_t trials = 0) {
    using UVT = UvalueType<T>;
    TestingCase<T> testCase;
    if (!bitwise && std::is_floating_point<UVT>::value) {
        //for float types lets add manual ranges
        UVT tolerance = getDefaultTolerance<UVT>();
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}, { (UVT)-10, (UVT)10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }, { (UVT)-10, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }, { (UVT)-100, (UVT)1000 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }, { (UVT)-100, (UVT)10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }, { (UVT)-1000, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
            .setTrialCount(trials)
            .setTestSeed(seed);
    }
    else {
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, false)
            .addDomain(CheckWithinDomains<UVT>{})
            .setTrialCount(trials)
            .setTestSeed(seed);
    }
    return testCase;
}

template<typename T>
TestingCase<T> createDefaultTernaryTestCase(TestSeed seed = TestSeed(), bool bitwise = false, bool checkWithTolerance = false, size_t trials = 0) {
    TestingCase<T> testCase = TestingCase<T>::getBuilder()
        .set(bitwise, false)
        .addDomain(CheckWithinDomains<UvalueType<T>>{})
        .setTrialCount(trials)
        .setTestSeed(seed);
    return testCase;
}
