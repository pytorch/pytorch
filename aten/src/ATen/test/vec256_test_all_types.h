#pragma once
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/Math.h>
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

#define CACHE_LINE 32
#if defined(__GNUC__)
#define CACHE_ALIGN __attribute__((aligned(CACHE_LINE)))
#elif defined(_WIN32)
#define CACHE_ALIGN __declspec(align(CACHE_LINE))
#else
CACHE_ALIGN #define
#endif

#define RESOLVE_OVERLOAD(...)                                  \
  [](auto&&... args) -> decltype(auto) {                       \
    return __VA_ARGS__(std::forward<decltype(args)>(args)...); \
  }

#if defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_AVX2) && (defined(__GNUC__) || defined(__GNUG__))
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
using Vec = typename at::vec256::Vec256<T>;

using vfloat = Vec<float>;
using vdouble = Vec<double>;
using vcomplex = Vec<Complex<float>>;
using vcomplexDbl = Vec<Complex<double>>;
using vlong = Vec<int64_t>;
using vint = Vec<int32_t>;
using vshort = Vec<int16_t>;
using vqint8 = Vec<c10::qint8>;
using vquint8 = Vec<c10::quint8>;
using vqint = Vec<c10::qint32>;

template <typename T>
using ValueType = typename T::value_type;

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

template <int N>
struct BitStr {
    using type = uintmax_t;
};

template <>
struct BitStr<8> {
    using type = uint64_t;
};

template <>
struct BitStr<4> {
    using type = uint32_t;
};

template <>
struct BitStr<2> {
    using type = uint16_t;
};

template <>
struct BitStr<1> {
    using type = uint8_t;
};

template <typename T>
struct DomainRange {
    T start;  // start [
    T end;    // end is not included. one could use  nextafter for including his end case for tests
};

template <typename T>
struct CustomCheck {
    std::vector<T> Args;
    T expectedResult;
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
            if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
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

template <class To, class From>
typename std::enable_if<
    (sizeof(To) == sizeof(From)) && std::is_trivially_copyable<From>::value &&
    std::is_trivial<To>::value,
    // this implementation requires that To is trivially default constructible
    To>::type
    bit_cast(const From& src) noexcept {
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <class To, class T>
To bit_cast_ptr(T* p, size_t N = sizeof(To)) noexcept {
    unsigned char p1[sizeof(To)] = {};
    std::memcpy(p1, p, std::min(N, sizeof(To)));
    return bit_cast<To>(p1);
}

template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> check_both_nan(T x,
    T y) {
    return std::isnan(x) && std::isnan(y);  //(std::fpclassify(x) == FP_NAN &&
                                      //std::fpclassify(y) == FP_NAN);
}

template <typename T>
std::enable_if_t<!std::is_floating_point<T>::value, bool> check_both_nan(T x,
    T y) {
    return false;
}

template<class T> struct is_complex : std::false_type {};

template<class T> struct is_complex<Complex<T>> : std::true_type {};

template<typename T>
T safe_fpt_division(T f1, T f2)
{
    //code was taken from boost
    // Avoid overflow.
    if( (f2 < static_cast<T>(1)) && (f1 > f2*std::numeric_limits<T>::max()) ){
        return std::numeric_limits<T>::max();
    }
    // Avoid underflow.
    if( (f1 == static_cast<T>(0)) ||
        ((f2 > static_cast<T>(1)) && (f1 < f2*std::numeric_limits<T>::min())) ){
        return static_cast<T>(0);
    }
    return f1 / f2;
}

template<class T>
bool nearlyEqual(T a, T b, T tolerance) {
    if (std::isinf(a) && std::isinf(b)) return true;
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

template <typename T>
T reciprocal(T x) {
    return 1 / x;
}

template <typename T>
T rsqrt(T x) {
    return 1 / std::sqrt(x);
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
void filter_zero(T& val) {
    val = is_zero(val) ? (T)1 : val;
}

template <typename T>
void filter_int_minimum(T& val) {
    if (!std::is_integral<T>::value) return;
    if (val == std::numeric_limits<T>::min()) {
        val = 0;
    }
}

template <typename T>
std::enable_if_t<is_complex<T>::value, void> filter_op(T& a, T& b, bool minus)
{
    //missing for complex
}

template <typename T>
std::enable_if_t < !is_complex<T>::value, void > filter_op(T& a, T& b, bool minus) {
    T max = std::numeric_limits<T>::max();
    T min = std::numeric_limits<T>::min();
    if (minus) {
        if (b == min) b = min + 1;
        b = -b;
    }
    bool sgn1 = a > 0;
    bool sgn2 = b > 0;
    if (sgn1 == sgn2) {
        if (sgn1 && a > max - b) {
            a = max - b;
        }
        else if (!sgn1 && a < min - b) {
            a = min - b;
        }
    }
}

template <typename T>
void filter_add_overflow(T& val1, T& val2) {
    if (std::is_integral<T>::value == false) return;
    return filter_op(val1, val2, false);
}

template <typename T>
void filter_minus_overflow(T& val1, T& val2) {
    if (std::is_integral<T>::value == false) return;
    return filter_op(val1, val2, true);
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
    if (std::is_integral<T>::value == false) return;
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
    if (std::is_integral<T>::value == false) return;
    if (is_zero(val2)) {
        val2 = 1;
    }
    else if (val1 == std::numeric_limits<T>::min() && val2 == -1) {
        val2 = 1;
    }
}

template<typename T>
struct CmpHelper {
    using cmpType = T;
    static constexpr int size() { return 1; }
    static void bitCheck(const T& act, const T& exp, int i, const std::function<std::string(int index)>& get_details) {
        using bit_rep = typename BitStr<sizeof(T)>::type;
        bit_rep b_exp = bit_cast<bit_rep>(exp);
        bit_rep b_act = bit_cast<bit_rep>(act);
        ASSERT_EQ(b_exp, b_act) << (get_details ? get_details(i) : "");
    }
    static void nearCheck(const T& act, const T& exp, const T& absErr, int i, const std::function<std::string(int index)>& get_details) {
    }
    static void eqCheck(const T& act, const T& exp, int i, const std::function<std::string(int index)>& get_details) {
        ASSERT_EQ(act, exp) << (get_details ? get_details(i) : "");
    }
};

template<>
struct CmpHelper<double> {
    using cmpType = double;
    static constexpr int size() { return 1; }
    static void bitCheck(const double& act, const double& exp, int i, const std::function<std::string(int index)>& get_details) {
        using bit_rep = typename BitStr<sizeof(double)>::type;
        bit_rep b_exp = bit_cast<bit_rep>(exp);
        bit_rep b_act = bit_cast<bit_rep>(act);
        ASSERT_EQ(b_exp, b_act) << (get_details ? get_details(i) : "");
    }
    static void nearCheck(const double& act, const double& exp, const double& absErr, int i, const std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp, act))) {
            ASSERT_EQ(nearlyEqual(exp, act, absErr), true) << exp << " " << act << "\n" << (get_details ? get_details(i) : "");
        }
    }
    static void eqCheck(const double& act, const double& exp, int i, const std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp, act))) {
            ASSERT_DOUBLE_EQ(exp, act) << (get_details ? get_details(i) : "");
        }
    }
};

template<>
struct CmpHelper<float> {
    using cmpType = float;
    static constexpr int size() { return 1; }
    static void bitCheck(const float& act, const float& exp, int i, const std::function<std::string(int index)>& get_details) {
        using bit_rep = typename BitStr<sizeof(float)>::type;
        bit_rep b_exp = bit_cast<bit_rep>(exp);
        bit_rep b_act = bit_cast<bit_rep>(act);
        ASSERT_EQ(b_exp, b_act) << (get_details ? get_details(i) : "");
    }
    static void nearCheck(const float& act, const float& exp, const float& absErr, int i, const std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp, act))) {
            ASSERT_EQ(nearlyEqual(exp, act, absErr), true) << exp << " " << act << "\n" << (get_details ? get_details(i) : "");
        }
    }
    static void eqCheck(const float& act, const float& exp, int i, const std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp, act))) {
            ASSERT_FLOAT_EQ(exp, act) << (get_details ? get_details(i) : "");
        }
    }
};

template<>
struct CmpHelper<Complex<double>> {
    using cmpType = double;
    static constexpr int size() { return 2; }
    static void bitCheck(const Complex<double>& act, const Complex<double>& exp, int i, std::function<std::string(int index)>& get_details) {
        using bit_rep = typename BitStr<sizeof(double)>::type;
        bit_rep b_expReal = bit_cast<bit_rep>(exp.real());
        bit_rep b_actReal = bit_cast<bit_rep>(act.real());
        ASSERT_EQ(b_expReal, b_actReal) << (get_details ? get_details(i) : "");
        bit_rep b_expI = bit_cast<bit_rep>(exp.imag());
        bit_rep b_actI = bit_cast<bit_rep>(act.imag());
        ASSERT_EQ(b_expI, b_actI) << (get_details ? get_details(i) : "");
    }
    static void nearCheck(const Complex<double>& act, const Complex<double>& exp, const Complex<double>& absErr, int i, std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp.real(), act.real()))) {
            ASSERT_EQ(nearlyEqual(exp.real(), act.real(), absErr.real()), true) << exp.real() << " " << act.real() << "\n" << (get_details ? get_details(i) : "");
        }
        if (!(check_both_nan(exp.imag(), act.imag()))) {
            ASSERT_EQ(nearlyEqual(exp.imag(), act.imag(), absErr.real()), true) << exp.imag() << " " << act.imag() << "\n" << (get_details ? get_details(i) : "");
        }
    }
    static void eqCheck(const Complex<double>& act, const Complex<double>& exp, int i, std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp.real(), act.real()))) {
            ASSERT_DOUBLE_EQ(exp.real(), act.real()) << (get_details ? get_details(i) : "");
        }
        if (!(check_both_nan(exp.imag(), act.imag()))) {
            ASSERT_DOUBLE_EQ(exp.imag(), act.imag()) << (get_details ? get_details(i) : "");
        }
    }
};

template<>
struct CmpHelper<Complex<float>> {
    using cmpType = float;
    static constexpr int size() { return 2; }
    static void bitCheck(const Complex<float>& act, const Complex<float>& exp, int i, std::function<std::string(int index)>& get_details) {
        using bit_rep = typename BitStr<sizeof(float)>::type;
        bit_rep b_expReal = bit_cast<bit_rep>(exp.real());
        bit_rep b_actReal = bit_cast<bit_rep>(act.real());
        ASSERT_EQ(b_expReal, b_actReal) << (get_details ? get_details(i) : "");
        bit_rep b_expI = bit_cast<bit_rep>(exp.imag());
        bit_rep b_actI = bit_cast<bit_rep>(act.imag());
        ASSERT_EQ(b_expI, b_actI) << (get_details ? get_details(i) : "");
    }
    static void nearCheck(const Complex<float>& act, const Complex<float>& exp, const Complex<float>& absErr, int i, std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp.real(), act.real()))) {
            ASSERT_EQ(nearlyEqual(exp.real(), act.real(), absErr.real()), true) << exp.real() << " " << act.real() << "\n" << (get_details ? get_details(i) : "");;
        }
        if (!(check_both_nan(exp.imag(), act.imag()))) {
            ASSERT_EQ(nearlyEqual(exp.imag(), act.imag(), absErr.real()), true) << exp.imag() << " " << act.imag() << "\n" << (get_details ? get_details(i) : "");;
        }
    }
    static void eqCheck(const Complex<float>& act, const Complex<float>& exp, int i, std::function<std::string(int index)>& get_details) {
        if (!(check_both_nan(exp.real(), act.real()))) {
            ASSERT_FLOAT_EQ(exp.real(), act.real()) << (get_details ? get_details(i) : "");
        }
        if (!(check_both_nan(exp.imag(), act.imag()))) {
            ASSERT_FLOAT_EQ(exp.imag(), act.imag()) << (get_details ? get_details(i) : "");
        }
    }
};
//to extract underline type from complex<float>

template <typename T>
using UvalueType = typename CmpHelper<ValueType<T>>::cmpType;

template <typename T>
using UnitType = typename CmpHelper<T>::cmpType;

template <typename T>
using BitType = typename BitStr<sizeof(T)>::type;

template <typename T>
using BitValueType = typename BitStr<sizeof(ValueType<T>)>::type;

template <typename T>
using BitUvalueType = typename BitStr<sizeof(UvalueType<T>)>::type;

template <typename T>
void AssertVec256(T expected, T actual,
    std::function<std::string(int index)> get_details = {},
    bool bitwise = false, bool check_absError = false,
    ValueType<T> absError = {}) {
    using VT = ValueType<T>;
    constexpr auto sizeX = T::size();
    CACHE_ALIGN VT exp[sizeX];
    CACHE_ALIGN VT act[sizeX];
    expected.store(exp);
    actual.store(act);
    if (bitwise) {
        for (int i = 0; i < sizeX; i++) {
            CmpHelper<VT>::bitCheck(act[i], exp[i], i, get_details);
            if (::testing::Test::HasFailure()) {
                break;
            }
        }
    }
    else if (check_absError) {
        for (int i = 0; i < sizeX; i++) {
            CmpHelper<VT>::nearCheck(act[i], exp[i], absError, i, get_details);
            if (::testing::Test::HasFailure()) {
                break;
            }
        }
    }
    else {
        for (int i = 0; i < sizeX; i++) {
            CmpHelper<VT>::eqCheck(act[i], exp[i], i, get_details);
            if (::testing::Test::HasFailure()) {
                break;
            }
        }
    }
}
struct TestSeed {
    TestSeed() : testSeed( std::chrono::high_resolution_clock::now().time_since_epoch().count()){
    }
    TestSeed(uint64_t seed) : testSeed(seed) {
    }
    uint64_t getSeed() {
        return testSeed;
    }
    operator uint64_t () {
        return testSeed;
    }
    uint64_t nextSeed() {
        auto ret = testSeed + index;
        ++index;
        return ret;
    }
private:
    uint64_t testSeed;
    int index = 0;
};

template <typename T, typename U = typename CmpHelper<T>::cmpType, bool is_floating_point = std::is_floating_point<U>::value>
struct ValueGen {
    std::uniform_int_distribution<int64_t> dis;
    std::mt19937 gen;
    ValueGen() :ValueGen(std::numeric_limits<U>::min(), std::numeric_limits<U>::max()) {
    }
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<U>::min(), std::numeric_limits<U>::max(), seed) {
    } 
    ValueGen(U start, U stop, uint64_t seed = TestSeed()) {
        gen = std::mt19937(seed);
        dis = std::uniform_int_distribution<int64_t>(start, stop);
    }
    T get() {
        return (T)dis(gen);
    }
};

template<typename T, typename U>
struct ValueGen<T, U, true> {
    std::mt19937 gen;
    std::normal_distribution<U> normal;
    std::uniform_int_distribution<int> roundChance;
    U _start;
    U _stop;
    bool use_sign_change = false;
    bool use_round = true;
    ValueGen(): ValueGen(std::numeric_limits<U>::min(), std::numeric_limits<U>::max()) {
    }
    ValueGen(uint64_t seed) : ValueGen(std::numeric_limits<U>::min(), std::numeric_limits<U>::max(), seed) {
    }
    ValueGen(U start, U stop, uint64_t seed = TestSeed()) {
        gen = std::mt19937(seed);
        U mean = start * (U)0.5 + stop * (U)0.5;
        //make it  normal +-3sigma  
        U divRange = (U)(6.0);
        U stdev = std::abs(stop / divRange - start / divRange);
        normal = std::normal_distribution<U>{ mean,stdev };
        // in real its hard to get rounded value
        // so we will force it by  uniform chance
        roundChance = std::uniform_int_distribution<int>(0, 5);
        _start = start;
        _stop = stop;
    }
    
template<typename ST = T, typename SU = U>
    std::enable_if_t<std::is_same<ST, SU>::value, T>
        get() {
        T a = normal(gen);
        //make rounded value ,too
        auto rChoice = roundChance(gen);
        if (rChoice == 1) a = std::round(a);
        if (a < _start) return nextafter(_start, _stop);
        if (a >= _stop) return nextafter(_stop, _start);
        return a;
    }
    //complex
    
template<typename ST = T, typename SU = U>
    std::enable_if_t<!std::is_same<ST, SU>::value, T>
        get() {
        U a = normal(gen);
        U b = normal(gen);
        //make rounded value ,too
        auto rChoice = roundChance(gen);
        rChoice = rChoice & 3;
        if (rChoice & 1) a = std::round(a);
        if (rChoice & 2) b = std::round(b);
        if (a < _start) a = nextafter(_start, _stop);
        else if (a >= _stop) a = nextafter(_stop, _start);
        if (b < _start) b = nextafter(_start, _stop);
        else if (b >= _stop) b = nextafter(_stop, _start);
        return T(a, b);
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
        std::cout << "Total Trial Count:" << trialCount * domains_size << std::endl;
    }
    else {
        std::cout << "Total Trial Count:" << trialCount << std::endl;
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
    TestCaseBuilder<T, U>& setRandomSeed(uint64_t seed) {
        _case.testSeed = TestSeed(seed);
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

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_unary(
    std::string testName,
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
    std::cout << "Test Seed: " << seed << std::endl;
    for (const CheckWithinDomains<UVT>& dmn : domains) {
        size_t dmn_argc = dmn.ArgsDomain.size();
        UVT start = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        std::cout << dmn << std::endl;
        ValueGen<VT> generator(start, end, seed.nextSeed());
        for (int trial = 0; trial < trialCount; trial++) {
            for (int k = 0; k < el_count; k++) {
                vals[k] = generator.get();
                call_filter(filter, vals[k]);
                //map operator
                expected[k] = expectedFunction(vals[k]);
            }
            // test
            auto input = vec_type::loadu(vals);
            auto actual = actualFunction(input);
            auto vec_expected = vec_type::loadu(expected);
            std::function<std::string(int i)> detail = [testName, input, actual, vec_expected](int i) {
                std::stringstream stream;
                stream << testName << ": {\n" << input << "\nvec_exp:" << vec_expected << "\nvec_act:" << actual << "\n}";
                return stream.str();
            };
            AssertVec256(vec_expected, actual, detail, bitwise, dmn.CheckWithTolerance, dmn.ToleranceError);
            if (::testing::Test::HasFailure()) {
                return;
            }
        }// trial 
    }
    for (auto& custom : testCase.getCustomChecks()) {
        auto args = custom.Args;
        if (args.size() > 0) {
            auto input = vec_type{ args[0] };
            auto actual = actualFunction(input);
            auto vec_expected = vec_type{ custom.expectedResult };
            std::function<std::string(int i)> detail = [testName, input, actual, vec_expected](int i) {
                std::stringstream stream;
                stream << testName << ": {\n" << input << "\nvec_exp:" << vec_expected << "\nvec_act:" << actual << "\n}";
                return stream.str();
            };
            AssertVec256(vec_expected, actual, detail);
        }
    }
}

template<typename T>
T getDefaultTolerance() {
    return static_cast<T>(0.0);
}

template<>
float getDefaultTolerance() {
    return 1.e-5f;
}

template<>
double getDefaultTolerance() {
    return 1.e-9;
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void
test_unary(
    std::string testName,
    Op1 expectedFunction,
    Op2 actualFunction, bool bitwise = false,
    Filter filter = {}, bool checkWithTolerance = false, bool testSpecialValues = false, size_t trials = 0) {
    using UVT = UvalueType<T>;
    TestingCase<T> testCase;
    if (!bitwise && std::is_floating_point<UVT>::value) {
        //for float types lets add manual ranges  
        UVT tolerance = getDefaultTolerance<UVT>();
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, testSpecialValues)
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
            .setTrialCount(trials);
    }
    else {
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, testSpecialValues)
            .addDomain(CheckWithinDomains<UVT>{})
            .setTrialCount(trials);
    }
    test_unary<T, Op1, Op2, Filter>(testName, expectedFunction, actualFunction, testCase, filter);
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_binary(
    std::string testName,
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
    std::cout << "Test Seed: " << seed << std::endl;
    for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
        size_t dmn_argc = dmn.ArgsDomain.size();
        UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
        UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
        std::cout << dmn << std::endl;
        ValueGen<VT> generator0(start0, end0, seed.nextSeed());
        ValueGen<VT> generator1(start1, end1, seed.nextSeed());
        for (int trial = 0; trial < trialCount; trial++) {
            for (int k = 0; k < el_count; k++) {
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
            std::function<std::string(int i)> detail = [testName, input0, input1, actual, vec_expected](int i) {
                std::stringstream stream;
                stream << testName << ": {\n" << input0 << "," << input1 << "\nvec_exp:" << vec_expected << "\nvec_act:" << actual << "\n}";
                return stream.str();
            };
            AssertVec256(vec_expected, actual, detail, bitwise, dmn.CheckWithTolerance, dmn.ToleranceError);
            if (::testing::Test::HasFailure()) {
                return;
            }
        }// trial 
    }
    for (auto& custom : testCase.getCustomChecks()) {
        auto args = custom.Args;
        if (args.size() > 0) {
            auto input0 = vec_type{ args[0] };
            auto input1 = args.size() > 1 ? vec_type{ args[1] } : vec_type{ args[0] };
            auto actual = actualFunction(input0, input1);
            auto vec_expected = vec_type(custom.expectedResult);
            std::function<std::string(int i)> detail = [testName, input0, input1, actual, vec_expected](int i) {
                std::stringstream stream;
                stream << testName << ": {\n" << input0 << "," << input1 << "\nvec_exp:" << vec_expected << "\nvec_act:" << actual << "\n}";
                return stream.str();
            };
            AssertVec256(vec_expected, actual, detail);
        }
    }
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void
test_binary(
    std::string testName,
    Op1 expectedFunction,
    Op2 actualFunction, bool bitwise = false,
    Filter filter = {}, bool checkWithTolerance = false, bool testSpecialValues = false, size_t trials = 0) {
    using UVT = UvalueType<T>;
    TestingCase<T> testCase;
    if (!bitwise && std::is_floating_point<UVT>::value) {
        //for float types lets add manual ranges  
        UVT tolerance = getDefaultTolerance<UVT>();
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, testSpecialValues)
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-10, (UVT)10}, { (UVT)-10, (UVT)10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)10, (UVT)100 }, { (UVT)-10, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)100, (UVT)1000 }, { (UVT)-100, (UVT)1000 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)-10 }, { (UVT)-100, (UVT)10 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-1000, (UVT)-100 }, { (UVT)-1000, (UVT)100 }}, checkWithTolerance, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ {}, checkWithTolerance, tolerance})
            .setTrialCount(trials);
    }
    else {
        testCase = TestingCase<T>::getBuilder()
            .set(bitwise, testSpecialValues)
            .addDomain(CheckWithinDomains<UVT>{})
            .setTrialCount(trials);
    }
    test_binary<T, Op1, Op2, Filter>(testName, expectedFunction, actualFunction, testCase, filter);
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void test_ternary(
    std::string testName,
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
    std::cout << "Test Seed: " << seed << std::endl;
    for (const CheckWithinDomains<UVT>& dmn : testCase.getDomains()) {
        size_t dmn_argc = dmn.ArgsDomain.size();
        UVT start0 = dmn_argc > 0 ? dmn.ArgsDomain[0].start : default_start;
        UVT end0 = dmn_argc > 0 ? dmn.ArgsDomain[0].end : default_end;
        UVT start1 = dmn_argc > 1 ? dmn.ArgsDomain[1].start : default_start;
        UVT end1 = dmn_argc > 1 ? dmn.ArgsDomain[1].end : default_end;
        UVT start2 = dmn_argc > 2 ? dmn.ArgsDomain[2].start : default_start;
        UVT end2 = dmn_argc > 2 ? dmn.ArgsDomain[2].end : default_end;
        ValueGen<VT> generator0(start0, end0, seed.nextSeed());
        ValueGen<VT> generator1(start1, end1, seed.nextSeed());
        ValueGen<VT> generator2(start2, end2, seed.nextSeed());
        std::cout << dmn << std::endl;
        for (int trial = 0; trial < trialCount; trial++) {
            for (int k = 0; k < el_count; k++) {
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
            std::function<std::string(int i)> detail = [testName, input0, input1, input2, actual, vec_expected](int i) {
                std::stringstream stream;
                stream << testName << ": {\n" << input0 << "," << input1 << "," << input2 << "\nvec_exp:" << vec_expected << "\nvec_act:" << actual << "\n}";
                return stream.str();
            };
            AssertVec256(vec_expected, actual, detail, bitwise, dmn.CheckWithTolerance, dmn.ToleranceError);
            if (::testing::Test::HasFailure()) {
                return;
            }
        }// trial 
    }
}

template< typename T, typename Op1, typename Op2, typename Filter = std::nullptr_t>
void
test_ternary(
    std::string testName,
    Op1 expectedFunction,
    Op2 actualFunction, bool bitwise = false,
    Filter filter = {}, bool testSpecialValues = false, size_t trials = 0) {
    TestingCase<T> testCase = TestingCase<T>::getBuilder()
        .set(bitwise, testSpecialValues)
        .addDomain(CheckWithinDomains<UvalueType<T>>{})
        .setTrialCount(trials);
    test_ternary<T, Op1, Op2, Filter>(testName, expectedFunction, actualFunction, testCase, filter);
}

template <typename T, typename Op>
T func_cmp(Op call, T v0, T v1) {
    using bit_rep = BitType<T>;
    constexpr bit_rep mask = std::numeric_limits<bit_rep>::max();
    bit_rep  ret = call(v0, v1) ? mask : 0;
    return bit_cast<T>(ret);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_log2(T x) {
    return std::log2(x);
}

template <typename T>
std::enable_if_t<is_complex<T>::value, T> local_log2(T x) {
    T ret = std::log(x);
    UnitType<T> real = ret.real() / std::log((UnitType<T>)2);
    UnitType<T> imag = ret.imag() / std::log((UnitType<T>)2);
    return T(real, imag);
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_abs(T x) {
    return std::abs(x);
}

template <typename T>
std::enable_if_t<is_complex<T>::value, T> local_abs(T x) {
#if defined(CPU_CAPABILITY_DEFAULT)
    return std::abs(x);
#else
    UnitType<T> real = x.real();
    UnitType<T> imag = x.imag();
    UnitType<T> rr = real * real;
    UnitType<T> ii = imag * imag;
    return T{ std::sqrt(rr + ii), 0 };
#endif
}

template <typename T>
std::enable_if_t<!is_complex<T>::value, T> local_multiply(T x, T y) {
    return x * y;
}

template <typename T>
std::enable_if_t<is_complex<T>::value, T> local_multiply(T x, T y) {
#if defined(CPU_CAPABILITY_DEFAULT)
    return x * y;
#else
    //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
    UnitType<T> x_real = x.real();
    UnitType<T> x_imag = x.imag();
    UnitType<T> y_real = y.real();
    UnitType<T> y_imag = y.imag();
#if defined(CPU_CAPABILITY_VSX)
    //check multiplication considerin swap and fma
    UnitType<T> rr = x_real * y_real;
    UnitType<T> ii = x_imag * y_real;
    UnitType<T> neg_imag = -y_imag;
    rr = fma(x_imag, neg_imag, rr);
    ii = fma(x_real, y_imag, ii);
#else
    UnitType<T> rr = x_real * y_real - x_imag * y_imag;
    UnitType<T> ii = x_real * y_imag + x_imag * y_real;
#endif
    return T{ rr, ii };
#endif
}

template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_and(const T& val0, const T& val1) {
    using bit_rep = BitType<T>;
    bit_rep ret = bit_cast<bit_rep>(val0) & bit_cast<bit_rep>(val1);
    return bit_cast<T> (ret);
}

template<typename T>
std::enable_if_t<is_complex<T>::value, T>
local_and(const T& val0, const T& val1) {
    using UVT = UnitType<T>;
    using bit_rep = BitUvalueType<T>;
    UVT real1 = val0.real();
    UVT imag1 = val0.imag();
    UVT real2 = val1.real();
    UVT imag2 = val1.imag();
    bit_rep real_ret = bit_cast<bit_rep>(real1) & bit_cast<bit_rep>(real2);
    bit_rep imag_ret = bit_cast<bit_rep>(imag1) & bit_cast<bit_rep>(imag2);
    return T(bit_cast<UVT> (real_ret), bit_cast<UVT>(imag_ret));
}

template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_or(const T& val0, const T& val1) {
    using bit_rep = BitType<T>;
    bit_rep ret = bit_cast<bit_rep>(val0) | bit_cast<bit_rep>(val1);
    return bit_cast<T> (ret);
}

template<typename T>
std::enable_if_t<is_complex<T>::value, T>
local_or(const T& val0, const T& val1) {
    using UVT = UnitType<T>;
    using bit_rep = BitUvalueType<T>;
    UVT real1 = val0.real();
    UVT imag1 = val0.imag();
    UVT real2 = val1.real();
    UVT imag2 = val1.imag();
    bit_rep real_ret = bit_cast<bit_rep>(real1) | bit_cast<bit_rep>(real2);
    bit_rep imag_ret = bit_cast<bit_rep>(imag1) | bit_cast<bit_rep>(imag2);
    return T(bit_cast<UVT> (real_ret), bit_cast<UVT> (imag_ret));
}

template<typename T>
std::enable_if_t<!is_complex<T>::value, T>
local_xor(const T& val0, const T& val1) {
    using bit_rep = BitType<T>;
    bit_rep ret = bit_cast<bit_rep>(val0) ^ bit_cast<bit_rep>(val1);
    return bit_cast<T> (ret);
}

template<typename T>
std::enable_if_t<is_complex<T>::value, T>
local_xor(const T& val0, const T& val1) {
    using UVT = UnitType<T>;
    using bit_rep = BitUvalueType<T>;
    UVT real1 = val0.real();
    UVT imag1 = val0.imag();
    UVT real2 = val1.real();
    UVT imag2 = val1.imag();
    bit_rep real_ret = bit_cast<bit_rep>(real1) ^ bit_cast<bit_rep>(real2);
    bit_rep imag_ret = bit_cast<bit_rep>(imag1) ^ bit_cast<bit_rep>(imag2);
    return T(bit_cast<UVT> (real_ret), bit_cast<UVT> (imag_ret));
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
T requantize_from_int(float multiplier, int64_t zero_point, int64_t src) {
    int64_t quantize_down = static_cast<int64_t>(zero_point + std::lrintf(src * static_cast<float>(multiplier)));
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
T relu(const T& val, const T& zero_point) {
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

