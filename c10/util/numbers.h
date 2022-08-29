#pragma once

namespace c10 {
namespace numbers {
template <typename T1>
constexpr T1 catalan_v = T1(0.915965594177219015054603514932384110773L);

template <typename T1>
constexpr T1 cbrt3_v = T1(1.442249570307408382321638310780109588390L);

template <typename T1>
constexpr T1 cbrtpi_v = T1(1.464591887561523263020142527263790391736L);

template <typename T1>
constexpr T1 deg_v = T1(180) /
T1(3.141592653589793238462643383279502884195e+0L);

template <typename T1>
constexpr T1 e_v = T1(2.718281828459045235360287471352662497759e+0L);

template <typename T1>
constexpr T1 egamma_v = T1(5.772156649015328606065120900824024310432e-1L);

template <typename T1>
constexpr T1 inv_e_v = T1(0.367879441171442321595523770161460867446L);

template <typename T1>
constexpr T1 inv_pi_v = T1(3.183098861837906715377675267450287240691e-1L);

template <typename T1>
constexpr T1 inv_sqrt2_v = T1(0.707106781186547524400844362104849039285L);

template <typename T1>
constexpr T1 inv_sqrt3_v = T1(5.773502691896257645091487805019574556475e-1L);

template <typename T1>
constexpr T1 inv_sqrtpi_v = T1(5.641895835477562869480794515607725858438e-1L);

template <typename T1>
constexpr T1 inv_tau_v = T1(0.159154943091895335768883763372514362035L);

template <typename T1>
constexpr T1 ln10_v = T1(2.302585092994045684017991454684364207602e+0L);

template <typename T1>
constexpr T1 ln2_v = T1(6.931471805599453094172321214581765680748e-1L);

template <typename T1>
constexpr T1 ln3_v = T1(1.098612288668109691395245236922525704648L);

template <typename T1>
constexpr T1 lnpi_v = T1(1.144729885849400174143427351353058711646L);

template <typename T1>
constexpr T1 lnsqrttau_v = T1(0.918938533204672741780329736405617639862L);

template <typename T1>
constexpr T1 log10e_v = T1(4.342944819032518276511289189166050822940e-1L);

template <typename T1>
constexpr T1 log2e_v = T1(1.442695040888963407359924681001892137427e+0L);

template <typename T1>
constexpr T1 phi_v = T1(1.618033988749894848204586834365638117720e+0L);

template <typename T1>
constexpr T1 pi_4_div_3_v = T1(4.188790204786390984616857844372670512253L);

template <typename T1>
constexpr T1 pi_4_v = T1(12.56637061435917295385057353311801153678L);

template <typename T1>
constexpr T1 pi_sqr_div_6_v = T1(1.644934066848226436472415166646025189221L);

template <typename T1>
constexpr T1 pi_v = T1(3.141592653589793238462643383279502884195e+0L);

template <typename T1>
constexpr T1 rad_v = T1(3.141592653589793238462643383279502884195e+0L) /
T1(180);

template <typename T1>
constexpr T1 sqrt2_v = T1(1.414213562373095048801688724209698078569e+0L);

template <typename T1>
constexpr T1 sqrt3_div_2_v = T1(0.866025403784438646763723170752936183473L);

template <typename T1>
constexpr T1 sqrt3_v = T1(1.732050807568877293527446341505872366945e+0L);

template <typename T1>
constexpr T1 sqrt5_v = T1(2.236067977499789696409173668731276235440L);

template <typename T1>
constexpr T1 sqrt7_v = T1(2.645751311064590590501615753639260425706L);

template <typename T1>
constexpr T1 sqrtpi_v = T1(1.772453850905516027298167483341145182798L);

template <typename T1>
constexpr T1 sqrttau_v = T1(2.506628274631000502415765284811045253010L);

template <typename T1>
constexpr T1 tau_v = T1(6.283185307179586476925286766559005768391L);

template <typename T1>
constexpr T1 two_div_pi_v = T1(0.636619772367581343075535053490057448138L);

template <typename T1>
constexpr T1 two_div_sqrtpi_v = T1(1.128379167095512573896158903121545171688L);

constexpr double catalan = catalan_v<double>;
constexpr double cbrt3 = cbrt3_v<double>;
constexpr double cbrtpi = cbrtpi_v<double>;
constexpr double deg = deg_v<double>;
constexpr double e = e_v<double>;
constexpr double egamma = egamma_v<double>;
constexpr double inv_e = inv_e_v<double>;
constexpr double inv_pi = inv_pi_v<double>;
constexpr double inv_sqrt2 = inv_sqrt2_v<double>;
constexpr double inv_sqrt3 = inv_sqrt3_v<double>;
constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
constexpr double inv_tau = inv_tau_v<double>;
constexpr double ln10 = ln10_v<double>;
constexpr double ln2 = ln2_v<double>;
constexpr double ln3 = ln3_v<double>;
constexpr double lnpi = lnpi_v<double>;
constexpr double lnsqrttau = lnsqrttau_v<double>;
constexpr double log10e = log10e_v<double>;
constexpr double log2e = log2e_v<double>;
constexpr double phi = phi_v<double>;
constexpr double pi = pi_v<double>;
constexpr double pi_4 = pi_4_v<double>;
constexpr double pi_4_div_3 = pi_4_div_3_v<double>;
constexpr double pi_sqr_div_6 = pi_sqr_div_6_v<double>;
constexpr double rad = rad_v<double>;
constexpr double sqrt2 = sqrt2_v<double>;
constexpr double sqrt3 = sqrt3_v<double>;
constexpr double sqrt3_div_2 = sqrt3_div_2_v<double>;
constexpr double sqrt5 = sqrt5_v<double>;
constexpr double sqrt7 = sqrt7_v<double>;
constexpr double sqrtpi = sqrtpi_v<double>;
constexpr double sqrttau = sqrttau_v<double>;
constexpr double tau = tau_v<double>;
constexpr double two_div_pi = two_div_pi_v<double>;
constexpr double two_div_sqrtpi = two_div_sqrtpi_v<double>;
} // namespace numbers
} // namespace c10
