#pragma once

namespace c10 {
namespace numbers {
template<typename T> inline constexpr T catalan_v = T(0.915965594177219015054603514932384110773L);
template<typename T> inline constexpr T cbrt3_v = T(1.442249570307408382321638310780109588390L);
template<typename T> inline constexpr T cbrtpi_v = T(1.464591887561523263020142527263790391736L);
template<typename T> inline constexpr T deg_v = T(180) / T(3.141592653589793238462643383279502884195e+0L);
template<typename T> inline constexpr T e_v = T(2.718281828459045235360287471352662497759e+0L);
template<typename T> inline constexpr T egamma_v = T(5.772156649015328606065120900824024310432e-1L);
template<typename T> inline constexpr T inv_e_v = T(0.367879441171442321595523770161460867446L);
template<typename T> inline constexpr T inv_pi_v = T(3.183098861837906715377675267450287240691e-1L);
template<typename T> inline constexpr T inv_sqrt2_v = T(0.707106781186547524400844362104849039285L);
template<typename T> inline constexpr T inv_sqrt3_v = T(5.773502691896257645091487805019574556475e-1L);
template<typename T> inline constexpr T inv_sqrtpi_v = T(5.641895835477562869480794515607725858438e-1L);
template<typename T> inline constexpr T inv_tau_v = T(0.159154943091895335768883763372514362035L);
template<typename T> inline constexpr T ln10_v = T(2.302585092994045684017991454684364207602e+0L);
template<typename T> inline constexpr T ln2_v = T(6.931471805599453094172321214581765680748e-1L);
template<typename T> inline constexpr T ln3_v = T(1.098612288668109691395245236922525704648L);
template<typename T> inline constexpr T lnpi_v = T(1.144729885849400174143427351353058711646L);
template<typename T> inline constexpr T lnsqrttau_v = T(0.918938533204672741780329736405617639862L);
template<typename T> inline constexpr T log10e_v = T(4.342944819032518276511289189166050822940e-1L);
template<typename T> inline constexpr T log2e_v = T(1.442695040888963407359924681001892137427e+0L);
template<typename T> inline constexpr T phi_v = T(1.618033988749894848204586834365638117720e+0L);
template<typename T> inline constexpr T pi_4_div_3_v = T(4.188790204786390984616857844372670512253L);
template<typename T> inline constexpr T pi_4_v = T(12.56637061435917295385057353311801153678L);
template<typename T> inline constexpr T pi_sqr_div_6_v = T(1.644934066848226436472415166646025189221L);
template<typename T> inline constexpr T pi_v = T(3.141592653589793238462643383279502884195e+0L);
template<typename T> inline constexpr T rad_v = T(3.141592653589793238462643383279502884195e+0L) / T(180);
template<typename T> inline constexpr T sqrt2_v = T(1.414213562373095048801688724209698078569e+0L);
template<typename T> inline constexpr T sqrt3_div_2_v = T(0.866025403784438646763723170752936183473L);
template<typename T> inline constexpr T sqrt3_v = T(1.732050807568877293527446341505872366945e+0L);
template<typename T> inline constexpr T sqrt5_v = T(2.236067977499789696409173668731276235440L);
template<typename T> inline constexpr T sqrt7_v = T(2.645751311064590590501615753639260425706L);
template<typename T> inline constexpr T sqrtpi_v = T(1.772453850905516027298167483341145182798L);
template<typename T> inline constexpr T sqrttau_v = T(2.506628274631000502415765284811045253010L);
template<typename T> inline constexpr T tau_v = T(6.283185307179586476925286766559005768391L);
template<typename T> inline constexpr T two_div_pi_v = T(0.636619772367581343075535053490057448138L);
template<typename T> inline constexpr T two_div_sqrtpi_v = T(1.128379167095512573896158903121545171688L);

inline constexpr double catalan = catalan_v<double>;
inline constexpr double cbrt3 = cbrt3_v<double>;
inline constexpr double cbrtpi = cbrtpi_v<double>;
inline constexpr double deg = deg_v<double>;
inline constexpr double e = e_v<double>;
inline constexpr double egamma = egamma_v<double>;
inline constexpr double inv_e = inv_e_v<double>;
inline constexpr double inv_pi = inv_pi_v<double>;
inline constexpr double inv_sqrt2 = inv_sqrt2_v<double>;
inline constexpr double inv_sqrt3 = inv_sqrt3_v<double>;
inline constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
inline constexpr double inv_tau = inv_tau_v<double>;
inline constexpr double ln10 = ln10_v<double>;
inline constexpr double ln2 = ln2_v<double>;
inline constexpr double ln3 = ln3_v<double>;
inline constexpr double lnpi = lnpi_v<double>;
inline constexpr double lnsqrttau = lnsqrttau_v<double>;
inline constexpr double log10e = log10e_v<double>;
inline constexpr double log2e = log2e_v<double>;
inline constexpr double phi = phi_v<double>;
inline constexpr double pi = pi_v<double>;
inline constexpr double pi_4 = pi_4_v<double>;
inline constexpr double pi_4_div_3 = pi_4_div_3_v<double>;
inline constexpr double pi_sqr_div_6 = pi_sqr_div_6_v<double>;
inline constexpr double rad = rad_v<double>;
inline constexpr double sqrt2 = sqrt2_v<double>;
inline constexpr double sqrt3 = sqrt3_v<double>;
inline constexpr double sqrt3_div_2 = sqrt3_div_2_v<double>;
inline constexpr double sqrt5 = sqrt5_v<double>;
inline constexpr double sqrt7 = sqrt7_v<double>;
inline constexpr double sqrtpi = sqrtpi_v<double>;
inline constexpr double sqrttau = sqrttau_v<double>;
inline constexpr double tau = tau_v<double>;
inline constexpr double two_div_pi = two_div_pi_v<double>;
inline constexpr double two_div_sqrtpi = two_div_sqrtpi_v<double>;
}
}
