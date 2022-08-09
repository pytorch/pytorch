#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/stieltjes.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/sin_pi.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>
#include <ATen/native/special_functions/detail/is_even_integer.h>
#include <ATen/native/special_functions/detail/gamma.h>
#include <ATen/native/special_functions/prime_number.h>

namespace at::native::special_functions {
constexpr std::uint32_t
prime_number(std::uint16_t n);
}

namespace at::native::special_functions::detail {
template<typename T1>
T1
riemann_zeta_laurent_series(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto p = T2(1);
  auto q = T2(1) / (s - T2(1)) + STIELTJES[0];

  for (unsigned int k = 1; k < STIELTJES_SIZE; ++k) {
    p = p * (-(s - T2(1)) / k);
    q = q + (STIELTJES[k] * p);

    if (std::abs(STIELTJES[k] * p) < std::numeric_limits<T3>::epsilon() * std::abs(q)) {
      break;
    }
  }

  return q;
}

template<typename T1>
T1
riemann_zeta_sum(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::real(s) < T3(1)) {
    throw std::domain_error("bad `s`");
  } else if (std::real(s) > T3(1)) {
    T2 p = T2(1);
    T2 q;

    for (unsigned int k = 2; k < 10000; k++) {
      q = std::pow(T2(k), -s);
      p = p + q;

      if (std::abs(q) < std::numeric_limits<T3>::epsilon() * std::abs(p)
          || (std::abs(q) < std::numeric_limits<T3>::epsilon()
              && std::abs(p) < T3(100) * std::numeric_limits<T3>::epsilon())) {
        break;
      }
    }

    return p;
  } else {
    return std::pow(T3(2) * c10::numbers::pi_v<T3>, s) * sin_pi(T3{0.5L} * s) * gamma(T2(1) - s)
        * riemann_zeta_sum(T2(1) - s) / c10::numbers::pi_v<T3>;
  }
}

template<typename T1>
T1
riemann_zeta_m_1_globally_convergent_series(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto zeta_m_1 = T2(0);
  // This for loop starts at 1 because we already calculated the
  // value of the zeroeth order in zeta_m_1 above
  auto num = T3{0.25L};
  for (unsigned int i = 1; i < 10000; i++) {
    bool punt = false;
    auto binom = T3(1);
    auto term = T2(0);
    // This for loop starts at 1 because we already calculated the value
    // of the zeroeth order in term above.
    for (unsigned int j = 1; j <= i; j++) {
      binom *= -T3(i - j + 1) / T3(j);
      if (std::abs(binom) > std::exp(std::numeric_limits<T3>::max_exponent10 * std::log(T3(10)) - T3(1))) {
        // This only gets hit for x << 0.
        punt = true;
        break;
      }
      term += binom * std::pow(T2(1 + j), -s);
    }
    if (punt)
      break;
    term *= num;
    zeta_m_1 += term;
    if (std::abs(term) < std::numeric_limits<T3>::epsilon() * std::abs(zeta_m_1)
        || (std::abs(term) < std::numeric_limits<T3>::epsilon()
            && std::abs(zeta_m_1) < T3(100) * std::numeric_limits<T3>::epsilon()))
      break;
    num *= T3{0.5L};
  }

  return (zeta_m_1 + (std::pow(T2(2), T2(1) - s))) / (T2(1) - std::pow(T2(2), T2(1) - s));
}

template<typename T1>
T1
riemann_zeta_globally_convergent_series(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::real(s) < T3(0)) {
    if (is_even_integer(s)) {
      return T2(0);
    } else {
      return riemann_zeta_globally_convergent_series(T2(1) - s)
          * (std::pow(T3(2) * c10::numbers::pi_v<T3>, s) * sin_pi(T3{0.5L} * s) * gamma(T2(1) - s)
              / c10::numbers::pi_v<T3>);
    }
  } else {
    return T1(1) + riemann_zeta_m_1_globally_convergent_series(s);
  }
}

template<typename T1>
T1
riemann_zeta_product(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto p = T2(1);

  for (unsigned long j = 0; j < 10000; j++) {
    p = p * (T2(1) - std::pow(T3(at::native::special_functions::prime_number(j)), -s));

    if (std::abs(T1(1) - (T2(1) - std::pow(T3(at::native::special_functions::prime_number(j)), -s)))
        < std::numeric_limits<T3>::epsilon()) {
      break;
    }
  }

  return T1(1) / p;
}

constexpr size_t
    s_num_zetam1 = 121;

constexpr long double
    s_zetam1[s_num_zetam1]
    {
        -1.5L,                                          //   0
        std::numeric_limits<long double>::infinity(),   //   1
        6.449340668482264364724151666460251892177e-1L,  //   2
        2.020569031595942853997381615114499907647e-1L,  //   3
        8.232323371113819151600369654116790277462e-2L,  //   4
        3.692775514336992633136548645703416805713e-2L,  //   5
        1.734306198444913971451792979092052790186e-2L,  //   6
        8.349277381922826839797549849796759599843e-3L,  //   7
        4.077356197944339378685238508652465258950e-3L,  //   8
        2.008392826082214417852769232412060485604e-3L,  //   9
        9.945751278180853371459589003190170060214e-4L,  //  10
        4.941886041194645587022825264699364686068e-4L,  //  11
        2.460865533080482986379980477396709604160e-4L,  //  12
        1.227133475784891467518365263573957142749e-4L,  //  13
        6.124813505870482925854510513533374748177e-5L,  //  14
        3.058823630702049355172851064506258762801e-5L,  //  15
        1.528225940865187173257148763672202323739e-5L,  //  16
        7.637197637899762273600293563029213088257e-6L,  //  17
        3.817293264999839856461644621939730454694e-6L,  //  18
        1.908212716553938925656957795101353258569e-6L,  //  19
        9.539620338727961131520386834493459437919e-7L,  //  20
        4.769329867878064631167196043730459664471e-7L,  //  21
        2.384505027277329900036481867529949350419e-7L,  //  22
        1.192199259653110730677887188823263872549e-7L,  //  23
        5.960818905125947961244020793580122750393e-8L,  //  24
        2.980350351465228018606370506936601184471e-8L,  //  25
        1.490155482836504123465850663069862886482e-8L,  //  26
        7.450711789835429491981004170604119454712e-9L,  //  27
        3.725334024788457054819204018402423232885e-9L,  //  28
        1.862659723513049006403909945416948061669e-9L,  //  29
        9.313274324196681828717647350212198135677e-10L, //  30
        4.656629065033784072989233251220071062704e-10L, //  31
        2.328311833676505492001455975940495024831e-10L, //  32
        1.164155017270051977592973835456309516528e-10L, //  33
        5.820772087902700889243685989106305417368e-11L, //  34
        2.910385044497099686929425227884046410669e-11L, //  35
        1.455192189104198423592963224531842098334e-11L, //  36
        7.275959835057481014520869012338059265263e-12L, //  37
        3.637979547378651190237236355873273513051e-12L, //  38
        1.818989650307065947584832100730085030987e-12L, //  39
        9.094947840263889282533118386949087534482e-13L, //  40
        4.547473783042154026799112029488570339961e-13L, //  41
        2.273736845824652515226821577978691217250e-13L, //  42
        1.136868407680227849349104838025906441861e-13L, //  43
        5.684341987627585609277182967524068526363e-14L, //  44
        2.842170976889301855455073704942662033022e-14L, //  45
        1.421085482803160676983430714173953721447e-14L, //  46
        7.105427395210852712877354479956799457540e-15L, //  47
        3.552713691337113673298469534059343240771e-15L, //  48
        1.776356843579120327473349014400279865980e-15L, //  49
        8.881784210930815903096091386391386649172e-16L, //  50
        4.440892103143813364197770940268122986877e-16L, //  51
        2.220446050798041983999320094204660286072e-16L, //  52
        1.110223025141066133720544569921388238976e-16L, //  53
        5.551115124845481243723736590509454214979e-17L, //  54
        2.775557562136124172581632453854098189256e-17L, //  55
        1.387778780972523276283909490650087159020e-17L, //  56
        6.938893904544153697446085326249613606421e-18L, //  57
        3.469446952165922624744271496109153952849e-18L, //  58
        1.734723476047576572048972969937766807693e-18L, //  59
        8.673617380119933728342055067345929347336e-19L, //  60
        4.336808690020650487497023565906367637200e-19L, //  61
        2.168404344997219785013910168326102593523e-19L, //  62
        1.084202172494241406301271116539546929420e-19L, //  63
        5.421010862456645410918700404413613405660e-20L, //  64
        2.710505431223468831954621311921825336782e-20L, //  65
        1.355252715610116458148523399711726681995e-20L, //  66
        6.776263578045189097995298741000309894844e-21L, //  67
        3.388131789020796818085703100408435571778e-21L, //  68
        1.694065894509799165406492747048108912984e-21L, //  69
        8.470329472546998348246992605151870123760e-22L, //  70
        4.235164736272833347862270482171914722722e-22L, //  71
        2.117582368136194731844209444015663667353e-22L, //  72
        1.058791184068023385226500150767838272960e-22L, //  73
        5.293955920339870323813912246795908957429e-23L, //  74
        2.646977960169852961134116619883673563755e-23L, //  75
        1.323488980084899080309451049270121075322e-23L, //  76
        6.617444900424404067355245869046478332807e-24L, //  77
        3.308722450212171588946956563227359069812e-24L, //  78
        1.654361225106075646229923736818740547512e-24L, //  79
        8.271806125530344403671108096295678003592e-25L, //  80
        4.135903062765160926009383852215164090474e-25L, //  81
        2.067951531382576704395963965944918517449e-25L, //  82
        1.033975765691287099328403715492352137455e-25L, //  83
        5.169878828456431320410159441971315309917e-26L, //  84
        2.584939414228214268127816150081035315909e-26L, //  85
        1.292469707114106670038085128629065184730e-26L, //  86
        6.462348535570531803437454412518556478869e-27L, //  87
        3.231174267785265386134631538638949625204e-27L, //  88
        1.615587133892632521206406698623221009248e-27L, //  89
        8.077935669463162033155494313137423014210e-28L, //  90
        4.038967834731580825616620004023205489476e-28L, //  91
        2.019483917365790349161443228820936759716e-28L, //  92
        1.009741958682895153362818597460967711233e-28L, //  93
        5.048709793414475696133364781858743133105e-29L, //  94
        2.524354896707237824529247247973459828381e-29L, //  95
        1.262177448353618904396004317475305931953e-29L, //  96
        6.310887241768094495295138721816048328696e-30L, //  97
        3.155443620884047239436836171504799139404e-30L, //  98
        1.577721810442023616297279256834389142642e-30L, //  99
        7.888609052210118067801840968499904004972e-31L, // 100
        3.944304526105059027058642826413931148366e-31L, // 101
        1.972152263052529513529321413206965574183e-31L, // 102
        9.860761315262647567646607066034827870915e-32L, // 103
        4.930380657631323783823303533017413935458e-32L, // 104
        2.465190328815661891911651766508706967729e-32L, // 105
        1.232595164407830945955825883254353483864e-32L, // 106
        6.162975822039154729779129416271767419322e-33L, // 107
        3.081487911019577364889564708135883709661e-33L, // 108
        1.540743955509788682444782354067941854830e-33L, // 109
        7.888609052210118067801840968499904004972e-31L, // 110
        3.851859888774471706111955885169854637076e-34L, // 111
        1.925929944387235853055977942584927318538e-34L, // 112
        9.629649721936179265279889712924636592691e-35L, // 113
        4.814824860968089632639944856462318296345e-35L, // 114
        2.407412430484044816319972428231159148173e-35L, // 115
        1.203706215242022408159986214115579574086e-35L, // 116
        6.018531076210112040799931070577897870432e-36L, // 117
        3.009265538105056020399965535288948935216e-36L, // 118
        1.504632769052528010199982767644474467608e-36L, // 119
        7.523163845262640050999913838222372338039e-37L, // 120
    };

template<typename T1>
T1
riemann_zeta_m_1(T1 s) {
  using T2 = numeric_t<T1>;

  if (s == T2(1)) {
    return std::numeric_limits<T2>::infinity();
  }

  if (is_integer(s) && is_integer(s)() >= 0) {
    if (is_integer(s)() < static_cast<int>(s_num_zetam1)) {
      return T1(s_zetam1[is_integer(s)()]);
    } else {
      return T1(0);
    }
  } else if (std::abs(s - is_integer(s)()) < T2(100) * std::numeric_limits<T2>::epsilon()) {
    return riemann_zeta_laurent_series(s) - T2(1);
  } else if (std::real(s) > T2(0)) {
    return riemann_zeta_m_1_globally_convergent_series(s);
  } else {
    return std::pow(T2(2) * c10::numbers::pi_v<T2>, s) * sin_pi(T2{0.5L} * s) * gamma(T1(1) - s)
        * (T2(1) + riemann_zeta_m_1(T1(1) - s)) / c10::numbers::pi_v<T2> - T2(1);
  }
}

template<typename T1>
T1
riemann_zeta(T1 s) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(s)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (s == T2(1)) {
    return std::numeric_limits<T3>::infinity();
  } else if (is_integer(s) && is_integer(s)() >= 0) {
    return T3(1) + riemann_zeta_m_1(T3(is_integer(s)()));
  } else if (is_integer(s) && is_integer(s)() < 0) {
    return std::pow(T3(2) * c10::numbers::pi_v<T3>, T3(is_integer(s)())) * sin_pi(T3{0.5L} * T3(is_integer(s)()))
        * gamma(T3(1) - T3(is_integer(s)())) * (T3(1) + riemann_zeta_m_1(T3(1) - T3(is_integer(s)())))
        / c10::numbers::pi_v<T3>;
  } else if (std::real(s) < -T3(19)) {
    return riemann_zeta_product(T2(1) - s) * (riemann_zeta_product(T2(1) - s)
        * (std::pow(T3(2) * c10::numbers::pi_v<T3>, s) * sin_pi(T3{0.5L} * s) * std::exp(ln_gamma(T2(1) - s))
            / c10::numbers::pi_v<T3>));
  } else if (std::real(s) < std::numeric_limits<T3>::digits) {
    return riemann_zeta_globally_convergent_series(s);
  } else {
    return T2(1) + exp2(-s);
  }
}
}
