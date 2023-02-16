#pragma once

#include <ir_all_nodes.h>

#include <vector>

// Note: [The Mathematics of Integer Arithmetic]
//
// Note to reader: If you just want to know what rules we are using to simplify
// integer expressions, please go to section "Summary" in the end directly. But
// if you want to know why we are using these rules, why these rules are correct
// and why some others are not correct, or why some rule has some specific
// condition, please continue reading.
//
// 1. Pitfalls: Integer Arithmetic vs Real Arithmetic
// ==================================================
//
// We learnt arithmetic from as early as elementary school, and have been used
// to simplify expressions using rules like a(b+c) = ab+ac, but extra care is
// needed when dealing with integer divisions (such as Euclidean division as
// defined in number theory, truncation division as used by C++, and floor
// division used by Python). Because unlike real numbers, integer division can
// be anti-intuitive:
// - (a + b) / c != a/c + b/c, for example: (1 + 1) / 2 != 1/2 + 1/2
// - (a / b) * c != a / (b / c), for example: (64 / 256) * 4 != 64 / (256 / 4)
// - (a / b) * c != a * (c / b), for example: (1 / 4) * 256 != 1 * (256 / 4)
// To make sure we are doing the correct thing, it is crucial that we are clear
// about the underlying mathematics, especially abstract algebra and number
// theory. Abstract algebra tells us that which rule we are used to is still
// valid, and which is not. Number theory helps us to derive rules applicable to
// integer division and modulus.
//
// 2. A Review of Abstract Algebra
// ===============================
//
// The real numbers together with binary operators +-*/ form a field
// https://en.wikipedia.org/wiki/Field_(mathematics)
// which is defined by its basic properties:
//
// 1) Associativity: a + (b + c) = (a + b) + c, and a * (b * c) = (a * b) * c.
// 2) Commutativity: a + b = b + a, and a * b = b * a.
// 3) Distributivity: a * (b + c) = (a * b) + (a * c).
// 4) Identity: there exist two different elements 0 and 1 such that
//    a + 0 = a and a * 1 = a.
// 5) Additive inverses: for every a, there exists an element, denoted -a, such
//    that a + (-a) = 0.
// 6) Defition of subtraction: a - b := a + (-b)
// 7) Multiplicative inverses: for every a != 0, there exists an element,
//    denoted by 1/a, such that a * (1/a) = 1.
// 8) Defition of division: a / b := a * (1/b)
//
// Thanks to the field properties, div of real number has the following
// properties:
// 9) Associativity:
// - a*(b/c) = (a*b)/c
// - a/(b*c) = (a/b)/c
// - a/(b/c) = (a/b)*c = (a*c)/b
// 10) Right Distributivity: (a+b)/c = a/c + b/c
//
// For the set of integers Z, 7) is no longer true, therefore, the division can
// not be defined as 8), and as a result, neither 9) nor 10) is true anymore.
// So, the mathematical structure of integer arithmetics is not as strong as a
// field.
//
// The integer arithmetics form an Euclidean domain:
// https://en.wikipedia.org/wiki/Euclidean_domain
//
// The relationship between Euclidean domains and other algebraic structures is
// as follows (A > B means A is more general than B, that is, every B is an A,
// that is, B has more structure/stronger properties than A):
//   rings > commutative rings > [...] > Euclidean domains > fields
//
// where "commutative rings" is almost a field except that:
// - in 4), 0 and 1 are not required to be different
// - in 7) and 8) are not required to be true
//
// the [...] above can be further expanded as:
//   > integral domains > GCD domains > unique factorization domains >
//
// The expanded [...] and Euclidean domains are made by gradually adding basic
// properties of integers to the recipe. These properties include:
// - integral domains:
//   - 0 != 1
//   - if a != 0 and b != 0, then ab != 0
//   - cancellation property: if a != 0, then ab=ac implies b=c.
// - GCD domains: the existence of greatest common divisor
// - unique factorization domains: every non-zero element can be uniquely
//   written as a product of prime elements (for example, 12 = 2*2*3)
// - Euclidean domains: the fundamental division-with-remainder property (will
//   explain in section "A Review of Number theory")
//
// 3. A Review of Number theory
// ============================
//
// The definition of div and mod comes from the Division Theorem  (aka. the
// fundamental division-with-remainder property, aka. Euclid's division lemma)
// described as below:
//
// Theorem 3.1 (Division Theorem): for any integers a and b (b != 0), there
// exist unique integers q and r such that:
// 1) 0 <= r < |b|
// 2) a = bq + r
//
// if we define a/b := q, and a%b := r, then we can write 2) as:
//   a = (a/b)b + a%b
// I will call the above equation "the fundamental division-with-remainder
// equation" in later contexts.
//
// In this section and later sections, I will use \ for true division, and / for
// integer div. For example, 5\2 = 2.5, 5/2 = 2.
//
// Note that the a/b and a%b in number theory is not the same as those in C++,
// as explained in later section "Implementations of Div and Mod".
//
// A very important concept in number theory is "congruence", as described
// below:
//
// Definition 3.1: For c != 0, if (a-b)\c is an integer, then we say a is
// congruent to b modulo c, written as a = b (mod c).
//
// Note that the "(mod k)" in the statement "a = b (mod k)" should be understood
// as a qualifier for the statement "a = b" that modifies the meaning of the
// original statement [P] to "[P] under the modular arithmetic". It should NOT
// be understood as an operator operating on "b", i.e. "a = (b (mod k))".
//
// Theorem 3.2: The congruence is an equivalence relationship, that is:
// - a = a (mod c)
// - if a = b (mod c), then b = a (mod c)
// - if a = b (mod c) and b = d (mod c), then a = d (mod c)
//
// Theorem 3.3: The congruence also has the following properties:
// - a = b (mod c) iff a = b (mod (-c))
// - if a = a' (mod c), b = b' (mod c), then
//   - a + b = a' + b' (mod c)
//   - ab = a'b' (mod c)
// - if bd = bd' (mod c) and gcd(b, c) = 1, then d = d' (mod c)
//
// 4. Some More Theorems
// =====================
//
// In this section, I will prove a few theorems that I didn't find in textbooks,
// but still feel useful for us. The / and % are under the definition of number
// theory, not under the definition of C++. The properties of / and % with the
// definition of C++ will be revisited in later section "Properties of Div and
// Mod Under Trunc Div".
//
// Bad news comes first:
//
// Theorem 4.1: Integer div is not associative:
// 1) a*(b/c) != (a*b)/c
// 2) a/(b*c) != (a/b)/c
// 3) a/(b/c) != (a/b)*c != (a*c)/b
// Proof: a*(b/c) != (a*b)/c because of the counter example 2*(3/2) != (2*3)/2
// a/(b*c) != (a/b)/c because of the counter example 4/((-1)*5) != (4/(-1))/5
// Note that 4/(-5) = 0, (-4)/5 = -1
// a/(b/c) != (a/b)*c != (a*c)/b because of the counter example that
// 5/(3/2), (5/3)*2, and (5*2)/3 are three different numbers
//
// Theorem 4.2: Integer div is NOT right distributive
// Proof: Counter example: (1+1)/2 != 1/2 + 1/2
//
// Then comes good properties:
//
// Theorem 4.3: a%b = a'%b is equivalent to a = a' (mod b)
// Proof: Direction ==>:
// a%b = a'%b is equivalent to a-(a/b)b = a'-(a'/b)b
// which is equivalent to (a-a')\b = (a/b-a'/b) = integer
// So a%b = a'%b ==> a = a' (mod b)
// Direction <==:
// if a = a' (mod b), then a = a' + kb.
// According to the Euclid's division lemma, a' = q'b + r',
// then a = (q'+k)b + r', where q'+k and r'. It is easy to verify that, if we
// define q = q'+k and r = r', then a = qb + r also satisfies the condition in
// Euclid's division lemma. Thanks to the uniqueness of q and r, we have
// a%b = r = r' = a'%b
//
// Thanks to the great property of Theorem 4.3 some theorems below can be easily
// proved by converting % into congruence. But unfortunately, as we can see in
// section "Implementations of Div and Mod", for the trunc div in C++, the
// beautiful Theorem 4.3 does not hold, so many theorems in this section needs
// to be modified in order to be used as a rule to simplify C++ integer
// expressions.
//
// Theorem 4.4: a = a % b (mod b)
// Proof: According to Euclid's division lemma, (a - a % b) \ b = q is integer
//
// Theorem 4.5: if 0 <= r < |a|, then r % a = r, r / a = 0
// Proof: This can be proved directly from Euclid's division lemma
//
// Theorem 4.6: a/(-b) = -a/b, a%(-b) = a%b
// Proof: a = bq+r is equivalent to a = (-b)(-q) + r
// Due to the uniqueness of q and r, we get our conclusion
//
// Theorem 4.7: (a + b) % c = (a % c + b % c) % c
// Proof: According to Theorem 4.3, this is just to prove
// a + b = a % c + b % c (mod c)
// Because of Theorem 4.4, we have a = a % c (mod c), b = b % c (mod c)
// applying Theorem 3.3, we get what we need.
//
// Theorem 4.8: (a * b) % c = (a % c * b % c) % c
// Proof: Similar to above
//
// Theorem 4.9: If a is a multiple of b, then a % b = 0
// Proof: This can be proved directly from Euclid's division lemma
//
// Theorem 4.10: If b is a multiple of c, then we have: a*(b/c) = (a*b)/c
// Proof: If b is a multiple of c, then (a*b)%c = a*(b%c) = 0
// From the fundamental division-with-remainder equation, we know that:
// b = (b/c)*c + b%c ... (eq 1)
// (a*b) = ((a*b)/c)*c + (a*b)%c ... (eq 2)
// multiply a to both side of (eq 1), we get:
// (a*b) = a*(b/c)*c + a*(b%c)
// subtract (eq 2) by the above equation, we have:
// 0 = [(a*b)/c - a*(b/c)]*c + [(a*b)%c - a*(b%c)]
// The second term in the above equation is 0, so we have
// 0 = [(a*b)/c - a*(b/c)] * c
// Because c != 0, we have (a*b)/c = a*(b/c)
//
// Theorem 4.11: If b > 0, then a/(b*c) = (a/b)/c
// Proof: from the fundamental division-with-remainder equation, we have
// a = (a/b)b + a%b = (((a/b)/c)c + (a/b)%c)b + a%b
//   = ((a/b)/c)*bc + (a%b + ((a/b)%c)*b) ... (eq 1)
// where 0 <= a%b < b and 0 <= (a/b)%c*b <= (|c| - 1)*b,
// as a result, we have 0 <= (a%b + ((a/b)%c)*b) < |bc|,
// from the fundamental division-with-remainder equation, we can uniquely
// decompose a as a = (a/(bc))*(bc) + a%(bc)  ... (eq 2)
// since a%b + ((a/b)%c)*b is in the correct range of a%(bc) and due to the
// uniqueness of this decomposition, comparing (eq 1) and (eq 2) have
// a/(bc) = (a/b)/c and a%(bc) = a%b + ((a/b)%c)*b
//
// Theorem 4.12: If b > 0, then a % (b * c) = a % b + ((a / b) % c) * b
// Proof: Already proved in the proof of Theorem 4.11
//
// Theorem 4.13: If d > 0 and d divides a and b, then
// a % b = ((a / d) % (b / d)) * d
// Proof: From the fundamental division-with-remainder equation, we have
// b = d * (b / d), then a % b = a % (d * (b / d))
// From Theorem 4.12, we have:
// a % b = a % (d * (b / d)) = a % d + ((a / d) % (b / d)) * d
// according to Theorem 4.9, a % d = 0, so we have
// a % b = ((a / d) % (b / d)) * d
//
// Theorem 4.14: If b is a multiple of c and c > 0, then a/(b/c) = (a*c)/b
// Proof: If b is a multiple of c, then we have b % c = 0
// Also, according to Theorem 4.13, (ac) % b = (a % (b / c)) * c
// From the fundamental division-with-remainder equation, we have:
// ac = ((ac)/b)b + (ac)%b ... (eq 1)
// a = (a/(b/c))(b/c) + a%(b/c) ... (eq 2)
// Multiply c to both side of (eq 2), and note that b is a multiple of c:
// ac = (a/(b/c))b + (a%(b/c)) * c
// subtrace the above equation with (eq 1), we get
// 0 = [a/(b/c) - (ac)/b] * b + [(a%(b/c))*c - (ac)%b]
// The second term in the above equation is 0, so we have
// 0 = [a/(b/c) - (ac)/b] * b
// because b != 0, we get a/(b/c) = (ac)/b
//
// Theorem 4.15: If a % c + b % c < |c|, then (a+b)/c = a/c + b/c
// Proof: From Theorem 4.7 and Theorem 4.5, we have
// (a + b) % c = (a % c + b % c) % c = a % c + b % c
// From the fundamental division-with-remainder equation, we have:
// (a + b) = ((a + b) / c) * c + (a + b) % c
// a = (a / c) * c + a % c
// b = (b / c) * c + b % c
// Adding the last two equations and subtract with the first equation:
// 0 = [(a/c + b/c) - (a+b)/c] * c + [a % c + b % c - (a + b) % c]
// The second term in the above equation is 0, so we have
// 0 = [(a/c + b/c) - (a+b)/c] * c
// Because c != 0, we have (a/c + b/c) = (a+b)/c
//
// 5. Implementations of Div and Mod
// =================================
//
// Unfortunately, modern hardwares and programming languages does not implement
// div and mod consistent with Euclid's division lemma, although these
// implementations can be converted with Euclid's division easily. The
// implementations of div and mod depends on programming languages. The
// comparison of these implementations and their properties are discussed in the
// following paper:
//
// Boute, Raymond T. "The Euclidean definition of the functions div and mod."
// ACM Transactions on Programming Languages and Systems (TOPLAS) 14.2 (1992):
// 127-144.
//
// I will summarize some useful points from the above paper here, and add my own
// comments:
//
// For a >= 0 and b > 0, all implementation of a/b and a%b are the same and
// consistent with the Euclid's division. So no brainer in this region.
//
// Except for a few Languages (ISO Standard Pascal, Algol, Ada) all
// implementations are consistent with the fundamental division-with-remainder
// equation, although the range and sign of r can be different and the value of
// q can be different by 1. Implementations not satisfying the fundamental
// division-with-remainder equation is considered wrong because it has no
// mathematical properties. For all implementations, |a % b| < |b|.
//
// Common implementations are:
//
// trunc div (round to zero):
// a/b := trunc(a \ b)
// a%b defined by the fundamental division-with-remainder equation
//
// floor div:
// a/b := floor(a \ b)
// a%b defined by the fundamental division-with-remainder equation
//
// For C89, the result of negative div is not specified. C99 and C++ uses trunc
// div. Python and PyTorch uses floor div. We will only be interested in trunc
// div here because we use C++.
//
// The properties of trunc div are:
// 1) Good: (-a)/b = -(a/b) = a/(-b)
// 2) Good: (-a)%b = -(a%b) = a%(-b)
// 3) Bad: a % b = a' % b is not equivalent to a = a' (mod b)
//
// 6. Properties of Div and Mod Under Trunc Div
// ============================================
//
// In this section, I will study trunc div and its properties. I will first
// redefine trunc div using the same language as in Euclid's division lemma,
// which will be convenient for proving theorems. I will then prove that this
// new definition is equivalent to the definition of trunc div as described in
// section "Implementations of Div and Mod". Then I will study the theorems in
// section "Some More Theorems" to find out which is true and which needs
// change. All / and % in this section are using trunc div.
//
// Note that the trunc div is the definition of div and mod in C and C++, as C99
// standard says: When integers are divided, the result of the / operator is the
// algebraic quotient with any fractional part discarded. If the quotient a/b is
// representable, the expression (a/b)*b + a%b shall equal a.
//
// Definition 6.0: For any integers a and b (b != 0), there exist unique
// integers q and r such that
// 1) if a >= 0, 0 <= r < |b|; if a < 0, -|b| < r <= 0.
// 2) a = bq + r
// We can then define a/b := q, a%b := r
//
// Theorem 6.0: Definition 6.0 is equivalent to the definition of trunc div in
// section "Implementations of Div and Mod"
// Proof: trunc(a\b) is to remove the non-integer portion of a\b. To shift a by
// r, which obtains a - r, to the nearest multiple of b. To shift towards zero,
// r and a should have the same sign. Also, |r| must be smaller than |b|
// otherwise it won't be the nearest multiple of b, and this shift is unique.
// We therefore proved 1) and 2) above and the uniqueness.
//
// Now let's look at theorems in "Some More Theorems" to see how they should be
// modified when translating to the language of trunc div. Theorems will be
// numbered consistently, that is, theorem 6.x is the modified version of
// theorem 4.x.
//
// Theorem 6.1: Associativity of trunc div:
// 1) a*(b/c) != (a*b)/c
// 2) a/(b*c) = (a/b)/c
// 3) a/(b/c) != (a/b)*c != (a*c)/b
// Note that for 2), it is now a "=" instead of a "!=".
// Proof: for 1) and 3), the same counter example as in Theorem 4.1 applies.
// For 2), from Definition 6.0, we have
// a = (a/b)b + a%b = (((a/b)/c)c + (a/b)%c)b + a%b
//   = ((a/b)/c)*bc + (a%b + ((a/b)%c)*b)  ... (eq 1)
//   = ((a/b)/c)*bc + (a%b + ((a/|b|)%c)*|b|)
// if a >= 0, then 0 <= a%b < |b|, 0 <= (a/|b|)%c*|b| <= (|c| - 1)|b|,
// as a result, we have 0 <= (a%b + ((a/|b|)%c)*|b|) < |bc|,
// from Definition 6.0, we can uniquely decompose a as
// a = (a/(bc))*(bc) + a%(bc) ... (eq 2)
// since a%b + ((a/b)%c)*b is in the correct range of a%(bc) and due to the
// uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have
// a/(bc) = (a/b)/c and a%(bc) = a%b + ((a/b)%c)*b
// if a < 0, then -|b| < a%b <= 0, -(|c| - 1)|b| <= ((a/|b|)%c)*|b| <= 0,
// as a result, we have -|bc| <= (a%b + ((a/|b|)%c)*|b|) <= 0,
// from Definition 6.0, we can uniquely decompose a as
// a = (a/(bc))*(bc) + a%(bc) ... (eq 3)
// since a%b + ((a/b)%c)*b is in the correct range of a%(bc) and due to the
// uniqueness of this decomposition, comparing (eq 1) and (eq 2), we have
// a/(bc) = (a/b)/c and a%(bc) = a%b + ((a/b)%c)*b
//
// Theorem 6.2: Integer div is NOT right distributive
// Proof: the same counter example as in Theorem 4.2 applies.
//
// For trunc div, Theorem 4.3 no longer holds, because -3 = 2 (mod 5), however,
// -3 % 5 = -3, but 2 % 5 = 2.
//
// Theorem 6.3:
// 1) a%b = a'%b = 0 is equivalent to a = a' = 0 (mod b)
// 2) a%b = a'%b != 0 is equivalent to a = a' != 0 (mod b) and sign(a)=sign(a')
// 3) a%b = a'%b + |b| is equivalent to a = a' != 0 (mod b) and a>0 and a'<0
// Proof: For 1):
// a%b = a'%b = 0 is equivalent to a=bq and a'=bq', which is equivalent to
// a = a' = 0 (mod b)
// For 2) Direction ==>:
// a%b = a'%b is equivalent to a-(a/b)b = a'-(a'/b)b
// which is equivalent to (a-a')\b = (a/b-a'/b) = integer.
// So a%b = a'%b ==> a = a' (mod b),
// also, from 1), we know that a != 0 (mod b) and a' != 0 (mod b)
// From Definition 6.0, we know that since a%b is not 0,there is no overlap on
// the range of a%b for positive a and negative a. So the sign of a and a' must
// match, otherwise it is impossible to have a%b = a'%b.
// Direction <==:
// if a = a' (mod b), then a = a' + kb.
// According to Definition 0, a' = q'b + r', then a = (q'+k)b + r'.
// Because sign(a)=sign(a'), if r' is in the correct range for a', then it will
// also be in the correct range for a. Due to the uniqueness, a%b = r' = a'%b
// For 3) Direction ==>:
// If a%b = a'%b + |b|, then 0 < a%b < |b| and |b| < a'%b < 0,
// that is, a>0 and a'<0.
// Also, we have a-(a/b)b = a'-(a'/b)b + |b|, that is,
// (a-a')\b = (a/b-a'/b+sign(b)) = integer, so a = a' (mod b)
// also, from 1), we know that a != 0 (mod b) and a' != 0 (mod b)
// Direction <==:
// if a = a' != 0 (mod b), then a = a' + kb.
// According to Definition 0, a' = q'b + r', where -|b| < r' <= 0.
// from 1), we know that r' != 0, so -|b| < r' < 0.
// So a = (q'+k)b + r' = (q+k-sign(b))b + r' + |b|.
// Let q = q' + k, r = r' + |b|
// it is easy to verify that 0 < r < |b|
// Due to the uniqueness, a%b = r' + |b|, a/b = q+k-sign(b)
//
// Theorem 6.4: a = a % b (mod b)
// Proof: According to Definition 0, (a - a % b) \ b = q is integer
//
// Theorem 6.5: If -|a| < r < |a|, then r % a = r, r / a = 0
// Proof: This can be proved directly from Definition 0
//
// Theorem 6.6: a/(-b) = -a/b, a%(-b) = -a%b
// Proof: See "Implementations of Div and Mod", this is a written in the paper
//
// Theorem 6.7: If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
// where compatible_sign(a, b) is defined as ab>=0
// Proof: According to Theorem 6.3, this is just to prove
// a + b = a % c + b % c (mod c)
// Because of Theorem 6.4, we have a = a % c (mod c), b = b % c (mod c),
// applying Theorem 3.3, we get what we want
//
// Theorem 6.8: If compatible_sign(a, b), then (a * b) % c = (a % c * b % c) % c
// where compatible_sign(a, b) is defined as ab>=0
// Proof: Similar to above
//
// Theorem 6.9: If a is a multiple of b, then a % b = 0
// Proof:  This can be proved directly from Euclid's division lemma
//
// Theorem 6.10: If b is a multiple of c, then we have: a*(b/c) = (a*b)/c
// Proof: Same proof as 4.10
//
// Theorem 6.11: a/(b*c) = (a/b)/c
// Proof: This is part of Theorem 6.1
//
// Theorem 6.12: a % (b * c) = a % b + ((a / b) % c) * b
// Proof: Already proved in the proof of Theorem 6.1
//
// Theorem 6.13: If d divides a and b, then a % b = ((a / d) % (b / d)) * d
// Proof: Same proof as 4.13
//
// Theorem 6.14: If b is a multiple of c, then a/(b/c) = (a*c)/b
// Proof: Same proof as 4.14
//
// Theorem 6.15: If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
// (a+b)/c = a/c + b/c
// Proof: From Theorem 6.7 and Theorem 6.5
// (a + b) % c = (a % c + b % c) % c = a % c + b % c
// The rest of the proof is the same as 4.15
//
// 7. Summary
// ==========
//
// In summary, in nvFuser, we can use the following rules to simplify integer
// expressions:
//
// A) Associativity of +: a + (b + c) = (a + b) + c
// B) Associativity of *: a * (b * c) = (a * b) * c
// C) Commutativity of +: a + b = b + a
// D) Commutativity of *: a * b = b * a
// E) Distributivity of * over +: a * (b + c) = (a * b) + (a * c)
// F) Distributivity of * over +: (a + b) * c = (a * c) + (b * c)
// G) (-a) / b = -(a / b) = a / (-b)
// H) (-a) % b = -(a % b) = a % (-b)
// I) If -|a| < r < |a|, then r % a = r, r / a = 0
// J) Distributivity of % over +:
//    If compatible_sign(a, b), then (a + b) % c = (a % c + b % c) % c
// K) Distributivity of % over *:
//    If compatible_sign(a, b), then (a * b) % c = (a % c * b % c) % c
// L) If a is a multiple of b, then a % b = 0
// M) If b is a multiple of c, then we have: a*(b/c) = (a*b)/c
// N) a / (b * c) = (a / b) / c
// O) If d divides a and b, then a % b = ((a / d) % (b / d)) * d
// P) If b is a multiple of c, then a/(b/c) = (a*c)/b
// Q) If compatible_sign(a, b) and -|c| < a % c + b % c < |c|, then
//    (a+b)/c = a/c + b/c
//
// Some examples on applying the above rules to simplify expressions:
//
// Example 7.1: Given that a >= 0 and b >= 0, simplify (a*4 + b) % 4
// Answer: (a*4 + b) % 4 = ((a*4)%4 + b%4) % 4 (Rule J)
// = (0 + b%4) % 4 (Rule L)
// = b % 4 % 4 (Basic math)
// = b % 4 (Rule I)
//
// Example 7.2: Given that 0 <= a < 3, simplify a % 4
// Answer: a % 4 = a (Rule I)
//
// Example 7.3: Simplify (a * 256) / 4
// Answer: (a * 256) / 4 = a * (256 / 4) (Rule M)
// = a * 64 (Basic math)
//
// Example 7.4: Simplify (a / 4) / 64
// Answer: (a / 4) / 64 = a / (4 * 64) (Rule N)
// = a / 256 (Basic math)
//
// Example 7.5: Simplify (a * 64) % 256 / 4
// Answer: (a * 64) % 256 / 4 = ((a % 4) * 64) / 4 (Rule O)
// = (a % 4) * (64 / 4) (Rule M)
// = (a % 4) * 16 (Basic math)
//
// Example 7.6: Simplify (a * 4) / 256
// Answer: (a * 4) / 256 = a / (256 / 4) (Rule P)
// = a / 64 (Basic math)
//
// Example 7.7: Given that a >= 0 and b >= 0, simplify (a * 256 + b) / 4
// Answer: because (a * 256) % 4 = 0, we have
// (a * 256 + b) / 4 = a * 256 / 4 + b / 4 (Rule Q)
// = a * (256 / 4) + b / 4 (Rule M)
// = a * 64 + b / 4 (Basic math)
//
// Example 7.8: Given that a >= 0 and 0 <= b < 4, simplify (a * 4 + b) / 4
// Answer: Similar to above, we have
// (a * 4 + b) / 4 = a + b / 4
// = a + 0 (Rule I)
// = a

namespace nvfuser {

// Information for a single variable. Possible values that this variable can
// take is: start, start + step, start + 2 * step, ... (< stop), which is
// similar to the loop variable of for loop:
//   for variable in range(start, stop, step)
struct VarInfo {
  Val* variable = nullptr;
  Val* start = nullptr;
  Val* stop = nullptr;
  Val* step = nullptr;
  // If this variable is constant to nvRTC. Note that a constant to nvRTC might
  // not be a constant to nvFuser. For example, if I have loop
  // #pragma unroll
  // FOR i1 in ...:
  //   ...
  // Then `i1` is a compile constant to nvRTC, but not a compile time constant
  // to nvFuser.
  bool is_compile_time_const = false;
};

// Analyze expression register usage
enum class RegisterType { GeneralPurpose, Uniform, Immediate, Unknown };
RegisterType getRegisterType(Val* value);

// Simplify expressions with the given information of variables.
//
// The argument `variables` specifies which scalar are considered variable and
// some information about these variables. Any scalar not contained in
// `variables` are considered constants. Tensors are always considered as
// variables, regardless of if it is specified in `variables`.
//
// Note that in `variables`, the order matters. This order specifies how we
// should organize associative and commutative expressions. For example, if the
// `variables` is {a, b, c, d}, then we will simplify (a + d) + (c + b) as
// ((a + b) + c) + d. Tensors are always considered as at the right of all
// scalars, regardless of if it is inside `variables` or not.
// See note [Reordering associative and commutative operators] for detailed
// information about this reordering.
TORCH_CUDA_CU_API Val* simplifyExpr(
    Val* value,
    const std::list<VarInfo>& variables = {});

} // namespace nvfuser
