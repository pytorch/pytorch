from sympy import sin, Function, symbols, Dummy, Lambda, cos
from sympy.parsing.mathematica import parse_mathematica, MathematicaParser
from sympy.core.sympify import sympify
from sympy.abc import n, w, x, y, z
from sympy.testing.pytest import raises


def test_mathematica():
    d = {
        '- 6x': '-6*x',
        'Sin[x]^2': 'sin(x)**2',
        '2(x-1)': '2*(x-1)',
        '3y+8': '3*y+8',
        'ArcSin[2x+9(4-x)^2]/x': 'asin(2*x+9*(4-x)**2)/x',
        'x+y': 'x+y',
        '355/113': '355/113',
        '2.718281828': '2.718281828',
        'Cos(1/2 * π)': 'Cos(π/2)',
        'Sin[12]': 'sin(12)',
        'Exp[Log[4]]': 'exp(log(4))',
        '(x+1)(x+3)': '(x+1)*(x+3)',
        'Cos[ArcCos[3.6]]': 'cos(acos(3.6))',
        'Cos[x]==Sin[y]': 'Eq(cos(x), sin(y))',
        '2*Sin[x+y]': '2*sin(x+y)',
        'Sin[x]+Cos[y]': 'sin(x)+cos(y)',
        'Sin[Cos[x]]': 'sin(cos(x))',
        '2*Sqrt[x+y]': '2*sqrt(x+y)',   # Test case from the issue 4259
        '+Sqrt[2]': 'sqrt(2)',
        '-Sqrt[2]': '-sqrt(2)',
        '-1/Sqrt[2]': '-1/sqrt(2)',
        '-(1/Sqrt[3])': '-(1/sqrt(3))',
        '1/(2*Sqrt[5])': '1/(2*sqrt(5))',
        'Mod[5,3]': 'Mod(5,3)',
        '-Mod[5,3]': '-Mod(5,3)',
        '(x+1)y': '(x+1)*y',
        'x(y+1)': 'x*(y+1)',
        'Sin[x]Cos[y]': 'sin(x)*cos(y)',
        'Sin[x]^2Cos[y]^2': 'sin(x)**2*cos(y)**2',
        'Cos[x]^2(1 - Cos[y]^2)': 'cos(x)**2*(1-cos(y)**2)',
        'x y': 'x*y',
        'x  y': 'x*y',
        '2 x': '2*x',
        'x 8': 'x*8',
        '2 8': '2*8',
        '4.x': '4.*x',
        '4. 3': '4.*3',
        '4. 3.': '4.*3.',
        '1 2 3': '1*2*3',
        ' -  2 *  Sqrt[  2 3 *   ( 1   +  5 ) ]  ': '-2*sqrt(2*3*(1+5))',
        'Log[2,4]': 'log(4,2)',
        'Log[Log[2,4],4]': 'log(4,log(4,2))',
        'Exp[Sqrt[2]^2Log[2, 8]]': 'exp(sqrt(2)**2*log(8,2))',
        'ArcSin[Cos[0]]': 'asin(cos(0))',
        'Log2[16]': 'log(16,2)',
        'Max[1,-2,3,-4]': 'Max(1,-2,3,-4)',
        'Min[1,-2,3]': 'Min(1,-2,3)',
        'Exp[I Pi/2]': 'exp(I*pi/2)',
        'ArcTan[x,y]': 'atan2(y,x)',
        'Pochhammer[x,y]': 'rf(x,y)',
        'ExpIntegralEi[x]': 'Ei(x)',
        'SinIntegral[x]': 'Si(x)',
        'CosIntegral[x]': 'Ci(x)',
        'AiryAi[x]': 'airyai(x)',
        'AiryAiPrime[5]': 'airyaiprime(5)',
        'AiryBi[x]': 'airybi(x)',
        'AiryBiPrime[7]': 'airybiprime(7)',
        'LogIntegral[4]': ' li(4)',
        'PrimePi[7]': 'primepi(7)',
        'Prime[5]': 'prime(5)',
        'PrimeQ[5]': 'isprime(5)',
        'Rational[2,19]': 'Rational(2,19)',    # test case for issue 25716
        }

    for e in d:
        assert parse_mathematica(e) == sympify(d[e])

    # The parsed form of this expression should not evaluate the Lambda object:
    assert parse_mathematica("Sin[#]^2 + Cos[#]^2 &[x]") == sin(x)**2 + cos(x)**2

    d1, d2, d3 = symbols("d1:4", cls=Dummy)
    assert parse_mathematica("Sin[#] + Cos[#3] &").dummy_eq(Lambda((d1, d2, d3), sin(d1) + cos(d3)))
    assert parse_mathematica("Sin[#^2] &").dummy_eq(Lambda(d1, sin(d1**2)))
    assert parse_mathematica("Function[x, x^3]") == Lambda(x, x**3)
    assert parse_mathematica("Function[{x, y}, x^2 + y^2]") == Lambda((x, y), x**2 + y**2)


def test_parser_mathematica_tokenizer():
    parser = MathematicaParser()

    chain = lambda expr: parser._from_tokens_to_fullformlist(parser._from_mathematica_to_tokens(expr))

    # Basic patterns
    assert chain("x") == "x"
    assert chain("42") == "42"
    assert chain(".2") == ".2"
    assert chain("+x") == "x"
    assert chain("-1") == "-1"
    assert chain("- 3") == "-3"
    assert chain("α") == "α"
    assert chain("+Sin[x]") == ["Sin", "x"]
    assert chain("-Sin[x]") == ["Times", "-1", ["Sin", "x"]]
    assert chain("x(a+1)") == ["Times", "x", ["Plus", "a", "1"]]
    assert chain("(x)") == "x"
    assert chain("(+x)") == "x"
    assert chain("-a") == ["Times", "-1", "a"]
    assert chain("(-x)") == ["Times", "-1", "x"]
    assert chain("(x + y)") == ["Plus", "x", "y"]
    assert chain("3 + 4") == ["Plus", "3", "4"]
    assert chain("a - 3") == ["Plus", "a", "-3"]
    assert chain("a - b") == ["Plus", "a", ["Times", "-1", "b"]]
    assert chain("7 * 8") == ["Times", "7", "8"]
    assert chain("a + b*c") == ["Plus", "a", ["Times", "b", "c"]]
    assert chain("a + b* c* d + 2 * e") == ["Plus", "a", ["Times", "b", "c", "d"], ["Times", "2", "e"]]
    assert chain("a / b") == ["Times", "a", ["Power", "b", "-1"]]

    # Missing asterisk (*) patterns:
    assert chain("x y") == ["Times", "x", "y"]
    assert chain("3 4") == ["Times", "3", "4"]
    assert chain("a[b] c") == ["Times", ["a", "b"], "c"]
    assert chain("(x) (y)") == ["Times", "x", "y"]
    assert chain("3 (a)") == ["Times", "3", "a"]
    assert chain("(a) b") == ["Times", "a", "b"]
    assert chain("4.2") == "4.2"
    assert chain("4 2") == ["Times", "4", "2"]
    assert chain("4  2") == ["Times", "4", "2"]
    assert chain("3 . 4") == ["Dot", "3", "4"]
    assert chain("4. 2") == ["Times", "4.", "2"]
    assert chain("x.y") == ["Dot", "x", "y"]
    assert chain("4.y") == ["Times", "4.", "y"]
    assert chain("4 .y") == ["Dot", "4", "y"]
    assert chain("x.4") == ["Times", "x", ".4"]
    assert chain("x0.3") == ["Times", "x0", ".3"]
    assert chain("x. 4") == ["Dot", "x", "4"]

    # Comments
    assert chain("a (* +b *) + c") == ["Plus", "a", "c"]
    assert chain("a (* + b *) + (**)c (* +d *) + e") == ["Plus", "a", "c", "e"]
    assert chain("""a + (*
    + b
    *) c + (* d
    *) e
    """) == ["Plus", "a", "c", "e"]

    # Operators couples + and -, * and / are mutually associative:
    # (i.e. expression gets flattened when mixing these operators)
    assert chain("a*b/c") == ["Times", "a", "b", ["Power", "c", "-1"]]
    assert chain("a/b*c") == ["Times", "a", ["Power", "b", "-1"], "c"]
    assert chain("a+b-c") == ["Plus", "a", "b", ["Times", "-1", "c"]]
    assert chain("a-b+c") == ["Plus", "a", ["Times", "-1", "b"], "c"]
    assert chain("-a + b -c ") == ["Plus", ["Times", "-1", "a"], "b", ["Times", "-1", "c"]]
    assert chain("a/b/c*d") == ["Times", "a", ["Power", "b", "-1"], ["Power", "c", "-1"], "d"]
    assert chain("a/b/c") == ["Times", "a", ["Power", "b", "-1"], ["Power", "c", "-1"]]
    assert chain("a-b-c") == ["Plus", "a", ["Times", "-1", "b"], ["Times", "-1", "c"]]
    assert chain("1/a") == ["Times", "1", ["Power", "a", "-1"]]
    assert chain("1/a/b") == ["Times", "1", ["Power", "a", "-1"], ["Power", "b", "-1"]]
    assert chain("-1/a*b") == ["Times", "-1", ["Power", "a", "-1"], "b"]

    # Enclosures of various kinds, i.e. ( )  [ ]  [[ ]]  { }
    assert chain("(a + b) + c") == ["Plus", ["Plus", "a", "b"], "c"]
    assert chain(" a + (b + c) + d ") == ["Plus", "a", ["Plus", "b", "c"], "d"]
    assert chain("a * (b + c)") == ["Times", "a", ["Plus", "b", "c"]]
    assert chain("a b (c d)") == ["Times", "a", "b", ["Times", "c", "d"]]
    assert chain("{a, b, 2, c}") == ["List", "a", "b", "2", "c"]
    assert chain("{a, {b, c}}") == ["List", "a", ["List", "b", "c"]]
    assert chain("{{a}}") == ["List", ["List", "a"]]
    assert chain("a[b, c]") == ["a", "b", "c"]
    assert chain("a[[b, c]]") == ["Part", "a", "b", "c"]
    assert chain("a[b[c]]") == ["a", ["b", "c"]]
    assert chain("a[[b, c[[d, {e,f}]]]]") == ["Part", "a", "b", ["Part", "c", "d", ["List", "e", "f"]]]
    assert chain("a[b[[c,d]]]") == ["a", ["Part", "b", "c", "d"]]
    assert chain("a[[b[c]]]") == ["Part", "a", ["b", "c"]]
    assert chain("a[[b[[c]]]]") == ["Part", "a", ["Part", "b", "c"]]
    assert chain("a[[b[c[[d]]]]]") == ["Part", "a", ["b", ["Part", "c", "d"]]]
    assert chain("a[b[[c[d]]]]") == ["a", ["Part", "b", ["c", "d"]]]
    assert chain("x[[a+1, b+2, c+3]]") == ["Part", "x", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    assert chain("x[a+1, b+2, c+3]") == ["x", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]
    assert chain("{a+1, b+2, c+3}") == ["List", ["Plus", "a", "1"], ["Plus", "b", "2"], ["Plus", "c", "3"]]

    # Flat operator:
    assert chain("a*b*c*d*e") == ["Times", "a", "b", "c", "d", "e"]
    assert chain("a +b + c+ d+e") == ["Plus", "a", "b", "c", "d", "e"]

    # Right priority operator:
    assert chain("a^b") == ["Power", "a", "b"]
    assert chain("a^b^c") == ["Power", "a", ["Power", "b", "c"]]
    assert chain("a^b^c^d") == ["Power", "a", ["Power", "b", ["Power", "c", "d"]]]

    # Left priority operator:
    assert chain("a/.b") == ["ReplaceAll", "a", "b"]
    assert chain("a/.b/.c/.d") == ["ReplaceAll", ["ReplaceAll", ["ReplaceAll", "a", "b"], "c"], "d"]

    assert chain("a//b") == ["a", "b"]
    assert chain("a//b//c") == [["a", "b"], "c"]
    assert chain("a//b//c//d") == [[["a", "b"], "c"], "d"]

    # Compound expressions
    assert chain("a;b") == ["CompoundExpression", "a", "b"]
    assert chain("a;") == ["CompoundExpression", "a", "Null"]
    assert chain("a;b;") == ["CompoundExpression", "a", "b", "Null"]
    assert chain("a[b;c]") == ["a", ["CompoundExpression", "b", "c"]]
    assert chain("a[b,c;d,e]") == ["a", "b", ["CompoundExpression", "c", "d"], "e"]
    assert chain("a[b,c;,d]") == ["a", "b", ["CompoundExpression", "c", "Null"], "d"]

    # New lines
    assert chain("a\nb\n") == ["CompoundExpression", "a", "b"]
    assert chain("a\n\nb\n (c \nd)  \n") == ["CompoundExpression", "a", "b", ["Times", "c", "d"]]
    assert chain("\na; b\nc") == ["CompoundExpression", "a", "b", "c"]
    assert chain("a + \nb\n") == ["Plus", "a", "b"]
    assert chain("a\nb; c; d\n e; (f \n g); h + \n i") == ["CompoundExpression", "a", "b", "c", "d", "e", ["Times", "f", "g"], ["Plus", "h", "i"]]
    assert chain("\n{\na\nb; c; d\n e (f \n g); h + \n i\n\n}\n") == ["List", ["CompoundExpression", ["Times", "a", "b"], "c", ["Times", "d", "e", ["Times", "f", "g"]], ["Plus", "h", "i"]]]

    # Patterns
    assert chain("y_") == ["Pattern", "y", ["Blank"]]
    assert chain("y_.") == ["Optional", ["Pattern", "y", ["Blank"]]]
    assert chain("y__") == ["Pattern", "y", ["BlankSequence"]]
    assert chain("y___") == ["Pattern", "y", ["BlankNullSequence"]]
    assert chain("a[b_.,c_]") == ["a", ["Optional", ["Pattern", "b", ["Blank"]]], ["Pattern", "c", ["Blank"]]]
    assert chain("b_. c") == ["Times", ["Optional", ["Pattern", "b", ["Blank"]]], "c"]

    # Slots for lambda functions
    assert chain("#") == ["Slot", "1"]
    assert chain("#3") == ["Slot", "3"]
    assert chain("#n") == ["Slot", "n"]
    assert chain("##") == ["SlotSequence", "1"]
    assert chain("##a") == ["SlotSequence", "a"]

    # Lambda functions
    assert chain("x&") == ["Function", "x"]
    assert chain("#&") == ["Function", ["Slot", "1"]]
    assert chain("#+3&") == ["Function", ["Plus", ["Slot", "1"], "3"]]
    assert chain("#1 + #2&") == ["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]]
    assert chain("# + #&") == ["Function", ["Plus", ["Slot", "1"], ["Slot", "1"]]]
    assert chain("#&[x]") == [["Function", ["Slot", "1"]], "x"]
    assert chain("#1 + #2 & [x, y]") == [["Function", ["Plus", ["Slot", "1"], ["Slot", "2"]]], "x", "y"]
    assert chain("#1^2#2^3&") == ["Function", ["Times", ["Power", ["Slot", "1"], "2"], ["Power", ["Slot", "2"], "3"]]]

    # Strings inside Mathematica expressions:
    assert chain('"abc"') == ["_Str", "abc"]
    assert chain('"a\\"b"') == ["_Str", 'a"b']
    # This expression does not make sense mathematically, it's just testing the parser:
    assert chain('x + "abc" ^ 3') == ["Plus", "x", ["Power", ["_Str", "abc"], "3"]]
    assert chain('"a (* b *) c"') == ["_Str", "a (* b *) c"]
    assert chain('"a" (* b *) ') == ["_Str", "a"]
    assert chain('"a [ b] "') == ["_Str", "a [ b] "]
    raises(SyntaxError, lambda: chain('"'))
    raises(SyntaxError, lambda: chain('"\\"'))
    raises(SyntaxError, lambda: chain('"abc'))
    raises(SyntaxError, lambda: chain('"abc\\"def'))

    # Invalid expressions:
    raises(SyntaxError, lambda: chain("(,"))
    raises(SyntaxError, lambda: chain("()"))
    raises(SyntaxError, lambda: chain("a (* b"))


def test_parser_mathematica_exp_alt():
    parser = MathematicaParser()

    convert_chain2 = lambda expr: parser._from_fullformlist_to_fullformsympy(parser._from_fullform_to_fullformlist(expr))
    convert_chain3 = lambda expr: parser._from_fullformsympy_to_sympy(convert_chain2(expr))

    Sin, Times, Plus, Power = symbols("Sin Times Plus Power", cls=Function)

    full_form1 = "Sin[Times[x, y]]"
    full_form2 = "Plus[Times[x, y], z]"
    full_form3 = "Sin[Times[x, Plus[y, z], Power[w, n]]]]"
    full_form4 = "Rational[Rational[x, y], z]"

    assert parser._from_fullform_to_fullformlist(full_form1) == ["Sin", ["Times", "x", "y"]]
    assert parser._from_fullform_to_fullformlist(full_form2) == ["Plus", ["Times", "x", "y"], "z"]
    assert parser._from_fullform_to_fullformlist(full_form3) == ["Sin", ["Times", "x", ["Plus", "y", "z"], ["Power", "w", "n"]]]
    assert parser._from_fullform_to_fullformlist(full_form4) == ["Rational", ["Rational", "x", "y"], "z"]

    assert convert_chain2(full_form1) == Sin(Times(x, y))
    assert convert_chain2(full_form2) == Plus(Times(x, y), z)
    assert convert_chain2(full_form3) == Sin(Times(x, Plus(y, z), Power(w, n)))

    assert convert_chain3(full_form1) == sin(x*y)
    assert convert_chain3(full_form2) == x*y + z
    assert convert_chain3(full_form3) == sin(x*(y + z)*w**n)
