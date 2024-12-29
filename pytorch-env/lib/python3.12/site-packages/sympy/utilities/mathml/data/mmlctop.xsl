<?xml version="1.0"?>

<!-- ******************************************************** -->
<!--  XSL Transform of MathML content to MathML presentation  -->
<!--             Version 1.0 RC2 from 13-Jun-2003             -->
<!--                                                          -->
<!--    Complies with the W3C MathML 2.0 Recommenation of     -->
<!--                    21 February 2001.                     -->
<!--                                                          -->
<!--   Authors Igor Rodionov <igor@csd.uwo.ca>,               -->
<!--           Stephen Watt  <watt@csd.uwo.ca>.               -->
<!--                                                          -->
<!-- (C) Copyright 2000-2003 Symbolic Computation Laboratory, -->
<!--                         University of Western Ontario,   -->
<!--                         London, Canada N6A 5B7.          -->
<!--                                                          -->
<!-- Modified: Fabian Seoane <fabian@fseoane> 2007 for sympy  -->
<!-- ******************************************************** -->

<xsl:stylesheet id="mmlctop2.xsl"
                version="1.0"
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:mml="http://www.w3.org/1998/Math/MathML"
                xmlns="http://www.w3.org/1998/Math/MathML">

<xsl:output method="xml" indent="yes"/>

<xsl:strip-space elements="apply semantics annotation-xml
        csymbol fn cn ci interval matrix matrixrow vector
        lambda bvar condition logbase degree set list
        lowlimit uplimit math"/>


<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
<!--         Parameters, variables and constants           -->
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->

<!-- ~~~~~~~~ Semantics related *constants*: ~~~~~~~~ -->

<!-- Strip off semantics -->
<xsl:variable name="SEM_STRIP" select="-1"/>

<!-- Pass semantics "as is" -->
<xsl:variable name="SEM_PASS" select="0"/>

<!-- Add semantics at top level only -->
<xsl:variable name="SEM_TOP" select="1"/>

<!-- Add semantics at all levels -->
<xsl:variable name="SEM_ALL" select="2"/>

<!-- Semantics at top level only, with id refs -->
<!-- NOTE: ids have to be already present in the
           input for this feature to work. -->
<xsl:variable name="SEM_XREF" select="3"/>

<!-- No semantics at top level, with id refs -->
<!-- NOTE: ids have to be already present in the
           input for this feature to work. -->
<xsl:variable name="SEM_XREF_EXT" select="4"/>


<!-- ~~~~~~~~~~ Stylesheet *parameter*: SEM_SW ~~~~~~~~~~~~~~ -->
<!-- Assumes one of the above values; SEM_PASS is the default -->
<!-- The default can be overridden by specifying different    -->
<!-- value on the command line when the stylesheet is invoked -->

<xsl:param name="SEM_SW" select="SEM_PASS"/>


<!-- ~~~~~~ Operator precedence definitions ~~~~~~ -->

<xsl:variable name="NO_PREC" select="0"/>
<xsl:variable name="UNION_PREC" select="1"/>
<xsl:variable name="SETDIFF_PREC" select="1"/>
<xsl:variable name="INTERSECT_PREC" select="3"/>
<xsl:variable name="CARTPROD_PREC" select="3"/>
<xsl:variable name="OR_PREC" select="5"/>
<xsl:variable name="XOR_PREC" select="7"/>
<xsl:variable name="AND_PREC" select="9"/>
<xsl:variable name="NOT_PREC" select="11"/>
<xsl:variable name="PLUS_PREC" select="13"/>
<xsl:variable name="MINUS_PREC" select="13"/>
<xsl:variable name="NEG_PREC" select="15"/>
<xsl:variable name="MUL_PREC" select="17"/>
<xsl:variable name="DIV_PREC" select="17"/>
<xsl:variable name="REM_PREC" select="17"/>
<xsl:variable name="FUNCTN_PREC" select="97"/>
<xsl:variable name="GEN_FUN_PREC" select="99"/>

<!-- ~~~~~ Miscellaneous constant definitions ~~~~~ -->

<xsl:variable name="YES" select="1"/>
<xsl:variable name="NO" select="0"/>
<xsl:variable name="NO_PARAM" select="-1"/>
<xsl:variable name="PAR_SAME" select="-3"/>
<xsl:variable name="PAR_YES" select="-5"/>
<xsl:variable name="PAR_NO" select="-7"/>



<!-- +++++++++++++++++ INDEX OF TEMPLATES +++++++++++++++++++ -->

<!-- All templates are subdivided into the following categories
     (listed in the order of appearance in the stylesheet):

THE TOPMOST ELEMENT: MATH
 math

SEMANTICS HANDLING
 semantics

BASIC CONTAINER ELEMENTS
 cn, ci; csymbol

BASIC CONTENT ELEMENTS
 fn, interval, inverse, sep, condition, declare, lambda, compose,
 ident; domain, codomain, image, domainofapplication, piecewise

ARITHMETIC, ALGEBRA & LOGIC
 quotient, exp, factorial, max, min, minus, plus, power, rem, divide,
 times, root, gcd, and, or, xor, not, forall, exists, abs, conjugate;
 arg, real, imaginary, lcm, floor, ceiling

RELATIONS
 neq, approx, tendsto, implies, in, notin, notsubset, notprsubset,
 subset, prsubset, eq, gt, lt, geq, leq; equivalent, factorof

CALCULUS
 ln, log, diff, partialdiff, lowlimit, uplimit, bvar, degree,
 logbase; divergence, grad, curl, laplacian

SET THEORY
 set, list, union, intersect, setdiff; card, cartesianproduct

SEQUENCES AND SERIES
 sum, product, limit

TRIGONOMETRY
 sin, cos, tan, sec, csc, cot, sinh, cosh, tanh, sech, csch, coth,
 arcsin, arccos, arctan, arcsec, arccsc, arccot, arcsinh, arccosh,
 arctanh, arcsech, arccsch, arccoth

STATISTICS
 mean, sdev, variance, median, mode, moment, momentabout

LINEAR ALGEBRA
 vector, matrix, matrixrow, determinant, transpose, selector;
 vectorproduct, scalarproduct, outerproduct

CONSTANT and SYMBOL ELEMENTS
 integers, reals, rationals, naturalnumbers, complexes, primes,
 exponentiale, imaginaryi, notanumber, true, false, emptyset,
 pi, eulergamma, infinity
-->



<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~ TEMPLATES ~~~~~~~~~~~~~~~~~~~~~~~~~ -->
<!-- ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ -->



<!-- *********************** COPY THROUGH ************************ -->

<xsl:template match = "*">
  <xsl:copy>
    <xsl:copy-of select="@*"/>
    <xsl:apply-templates/>
  </xsl:copy>
</xsl:template>



<!-- ***************** THE TOPMOST ELEMENT: MATH ***************** -->

<xsl:template match = "math">
  <math>
    <xsl:copy-of select="@*"/>
    <xsl:choose>
      <xsl:when test="$SEM_SW=$SEM_TOP or $SEM_SW=$SEM_ALL and *[2] or
	                  $SEM_SW=$SEM_XREF">
        <semantics>
          <mrow>
            <xsl:apply-templates mode = "semantics"/>
          </mrow>
          <annotation-xml encoding="MathML">
            <xsl:copy-of select="*"/>
          </annotation-xml>
        </semantics>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates mode = "semantics"/>
      </xsl:otherwise>
    </xsl:choose>
  </math>
</xsl:template>



<!-- ***************** SEMANTICS HANDLING ***************** -->

<!-- This template is called recursively.  At each level   -->
<!-- in the source tree it decides whether to strip off,   -->
<!-- pass or add semantics at that level (depending on the -->
<!-- value of SEM_SW parameter).  Then the actual template -->
<!-- is applied to the node.                               -->

<xsl:template match = "*" mode = "semantics">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$SEM_SW=$SEM_STRIP and self::semantics">
      <xsl:apply-templates select="annotation-xml[@encoding='MathML']">
        <xsl:with-param name="IN_PREC" select="$IN_PREC"/>
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:when test="($SEM_SW=$SEM_PASS or $SEM_SW=$SEM_TOP) and self::semantics">
      <semantics>
        <xsl:apply-templates select="*[1]">
        <xsl:with-param name="IN_PREC" select="$IN_PREC"/>
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
        <xsl:copy-of select="annotation-xml"/>
      </semantics>
    </xsl:when>
    <xsl:when test="$SEM_SW=$SEM_ALL">
      <semantics>
        <xsl:choose>
          <xsl:when test="self::semantics">
            <xsl:apply-templates select="*[1]">
              <xsl:with-param name="IN_PREC" select="$IN_PREC"/>
              <xsl:with-param name="PARAM" select="$PARAM"/>
              <xsl:with-param name="PAREN" select="$PAREN"/>
              <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
            </xsl:apply-templates>
            <xsl:copy-of select="annotation-xml"/>
          </xsl:when>
          <xsl:otherwise>
            <xsl:apply-templates select=".">
              <xsl:with-param name="IN_PREC" select="$IN_PREC"/>
              <xsl:with-param name="PARAM" select="$PARAM"/>
              <xsl:with-param name="PAREN" select="$PAREN"/>
              <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
            </xsl:apply-templates>
            <annotation-xml encoding="MathML">
              <xsl:copy-of select="."/>
            </annotation-xml>
          </xsl:otherwise>
        </xsl:choose>
      </semantics>
    </xsl:when>
    <xsl:when test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:choose>
        <xsl:when test="self::semantics">
          <xsl:copy>
            <xsl:copy-of select="@*"/>
            <xsl:attribute name="xref">
              <xsl:value-of select="@id"/>
            </xsl:attribute>
            <xsl:copy-of select="*[1]"/>
            <xsl:copy-of select="annotation-xml"/>
          </xsl:copy>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates select=".">
            <xsl:with-param name="IN_PREC" select="$IN_PREC"/>
            <xsl:with-param name="PARAM" select="$PARAM"/>
            <xsl:with-param name="PAREN" select="$PAREN"/>
            <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
          </xsl:apply-templates>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:when>
    <xsl:otherwise>
      <xsl:apply-templates select=".">
        <xsl:with-param name="IN_PREC" select="$IN_PREC"/>
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match = "semantics">
  <xsl:apply-templates select="*[1]" mode = "semantics"/>
</xsl:template>



<!-- ***************** BASIC CONTAINER ELEMENTS ***************** -->

<xsl:template match = "cn">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test=". &lt; 0 and $IN_PREC &gt; $NO_PREC and $PAREN=$PAR_NO
                                                   and $PAR_NO_IGNORE=$NO">
      <mfenced separators="">
        <xsl:apply-templates select="." mode="cn"/>
      </mfenced>
    </xsl:when>
    <xsl:otherwise>
      <xsl:apply-templates select="." mode="cn"/>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "cn" mode="cn">
  <xsl:choose>
    <xsl:when test="(not(@type) or @type='integer' or @type='real') and @base">
      <msub>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mn> <xsl:apply-templates mode = "semantics"/> </mn>
        <mn> <xsl:value-of select="@base"/> </mn>
      </msub>
    </xsl:when>
    <xsl:when test="not(@type) or @type='integer' or @type='real'">
      <mn>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates mode = "semantics"/>
      </mn>
    </xsl:when>
    <xsl:when test="@type='constant'">
      <mn>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates mode = "semantics"/>
      </mn>
    </xsl:when>
    <xsl:when test="@type='e-notation' and not(@base) and child::sep[1]">
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mn> <xsl:apply-templates select="text()[1]" mode = "semantics"/> </mn>
        <mo> e </mo>
        <mn> <xsl:apply-templates select="text()[2]" mode = "semantics"/> </mn>
      </mrow>
    </xsl:when>
    <xsl:when test="@type='complex-cartesian' and not(@base) and child::sep[1]">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mn> <xsl:apply-templates select="text()[1]" mode = "semantics"/> </mn>
        <xsl:if test="text()[2] &lt; 0">
          <mo> - </mo>
          <mn> <xsl:value-of select="-text()[2]"/> </mn>
        </xsl:if>
        <xsl:if test="not(text()[2] &lt; 0)">
          <mo> + </mo>
          <mn> <xsl:value-of select="text()[2]"/> </mn>
        </xsl:if>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2062;</xsl:text> </mo>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2148;</xsl:text> </mo>
      </mfenced>
    </xsl:when>
    <xsl:when test="@type='complex-cartesian' and @base and child::sep[1]">
      <msub>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mfenced separators="">
          <mn> <xsl:apply-templates select="text()[1]"/> </mn>
          <xsl:if test="text()[2] &lt; 0">
            <mo> - </mo>
            <mn> <xsl:value-of select="-text()[2]"/> </mn>
          </xsl:if>
          <xsl:if test="not(text()[2] &lt; 0)">
            <mo> + </mo>
            <mn> <xsl:apply-templates select="text()[2]"/> </mn>
          </xsl:if>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2062;</xsl:text> </mo>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2148;</xsl:text> </mo>
        </mfenced>
        <mn> <xsl:value-of select="@base"/> </mn>
      </msub>
    </xsl:when>
    <xsl:when test="@type='rational' and not(@base) and child::sep[1]">
      <mfrac>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mn> <xsl:apply-templates select="text()[1]"/> </mn>
        <mn> <xsl:apply-templates select="text()[2]"/> </mn>
      </mfrac>
    </xsl:when>
    <xsl:when test="@type='rational' and @base and child::sep[1]">
      <msub>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mfenced>
          <mfrac>
            <mn> <xsl:apply-templates select="text()[1]"/> </mn>
            <mn> <xsl:apply-templates select="text()[2]"/> </mn>
          </mfrac>
        </mfenced>
        <mn> <xsl:value-of select="@base"/> </mn>
      </msub>
    </xsl:when>
    <xsl:when test="@type='complex-polar' and not(@base) and child::sep[1]">
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mo> Polar </mo>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2062;</xsl:text> </mo>
        <mfenced separators=",">
          <mn> <xsl:apply-templates select="text()[1]"/> </mn>
          <mn> <xsl:apply-templates select="text()[2]"/> </mn>
        </mfenced>
      </mrow>
    </xsl:when>
    <xsl:when test="@type='complex-polar' and @base and child::sep[1]">
      <msub>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mrow>
          <mo> Polar </mo>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2062;</xsl:text> </mo>
          <mfenced separators=",">
            <mn> <xsl:apply-templates select="text()[1]"/> </mn>
            <mn> <xsl:apply-templates select="text()[2]"/> </mn>
          </mfenced>
        </mrow>
        <mn> <xsl:value-of select="@base"/> </mn>
      </msub>
   </xsl:when>
    <xsl:otherwise>
      <mn> <xsl:apply-templates mode = "semantics"/> </mn>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match = "ci">
  <xsl:choose>
    <xsl:when test="@type='vector' or @type='matrix' or @type='set'">
      <mi mathvariant="bold">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates mode = "semantics"/>
      </mi>
    </xsl:when>
    <xsl:when test="child::text() and not(child::*[1])">
      <mi>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates/>
      </mi>
    </xsl:when>
    <xsl:when test="child::text() and *[1] and not(*[1]=sep)">
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates/>
      </mrow>
    </xsl:when>
    <xsl:otherwise>
      <xsl:if test="*[2]">
        <mrow>
          <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
            <xsl:attribute name="xref">
              <xsl:value-of select="@id"/>
            </xsl:attribute>
          </xsl:if>
          <xsl:apply-templates select="*"/>
        </mrow>
      </xsl:if>
      <xsl:if test="not(*[2])">
        <xsl:apply-templates select="*[1]"/>
      </xsl:if>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "ci/*[not(self::sep)]">
  <xsl:copy-of select = "."/>
</xsl:template>


<xsl:template match = "csymbol">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:copy-of select = "* | text()"/>
  </mrow>
</xsl:template>



<!-- ***************** BASIC CONTENT ELEMENTS ***************** -->

<!-- General <apply> <AnyFunction/> ... </apply> -->
<!-- Dependants: csymbol apply[fn inverse compose] -->
<xsl:template match = "apply">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select = "*[1]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
      <xsl:with-param name="PARAM" select="$PAR_SAME"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <mfenced separators=",">
      <xsl:apply-templates select = "*[position()>1]" mode = "semantics">
        <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
        <xsl:with-param name="PARAM" select="$PAR_SAME"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
      </xsl:apply-templates>
    </mfenced>
 </mrow>
</xsl:template>


<!-- fn is ***DEPRECATED*** -->
<xsl:template match = "fn">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
    <xsl:with-param name="PARAM" select="$PAR_SAME"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
  </xsl:apply-templates>
</xsl:template>


<xsl:template match = "interval">
  <mfenced>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:if test="not(@closure) or @closure='closed' or @closure='closed-open' or not(@closure='open') and not(@closure='open-closed')">
      <xsl:attribute name="open"> [ </xsl:attribute>
    </xsl:if>
    <xsl:if test="not(@closure) or @closure='closed' or @closure='open-closed' or not(@closure='open') and not(@closure='closed-open')">
      <xsl:attribute name="close"> ] </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*" mode = "semantics"/>
  </mfenced>
</xsl:template>


<xsl:template match = "apply[*[1][self::inverse]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="*[2]=exp | *[2]=ln | *[2]=sin | *[2]=cos |
                    *[2]=tan | *[2]=sec | *[2]=csc | *[2]=cot |
                    *[2]=sinh | *[2]=cosh | *[2]=tanh | *[2]=sech |
                    *[2]=csch | *[2]=coth | *[2]=arcsin |
                    *[2]=arccos | *[2]=arctan">
      <mo>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="*[2]" mode="inverse"/>
      </mo>
    </xsl:when>
    <xsl:otherwise>
      <msup>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
		<mrow>
          <xsl:apply-templates select = "*[2]">
            <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
            <xsl:with-param name="PARAM" select="$PAR_SAME"/>
            <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
          </xsl:apply-templates>
		</mrow>
		<mfenced>
          <mn> -1 </mn>
        </mfenced>
      </msup>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "*" mode="inverse">
  <xsl:choose>
    <xsl:when test="self::exp">
      <xsl:value-of select="'ln'"/>
    </xsl:when>
    <xsl:when test="self::ln">
      <xsl:value-of select="'exp'"/>
    </xsl:when>
    <xsl:when test="self::sin">
      <xsl:value-of select="'arcsin'"/>
    </xsl:when>
    <xsl:when test="self::cos">
      <xsl:value-of select="'arccos'"/>
    </xsl:when>
    <xsl:when test="self::tan">
      <xsl:value-of select="'arctan'"/>
    </xsl:when>
    <xsl:when test="self::sec">
      <xsl:value-of select="'arcsec'"/>
    </xsl:when>
    <xsl:when test="self::csc">
      <xsl:value-of select="'arccsc'"/>
    </xsl:when>
    <xsl:when test="self::cot">
      <xsl:value-of select="'arccot'"/>
    </xsl:when>
    <xsl:when test="self::sinh">
      <xsl:value-of select="'arcsinh'"/>
    </xsl:when>
    <xsl:when test="self::cosh">
      <xsl:value-of select="'arccosh'"/>
    </xsl:when>
    <xsl:when test="self::tanh">
      <xsl:value-of select="'arctanh'"/>
    </xsl:when>
    <xsl:when test="self::sech">
      <xsl:value-of select="'arcsech'"/>
    </xsl:when>
    <xsl:when test="self::csch">
      <xsl:value-of select="'arccsch'"/>
    </xsl:when>
    <xsl:when test="self::coth">
      <xsl:value-of select="'arccoth'"/>
    </xsl:when>
    <xsl:when test="self::arcsin">
      <xsl:value-of select="'sin'"/>
    </xsl:when>
    <xsl:when test="self::arccos">
      <xsl:value-of select="'cos'"/>
    </xsl:when>
    <xsl:when test="self::arctan">
      <xsl:value-of select="'tan'"/>
    </xsl:when>
  </xsl:choose>
</xsl:template>


<xsl:template match = "condition">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*" mode = "semantics"/>
  </mrow>
</xsl:template>


<xsl:template match = "declare"/>


<xsl:template match = "lambda">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x03BB;</xsl:text> </mo>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <mfenced>
      <xsl:for-each select = "*">
        <xsl:choose>
          <xsl:when test="self::ci or self::cn">
            <xsl:apply-templates select = "." mode="semantics"/>
          </xsl:when>
          <xsl:otherwise>
            <mrow>
              <xsl:apply-templates select = "." mode="semantics"/>
            </mrow>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:for-each>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[*[1][self::compose]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $FUNCTN_PREC or $IN_PREC=$FUNCTN_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select = "*[2]" mode="semantics"/>
        <xsl:for-each select = "*[position()>2]">
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2218;</xsl:text> </mo>
          <xsl:apply-templates select = "." mode="semantics">
            <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
            <xsl:with-param name="PARAM" select="$PAR_SAME"/>
            <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
          </xsl:apply-templates>
        </xsl:for-each>
      </mfenced>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select = "*[2]" mode="semantics"/>
        <xsl:for-each select = "*[position()>2]">
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2218;</xsl:text> </mo>
          <xsl:apply-templates select = "." mode="semantics">
            <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
            <xsl:with-param name="PARAM" select="$PAR_SAME"/>
            <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
          </xsl:apply-templates>
        </xsl:for-each>
      </mrow>
	</xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match = "ident">
  <xsl:choose>
    <xsl:when test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <mtext xref="{@id}">id</mtext>
    </xsl:when>
    <xsl:otherwise>
      <mtext>id</mtext>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match="apply[*[1]=domain or *[1]=codomain or *[1]=image]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
	<xsl:if test="*[1]=domain">
      <mtext>domain</mtext>
	</xsl:if>
	<xsl:if test="*[1]=codomain">
      <mtext>codomain</mtext>
	</xsl:if>
	<xsl:if test="*[1]=image">
      <mtext>image</mtext>
	</xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <mfenced separators="">
      <xsl:apply-templates select="*[position()>1]" mode = "semantics"/>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "domainofapplication">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select = "*" mode = "semantics"/>
  </mrow>
</xsl:template>


<xsl:template match="piecewise">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo stretchy="true"> { </mo>
    <mtable columnalign="left left">
      <xsl:for-each select="piece">
        <mtr>
          <mtd>
            <xsl:apply-templates select="*[position()=1]" mode = "semantics"/>
          </mtd>
          <mtd>
            <mtext>if </mtext>
            <xsl:apply-templates select="*[position()=2]" mode = "semantics"/>
          </mtd>
        </mtr>
      </xsl:for-each>
      <xsl:if test="otherwise">
        <mtr>
          <mtd>
            <xsl:apply-templates select="otherwise/*" mode = "semantics"/>
          </mtd>
          <mtd>
            <mtext>otherwise</mtext>
          </mtd>
        </mtr>
      </xsl:if>
    </mtable>
  </mrow>
</xsl:template>



<!-- ***************** ARITHMETIC, ALGEBRA & LOGIC ***************** -->

<xsl:template match = "apply[quotient[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x230A;</xsl:text> </mo>
    <mfrac>
      <mrow>
        <xsl:apply-templates select="*[2]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$DIV_PREC"/>
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
      <mrow>
        <xsl:apply-templates select="*[3]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$DIV_PREC"/>
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </mfrac>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x230B;</xsl:text> </mo>
  </mrow>
</xsl:template>


<xsl:template match = "apply[*[1][self::exp]]">
  <msup>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mn> <xsl:text disable-output-escaping='yes'>&amp;#x2147;</xsl:text> </mn>
    <xsl:apply-templates select = "*[2]" mode = "semantics"/>
  </msup>
</xsl:template>


<xsl:template match = "apply[factorial[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select = "*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
    <mo> ! </mo>
  </mrow>
</xsl:template>


<xsl:template match = "apply[max[1] | min[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="*[2]=bvar">
        <munder>
          <xsl:if test="*[1]=max">
            <mo> max </mo>
          </xsl:if>
          <xsl:if test="*[1]=min">
            <mo> min </mo>
          </xsl:if>
          <xsl:apply-templates select="*[2]" mode = "semantics"/>
        </munder>
      </xsl:when>
      <xsl:otherwise>
        <xsl:if test="*[1]=max">
          <mo> max </mo>
        </xsl:if>
        <xsl:if test="*[1]=min">
          <mo> min </mo>
        </xsl:if>
      </xsl:otherwise>
	</xsl:choose>
    <mfenced open="{{" close="}}">
      <xsl:if test="child::condition">
        <xsl:attribute name="separators"/>
        <xsl:if test="*[position()>1 and not(self::bvar) and not(self::condition)]">
          <mfenced open="" close="" separators=",">
            <xsl:for-each select = "*[position()>1 and not(self::bvar) and not(self::condition)]">
              <xsl:apply-templates select = "." mode="semantics"/>
            </xsl:for-each>
          </mfenced>
          <mo lspace="0.1666em" rspace="0.1666em"> | </mo>
        </xsl:if>
        <xsl:apply-templates select="condition" mode = "semantics"/>
      </xsl:if>
      <xsl:if test="not(child::condition)">
        <xsl:for-each select = "*[position()>1 and not(self::bvar)]">
          <xsl:apply-templates select = "." mode="semantics"/>
        </xsl:for-each>
      </xsl:if>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[minus[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $MINUS_PREC or $IN_PREC=$MINUS_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="minus">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="minus">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="minus">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[minus[1]]" mode="minus">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:if test="not(*[3])">
    <mo> - </mo>
    <xsl:apply-templates select="*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$NEG_PREC"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:if>
  <xsl:if test="*[3]">
    <xsl:apply-templates select="*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$MINUS_PREC"/>
      <xsl:with-param name="PARAM" select="$PARAM"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
    </xsl:apply-templates>
    <mo> - </mo>
    <xsl:apply-templates select="*[3]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$MINUS_PREC"/>
      <xsl:with-param name="PARAM" select="$PAR_SAME"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:if>
</xsl:template>


<xsl:template match = "apply[plus[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $PLUS_PREC or $IN_PREC=$PLUS_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="plus">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="plus">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="plus">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[plus[1]]" mode="plus">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:if test="*[2]">
    <xsl:apply-templates select="*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$PLUS_PREC"/>
      <xsl:with-param name="PARAM" select="$PARAM"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
    </xsl:apply-templates>
    <xsl:for-each select = "*[position()>2]">
      <xsl:choose>
        <xsl:when test=". &lt; 0">
          <mo> - </mo>
          <mn> <xsl:value-of select="-."/> </mn>
        </xsl:when>
        <xsl:when test="self::apply[minus[1]] and not(*[3])">
          <xsl:apply-templates select="." mode = "semantics">
            <xsl:with-param name="IN_PREC" select="$PLUS_PREC"/>
            <xsl:with-param name="PAREN" select="$PAREN"/>
            <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
          </xsl:apply-templates>
        </xsl:when>
        <xsl:otherwise>
          <mo> + </mo>
          <xsl:apply-templates select="." mode = "semantics">
            <xsl:with-param name="IN_PREC" select="$PLUS_PREC"/>
            <xsl:with-param name="PAREN" select="$PAREN"/>
            <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
          </xsl:apply-templates>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:for-each>
  </xsl:if>
</xsl:template>


<xsl:template match = "apply[*[1][self::power]]">
  <xsl:choose>
    <xsl:when test="*[2]=apply[ln[1] | log[1] | abs[1] |
                         gcd[1] | lcm[1] | sin[1] | cos[1] | tan[1] |
                         sec[1] | csc[1] | cot[1] | sinh[1] |
                         cosh[1] | tanh[1] | sech[1] | csch[1] |
                         coth[1] | arcsin[1] | arccos[1] |
                         arctan[1] | arcsec[1] | arccsc[1] |
                         arccot[1] | arcsinh[1] | arccosh[1] |
                         arctanh[1] | arcsech[1] | arccsch[1] |
                         arccoth[1]]">
      <xsl:apply-templates select="*[2]" mode = "semantics"/>
    </xsl:when>
    <xsl:otherwise>
      <msup>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select = "*[2]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
        </xsl:apply-templates>
        <xsl:apply-templates select = "*[3]" mode = "semantics"/>
      </msup>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match = "apply[divide[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $DIV_PREC or $IN_PREC=$DIV_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="div">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="div">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="div">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[divide[1]]" mode="div">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <mfrac>
    <mrow>
      <xsl:apply-templates select = "*[2]" mode = "semantics">
        <xsl:with-param name="IN_PREC" select="$GEN_FUN_PREC"/>
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </mrow>
    <mrow>
      <xsl:apply-templates select = "*[3]" mode = "semantics">
        <xsl:with-param name="IN_PREC" select="$GEN_FUN_PREC"/>
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </mrow>
  </mfrac>
</xsl:template>


<xsl:template match = "apply[rem[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $REM_PREC or $IN_PREC=$REM_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="rem">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="rem">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="rem">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[rem[1]]" mode="rem">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select = "*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$REM_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <mo lspace="thickmathspace" rspace="thickmathspace"> <xsl:value-of select="'mod'"/> </mo>
  <xsl:apply-templates select = "*[3]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$REM_PREC"/>
    <xsl:with-param name="PARAM" select="$PAR_SAME"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
  </xsl:apply-templates>
</xsl:template>


<xsl:template match = "apply[times[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $MUL_PREC or $IN_PREC=$MUL_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="times">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="times">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="times">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[times[1]]" mode="times">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select="*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$MUL_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:if test="*[3]">
    <xsl:for-each select = "*[position()>2]">
      <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2062;</xsl:text> </mo>
      <xsl:apply-templates select="." mode = "semantics">
        <xsl:with-param name="IN_PREC" select="$MUL_PREC"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
      </xsl:apply-templates>
    </xsl:for-each>
  </xsl:if>
</xsl:template>


<xsl:template match = "apply[*[1]=root and *[2]=degree]">
  <mroot>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*[3]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$GEN_FUN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
    <xsl:apply-templates select="*[2]" mode = "semantics"/>
  </mroot>
</xsl:template>

<xsl:template match = "apply[*[1]=root and not(*[2]=degree)]">
  <msqrt>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$GEN_FUN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </msqrt>
</xsl:template>


<xsl:template match = "apply[gcd[1] | lcm[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:if test="not(parent::apply[power[1]])">
      <xsl:if test="gcd[1]">
        <mo> gcd </mo>
      </xsl:if>
      <xsl:if test="lcm[1]">
        <mo> lcm </mo>
      </xsl:if>
    </xsl:if>
    <xsl:if test="parent::apply[power[1]]">
      <msup>
      <xsl:if test="gcd[1]">
        <mo> gcd </mo>
      </xsl:if>
      <xsl:if test="lcm[1]">
        <mo> lcm </mo>
      </xsl:if>
        <xsl:apply-templates select = "../*[3]" mode = "semantics"/>
      </msup>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <mfenced separators=",">
      <xsl:for-each select = "*[position()>1]">
        <xsl:apply-templates select = "." mode="semantics"/>
      </xsl:for-each>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[and[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $AND_PREC or $IN_PREC=$AND_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="and">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="and">
        <xsl:with-param name="PARAM" select="$IN_PREC"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="and">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[and[1]]" mode="and">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select="*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$AND_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:for-each select = "*[position()>2]">
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2227;</xsl:text> </mo>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <xsl:apply-templates select="." mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$AND_PREC"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:for-each>
</xsl:template>


<xsl:template match = "apply[or[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $OR_PREC or $IN_PREC=$OR_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="or">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="or">
        <xsl:with-param name="PARAM" select="$IN_PREC"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="or">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[or[1]]" mode="or">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select="*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$OR_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:for-each select = "*[position()>2]">
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2228;</xsl:text> </mo>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <xsl:apply-templates select="." mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$OR_PREC"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:for-each>
</xsl:template>


<xsl:template match = "apply[xor[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and
                   ($IN_PREC &gt; $XOR_PREC or $IN_PREC=$XOR_PREC and $PARAM=$PAR_SAME)">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="xor">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAR_YES"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                                                and not($SEM_SW=$SEM_ALL)">
      <xsl:apply-templates select="." mode="xor">
        <xsl:with-param name="PARAM" select="$IN_PREC"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="xor">
          <xsl:with-param name="PARAM" select="$IN_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[xor[1]]" mode="xor">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select="*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$XOR_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:for-each select = "*[position()>2]">
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x22BB;</xsl:text> </mo>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <xsl:apply-templates select="." mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$XOR_PREC"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:for-each>
</xsl:template>


<xsl:template match = "apply[not[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &lt; $GEN_FUN_PREC and $IN_PREC &gt;= $NOT_PREC">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x00AC;</xsl:text> </mo>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
        <xsl:apply-templates select = "*[2]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$NOT_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
        </xsl:apply-templates>
	  </mfenced>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x00AC;</xsl:text> </mo>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
        <xsl:apply-templates select = "*[2]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$NOT_PREC"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match = "apply[forall[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2200;</xsl:text> </mo>
    <xsl:if test="count(bvar)=1">
      <xsl:apply-templates select = "bvar" mode="semantics"/>
    </xsl:if>
    <xsl:if test="count(bvar) &gt; 1">
      <mfenced open="" close="">
        <xsl:for-each select = "bvar">
          <xsl:apply-templates select = "." mode="semantics"/>
        </xsl:for-each>
      </mfenced>
    </xsl:if>
	<xsl:if test="condition">
      <mo> : </mo>
      <xsl:apply-templates select = "condition/*" mode = "semantics"/>
    </xsl:if>
	<xsl:if test="*[position()>1 and not(self::bvar) and not(self::condition)]">
      <mo> , </mo>
      <xsl:apply-templates select = "*[position()>1 and not(self::bvar) and
                                not(self::condition)]" mode = "semantics"/>
    </xsl:if>
  </mrow>
</xsl:template>


<xsl:template match = "apply[exists[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2203;</xsl:text> </mo>
    <xsl:if test="count(bvar) &gt; 1">
      <mfenced open="" close="">
        <xsl:for-each select = "bvar">
          <xsl:apply-templates select = "." mode="semantics"/>
        </xsl:for-each>
      </mfenced>
    </xsl:if>
    <xsl:if test="count(bvar)=1">
      <xsl:apply-templates select = "bvar" mode="semantics"/>
    </xsl:if>
    <xsl:if test="condition">
      <mo> : </mo>
      <xsl:apply-templates select = "condition/*" mode = "semantics"/>
    </xsl:if>
    <xsl:if test="*[position()>1 and not(self::bvar) and not(self::condition)]">
      <mo> , </mo>
      <xsl:apply-templates select = "*[position()>1 and not(self::bvar) and
                                not(self::condition)]" mode = "semantics"/>
    </xsl:if>
  </mrow>
</xsl:template>


<xsl:template match = "apply[abs[1]]">
  <xsl:if test="not(parent::apply[power[1]])">
    <mfenced open="&#x2223;" close="&#x2223;" separators="">
      <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
        <xsl:attribute name="xref">
          <xsl:value-of select="@id"/>
        </xsl:attribute>
      </xsl:if>
      <xsl:apply-templates select = "*[position()>1]" mode = "semantics"/>
    </mfenced>
  </xsl:if>
  <xsl:if test="parent::apply[power[1]]">
    <msup>
      <mfenced open="&#x2223;" close="&#x2223;" separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select = "*[position()>1]" mode = "semantics"/>
      </mfenced>
      <xsl:apply-templates select = "../*[3]" mode = "semantics"/>
    </msup>
  </xsl:if>
</xsl:template>


<xsl:template match = "apply[conjugate[1]]">
  <mover>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mrow>
      <xsl:apply-templates select = "*[position()>1]" mode = "semantics"/>
    </mrow>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x00AF;</xsl:text> </mo>
  </mover>
</xsl:template>


<xsl:template match = "apply[arg[1] | real[1] | imaginary[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo>
      <xsl:if test="arg">
        <xsl:value-of select="'arg'"/>
      </xsl:if>
      <xsl:if test="real">
        <xsl:text disable-output-escaping='yes'>&amp;#x211C;</xsl:text>
      </xsl:if>
      <xsl:if test="imaginary">
        <xsl:text disable-output-escaping='yes'>&amp;#x2111;</xsl:text>
      </xsl:if>
    </mo>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <mfenced separators="">
      <xsl:apply-templates select = "*[2]" mode = "semantics"/>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[floor[1] or ceiling[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo>
      <xsl:if test="floor[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x230A;</xsl:text>
      </xsl:if>
      <xsl:if test="ceiling[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x2308;</xsl:text>
      </xsl:if>
    </mo>
    <xsl:apply-templates select="*[position()>1]"  mode="semantics"/>
    <mo>
      <xsl:if test="floor[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x230B;</xsl:text>
      </xsl:if>
      <xsl:if test="ceiling[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x2309;</xsl:text>
      </xsl:if>
    </mo>
  </mrow>
</xsl:template>



<!-- ***************** RELATIONS ***************** -->

<xsl:template match = "apply[neq | approx | tendsto | implies
                     | in | notin | notsubset | notprsubset
                     | subset | prsubset | eq | gt | lt
                     | geq | leq | equivalent | factorof]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="."  mode="relations"/>
  </mrow>
</xsl:template>

<!-- reln is ***DEPRECATED*** -->
<xsl:template match = "reln[neq | approx | tendsto | implies
                     | in | notin | notsubset | notprsubset
                     | subset | prsubset | eq | gt | lt
                     | geq | leq | equivalent | factorof]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="."  mode="relations"/>
  </mrow>
</xsl:template>

<xsl:template match = "*" mode="relations">
  <xsl:if test="*[1]=neq or *[1]=approx or *[1]=factorof or *[1]=tendsto or
                *[1]=implies or *[1]=in or *[1]=notin or
                *[1]=notsubset or *[1]=notprsubset">
    <xsl:apply-templates select = "*[2]" mode = "semantics"/>
    <mo>
      <xsl:if test="*[1]=neq">
        <xsl:text disable-output-escaping='yes'>&amp;#x2260;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=approx">
        <xsl:text disable-output-escaping='yes'>&amp;#x2248;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=factorof">
        <xsl:text disable-output-escaping='yes'>&amp;#x2223;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=tendsto">
        <xsl:choose>
          <xsl:when test="tendsto[@type='above']">
            <xsl:text disable-output-escaping='yes'>&amp;#x2198;</xsl:text>
          </xsl:when>
          <xsl:when test="tendsto[@type='below']">
            <xsl:text disable-output-escaping='yes'>&amp;#x2197;</xsl:text>
          </xsl:when>
		  <xsl:otherwise>
            <xsl:text disable-output-escaping='yes'>&amp;#x2192;</xsl:text>
          </xsl:otherwise>
        </xsl:choose>
      </xsl:if>
      <xsl:if test="*[1]=implies">
        <xsl:text disable-output-escaping='yes'>&amp;#x21D2;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=in">
        <xsl:text disable-output-escaping='yes'>&amp;#x2208;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=notin">
        <xsl:text disable-output-escaping='yes'>&amp;#x2209;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=notsubset">
        <xsl:text disable-output-escaping='yes'>&amp;#x2284;</xsl:text>
      </xsl:if>
      <xsl:if test="*[1]=notprsubset">
        <xsl:text disable-output-escaping='yes'>&amp;#x2288;</xsl:text>
      </xsl:if>
    </mo>
    <xsl:apply-templates select = "*[3]" mode = "semantics"/>
  </xsl:if>
  <xsl:if test="*[1]=subset or *[1]=prsubset or *[1]=eq or *[1]=gt
             or *[1]=lt or *[1]=geq or *[1]=leq or *[1]=equivalent">
    <xsl:apply-templates select = "*[2]" mode="semantics"/>
    <xsl:for-each select = "*[position()>2]">
      <mo>
        <xsl:if test="../*[self::subset][1]">
          <xsl:text disable-output-escaping='yes'>&amp;#x2286;</xsl:text>
        </xsl:if>
        <xsl:if test="../*[self::prsubset][1]">
         <xsl:text disable-output-escaping='yes'>&amp;#x2282;</xsl:text>
        </xsl:if>
        <xsl:if test="../*[self::eq][1]">
          <xsl:value-of select="'='"/>
        </xsl:if>
        <xsl:if test="../*[self::gt][1]">
          <xsl:value-of select="'&gt;'"/>
        </xsl:if>
        <xsl:if test="../*[self::lt][1]">
          <xsl:value-of select="'&lt;'"/>
        </xsl:if>
        <xsl:if test="../*[self::geq][1]">
         <xsl:text disable-output-escaping='yes'>&amp;#x2265;</xsl:text>
        </xsl:if>
        <xsl:if test="../*[self::leq][1]">
         <xsl:text disable-output-escaping='yes'>&amp;#x2264;</xsl:text>
        </xsl:if>
        <xsl:if test="../*[self::equivalent][1]">
         <xsl:text disable-output-escaping='yes'>&amp;#x2261;</xsl:text>
        </xsl:if>
      </mo>
      <xsl:apply-templates select = "." mode="semantics"/>
    </xsl:for-each>
  </xsl:if>
</xsl:template>



<!-- ***************** CALCULUS ***************** -->

<xsl:template match = "apply[*[1][self::ln]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="parent::apply[power[1]]">
        <msup>
          <mo> ln </mo>
          <xsl:apply-templates select = "../*[3]" mode = "semantics"/>
        </msup>
      </xsl:when>
      <xsl:otherwise>
        <mo rspace="thinmathspace"> ln </mo>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:apply-templates select = "*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </mrow>
</xsl:template>


<xsl:template match = "apply[log[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="parent::apply[power[1]]">
        <xsl:if test="not(*[2]=logbase)">
          <msup>
            <mo> log </mo>
            <xsl:apply-templates select = "../*[3]" mode = "semantics"/>
          </msup>
        </xsl:if>
        <xsl:if test="*[2]=logbase">
          <msubsup>
            <mo> log </mo>
            <xsl:apply-templates select = "../*[3]" mode = "semantics"/>
            <xsl:apply-templates select = "logbase" mode = "semantics"/>
          </msubsup>
        </xsl:if>
      </xsl:when>
      <xsl:otherwise>
        <xsl:if test="not(*[2]=logbase)">
          <mo rspace="thinmathspace"> log </mo>
        </xsl:if>
        <xsl:if test="*[2]=logbase">
          <msub>
            <mo> log </mo>
            <xsl:apply-templates select = "logbase" mode = "semantics"/>
          </msub>
        </xsl:if>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:if test="*[2]=logbase">
      <xsl:apply-templates select = "*[3]" mode = "semantics">
        <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
      </xsl:apply-templates>
    </xsl:if>
    <xsl:if test="not(*[2]=logbase)">
      <xsl:apply-templates select = "*[2]" mode = "semantics">
        <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
      </xsl:apply-templates>
    </xsl:if>
  </mrow>
</xsl:template>


<xsl:template match = "apply[diff[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
	<xsl:choose>
      <xsl:when test="bvar">
        <xsl:if test="not(bvar[*[2]=degree])">
          <mfrac>
            <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2146;</xsl:text> </mo>
            <mrow>
              <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2146;</xsl:text> </mo>
              <xsl:apply-templates select = "bvar/*[1]" mode = "semantics"/>
            </mrow>
          </mfrac>
        </xsl:if>
        <xsl:if test="bvar[*[2]=degree]">
          <mfrac>
            <msup>
              <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2146;</xsl:text> </mo>
              <xsl:apply-templates select = "bvar/degree" mode = "semantics"/>
            </msup>
            <mrow>
              <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2146;</xsl:text> </mo>
              <msup>
                <xsl:apply-templates select = "bvar/*[1]" mode = "semantics"/>
                <xsl:apply-templates select = "bvar/degree" mode = "semantics"/>
              </msup>
            </mrow>
          </mfrac>
        </xsl:if>
        <xsl:apply-templates select = "*[position()=last() and not(self::bvar)]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
        </xsl:apply-templates>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select = "*[position()=last() and not(self::bvar)]" mode = "semantics">
          <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
        </xsl:apply-templates>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2032;</xsl:text> </mo>
	  </xsl:otherwise>
    </xsl:choose>
  </mrow>
</xsl:template>


<xsl:template match = "apply[partialdiff[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="list">
        <msub>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2145;</xsl:text> </mo>
          <xsl:apply-templates select = "list" mode = "semantics"/>
        </msub>
      </xsl:when>
      <xsl:otherwise>
        <xsl:if test="degree">
		  <mfrac>
            <msup>
              <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2202;</xsl:text> </mo>
              <xsl:apply-templates select = "degree" mode = "semantics"/>
            </msup>
            <mrow>
              <xsl:for-each select = "bvar">
                <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2202;</xsl:text> </mo>
                <xsl:if test="*[last()]=degree">
                  <msup>
                    <xsl:apply-templates select = "*[1]" mode = "semantics"/>
                    <xsl:apply-templates select = "degree" mode = "semantics"/>
                  </msup>
                </xsl:if>
                <xsl:if test="not(*[last()]=degree)">
                  <xsl:apply-templates select = "*[1]" mode = "semantics"/>
                </xsl:if>
              </xsl:for-each>
            </mrow>
		  </mfrac>
		</xsl:if>
        <xsl:if test="not(degree)">
          <xsl:for-each select = "bvar">
            <xsl:if test="*[last()]=degree">
              <mfrac>
                <msup>
                  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2202;</xsl:text> </mo>
                  <xsl:apply-templates select = "degree" mode = "semantics"/>
                </msup>
                <mrow>
                  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2202;</xsl:text> </mo>
                  <msup>
                    <xsl:apply-templates select = "*[1]" mode = "semantics"/>
                    <xsl:apply-templates select = "degree" mode = "semantics"/>
                  </msup>
                </mrow>
              </mfrac>
            </xsl:if>
            <xsl:if test="not(*[last()]=degree)">
              <mfrac>
                <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2202;</xsl:text> </mo>
                <mrow>
                  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2202;</xsl:text> </mo>
                  <xsl:apply-templates select = "*[1]" mode = "semantics"/>
                </mrow>
              </mfrac>
            </xsl:if>
          </xsl:for-each>
		</xsl:if>
	  </xsl:otherwise>
	</xsl:choose>
	<xsl:apply-templates select = "*[last()]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$GEN_FUN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </mrow>
</xsl:template>


<xsl:template match = "lowlimit | uplimit | bvar | degree | logbase">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*" mode = "semantics"/>
  </mrow>
</xsl:template>


<xsl:template match = "apply[divergence[1] | grad[1] | curl[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo>
      <xsl:if test="*[1]=divergence">
        <xsl:value-of select="'div'"/>
      </xsl:if>
      <xsl:if test="*[1]=grad">
        <xsl:value-of select="'grad'"/>
      </xsl:if>
      <xsl:if test="*[1]=curl">
        <xsl:value-of select="'curl'"/>
      </xsl:if>
    </mo>
    <mspace width="0.01em" linebreak="nobreak"/>
    <xsl:choose>
      <xsl:when test="*[2]=ci">
        <xsl:apply-templates select = "*[2]" mode = "semantics"/>
      </xsl:when>
      <xsl:otherwise>
        <mfenced separators="">
          <xsl:apply-templates select = "*[2]" mode = "semantics"/>
        </mfenced>
      </xsl:otherwise>
    </xsl:choose>
  </mrow>
</xsl:template>


<xsl:template match = "apply[laplacian[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <msup>
      <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2207;</xsl:text> </mo>
      <mn> 2 </mn>
    </msup>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2061;</xsl:text> </mo>
    <xsl:apply-templates select = "*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$GEN_FUN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </mrow>
</xsl:template>



<!-- ***************** SET THEORY ***************** -->

<xsl:template match = "set | list">
  <mfenced>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
	<xsl:if test="self::set">
      <xsl:attribute name="open">
        <xsl:value-of select="'{'"/>
      </xsl:attribute>
      <xsl:attribute name="close">
        <xsl:value-of select="'}'"/>
      </xsl:attribute>
    </xsl:if>
	<xsl:if test="self::list">
      <xsl:attribute name="open">
        <xsl:value-of select="'['"/>
      </xsl:attribute>
      <xsl:attribute name="close">
        <xsl:value-of select="']'"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="not(child::bvar) and not(child::condition)">
        <xsl:apply-templates select = "*" mode="semantics"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:attribute name="separators"/>
        <xsl:apply-templates select = "*[not(self::condition) and not(self::bvar)]" mode="semantics"/>
        <mo lspace="0.1666em" rspace="0.1666em"> | </mo>
        <xsl:apply-templates select="condition" mode = "semantics"/>
      </xsl:otherwise>
    </xsl:choose>
  </mfenced>
</xsl:template>


<xsl:template match = "apply[union[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &gt; $UNION_PREC or $IN_PREC=$UNION_PREC
                    and $PARAM=$PAR_SAME">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="union">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="union">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="union">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>


<xsl:template match = "apply[union[1]]" mode="union">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select = "*[2]" mode="semantics">
    <xsl:with-param name="IN_PREC" select="$UNION_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:for-each select = "*[position()>2]">
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x222A;</xsl:text> </mo>
    <xsl:apply-templates select = "." mode="semantics">
      <xsl:with-param name="IN_PREC" select="$UNION_PREC"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:for-each>
</xsl:template>


<xsl:template match = "apply[intersect[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &gt; $INTERSECT_PREC or $IN_PREC=$INTERSECT_PREC
                    and $PARAM=$PAR_SAME">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="intersect">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="intersect">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="intersect">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[intersect[1]]" mode="intersect">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select = "*[2]" mode="semantics">
    <xsl:with-param name="IN_PREC" select="$INTERSECT_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:for-each select = "*[position()>2]">
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2229;</xsl:text> </mo>
    <xsl:apply-templates select = "." mode="semantics">
      <xsl:with-param name="IN_PREC" select="$INTERSECT_PREC"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:for-each>
</xsl:template>


<xsl:template match = "apply[setdiff[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &gt; $SETDIFF_PREC or $IN_PREC=$SETDIFF_PREC
                    and $PARAM=$PAR_SAME">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="setdiff">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="setdiff">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="setdiff">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "apply[setdiff[1]]" mode="setdiff">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select = "*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$SETDIFF_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <mo>\</mo>
  <xsl:apply-templates select = "*[3]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$SETDIFF_PREC"/>
    <xsl:with-param name="PARAM" select="$PAR_SAME"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
  </xsl:apply-templates>
</xsl:template>


<xsl:template match = "apply[cartesianproduct[1]]">
  <xsl:param name="IN_PREC" select="$NO_PREC"/>
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:choose>
    <xsl:when test="$IN_PREC &gt; $CARTPROD_PREC or $IN_PREC=$CARTPROD_PREC
                    and $PARAM=$PAR_SAME">
      <mfenced separators="">
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="cartprod">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mfenced>
    </xsl:when>
    <xsl:when test="$IN_PREC &gt; $NO_PREC and $IN_PREC &lt; $GEN_FUN_PREC
                    and not($SEM_SW=$SEM_ALL) and not($SEM_SW=$SEM_XREF)
                    and not($SEM_SW=$SEM_XREF_EXT)">
      <xsl:apply-templates select="." mode="cartprod">
        <xsl:with-param name="PARAM" select="$PARAM"/>
        <xsl:with-param name="PAREN" select="$PAREN"/>
        <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
      </xsl:apply-templates>
    </xsl:when>
    <xsl:otherwise>
      <mrow>
        <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
          <xsl:attribute name="xref">
            <xsl:value-of select="@id"/>
          </xsl:attribute>
        </xsl:if>
        <xsl:apply-templates select="." mode="cartprod">
          <xsl:with-param name="PARAM" select="$PARAM"/>
          <xsl:with-param name="PAREN" select="$PAREN"/>
          <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
        </xsl:apply-templates>
      </mrow>
    </xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template match = "*" mode="cartprod">
  <xsl:param name="PARAM" select="$NO_PARAM"/>
  <xsl:param name="PAREN" select="$PAR_NO"/>
  <xsl:param name="PAR_NO_IGNORE" select="$YES"/>
  <xsl:apply-templates select = "*[2]" mode = "semantics">
    <xsl:with-param name="IN_PREC" select="$CARTPROD_PREC"/>
    <xsl:with-param name="PARAM" select="$PARAM"/>
    <xsl:with-param name="PAREN" select="$PAREN"/>
    <xsl:with-param name="PAR_NO_IGNORE" select="$PAR_NO_IGNORE"/>
  </xsl:apply-templates>
  <xsl:for-each select = "*[position()>2]">
    <mo><xsl:text disable-output-escaping='yes'>&amp;#x00D7;</xsl:text></mo>
    <xsl:apply-templates select = "." mode="semantics">
      <xsl:with-param name="IN_PREC" select="$CARTPROD_PREC"/>
      <xsl:with-param name="PARAM" select="$PAR_SAME"/>
      <xsl:with-param name="PAREN" select="$PAREN"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </xsl:for-each>
</xsl:template>


<xsl:template match = "apply[card[1]]">
  <mfenced open="&#x2223;" close="&#x2223;" separators=",">
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:for-each select = "*[position()>1]">
      <xsl:apply-templates select = "." mode="semantics"/>
    </xsl:for-each>
  </mfenced>
</xsl:template>



<!-- ***************** SEQUENCES AND SERIES ***************** -->

<xsl:template match = "apply[sum[1] | product[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="*[2]=bvar and lowlimit and uplimit">
        <munderover>
          <mo>
            <xsl:if test="*[1]=sum">
             <xsl:text disable-output-escaping='yes'>&amp;#x2211;</xsl:text>
            </xsl:if>
            <xsl:if test="*[1]=product">
             <xsl:text disable-output-escaping='yes'>&amp;#x220F;</xsl:text>
            </xsl:if>
          </mo>
          <mrow>
            <xsl:apply-templates select = "*[2]" mode = "semantics"/>
            <mo> = </mo>
            <xsl:apply-templates select = "lowlimit" mode = "semantics"/>
          </mrow>
          <xsl:apply-templates select = "uplimit" mode = "semantics"/>
        </munderover>
        <xsl:apply-templates select = "*[5]" mode = "semantics"/>
      </xsl:when>
      <xsl:when test="*[2]=bvar and *[3]=condition">
        <munder>
          <mo>
            <xsl:if test="*[1]=sum">
             <xsl:text disable-output-escaping='yes'>&amp;#x2211;</xsl:text>
            </xsl:if>
            <xsl:if test="*[1]=product">
             <xsl:text disable-output-escaping='yes'>&amp;#x220F;</xsl:text>
            </xsl:if>
          </mo>
          <xsl:apply-templates select = "*[3]" mode = "semantics"/>
        </munder>
        <xsl:apply-templates select = "*[4]" mode = "semantics"/>
      </xsl:when>
      <xsl:when test="*[2]=domainofapplication">
        <munder>
          <mo>
            <xsl:if test="*[1]=sum">
              <xsl:text disable-output-escaping='yes'>&amp;#x2211;</xsl:text>
            </xsl:if>
            <xsl:if test="*[1]=product">
              <xsl:text disable-output-escaping='yes'>&amp;#x220F;</xsl:text>
            </xsl:if>
          </mo>
          <xsl:apply-templates select="domainofapplication" mode = "semantics"/>
        </munder>
        <mrow>
          <xsl:apply-templates select="*[position()=last()]" mode = "semantics"/>
        </mrow>
      </xsl:when>
    </xsl:choose>
  </mrow>
</xsl:template>


<xsl:template match="apply[*[1][self::int]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
	<xsl:choose>
	  <xsl:when test="domainofapplication">
        <munder>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x222B;</xsl:text> </mo>
          <xsl:apply-templates select="domainofapplication" mode="semantics"/>
        </munder>
      </xsl:when>
	  <xsl:when test="condition">
        <munder>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x222B;</xsl:text> </mo>
          <xsl:apply-templates select="condition" mode="semantics"/>
        </munder>
      </xsl:when>
      <xsl:when test="interval">
        <munderover>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x222B;</xsl:text> </mo>
          <mrow>
            <xsl:apply-templates select="interval/*[position()=1]" mode="semantics"/>
          </mrow>
          <mrow>
            <mspace width="1em"/>
            <xsl:apply-templates select="interval/*[position()=2]" mode="semantics"/>
          </mrow>
		</munderover>
      </xsl:when>
      <xsl:when test="lowlimit | uplimit">
        <munderover>
          <mo> <xsl:text disable-output-escaping='yes'>&amp;#x222B;</xsl:text> </mo>
          <xsl:apply-templates select="lowlimit" mode="semantics"/>
          <mrow>
            <mspace width="1em"/>
            <xsl:apply-templates select="uplimit" mode="semantics"/>
          </mrow>
        </munderover>
      </xsl:when>
	  <xsl:otherwise>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x222B;</xsl:text> </mo>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:apply-templates select="*[position()=last() and last()>1 and not(self::domainofapplication) and not(self::condition) and not(self::interval) and not(self::lowlimit) and not(self::uplimit) and not(self::bvar)]" mode="semantics">
      <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
	<xsl:if test="bvar">
      <mrow>
        <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2146;</xsl:text> </mo>
        <xsl:apply-templates select="bvar" mode="semantics"/>
      </mrow>
    </xsl:if>
  </mrow>
</xsl:template>


<xsl:template match = "apply[limit[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <munder>
      <mo> lim </mo>
      <mrow>
        <xsl:if test="*[2]=bvar and *[3]=lowlimit">
            <xsl:apply-templates select = "*[2]" mode = "semantics"/>
            <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2192;</xsl:text> </mo>
            <xsl:apply-templates select = "*[3]" mode = "semantics"/>
        </xsl:if>
        <xsl:if test="*[2]=bvar and *[3]=condition">
          <xsl:apply-templates select = "*[3]" mode = "semantics"/>
        </xsl:if>
      </mrow>
    </munder>
    <xsl:apply-templates select = "*[4]" mode = "semantics"/>
  </mrow>
</xsl:template>



<!-- ***************** TRIGONOMETRY ***************** -->

<xsl:template match = "apply[*[1][self::sin | self::cos |
                       self::tan | self::sec | self::csc |
                       self::cot | self::sinh | self::cosh |
                       self::tanh | self::sech | self::csch |
                       self::coth | self::arcsin | self::arccos |
                       self::arctan | self::arcsec | self::arccsc |
                       self::arccot | self::arcsinh | self::arccosh |
                       self::arctanh | self::arcsech | self::arccsch |
                       self::arccoth]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:if test="not(parent::apply[power[1]])">
      <xsl:apply-templates select = "*[1]" mode = "trigonometry"/>
    </xsl:if>
    <xsl:if test="parent::apply[power[1]]">
      <msup>
        <xsl:apply-templates select = "*[1]" mode = "trigonometry"/>
        <xsl:apply-templates select = "../*[3]" mode = "semantics"/>
      </msup>
    </xsl:if>
    <mspace width="0.01em" linebreak="nobreak"/>
    <xsl:apply-templates select = "*[2]" mode = "semantics">
      <xsl:with-param name="IN_PREC" select="$FUNCTN_PREC"/>
      <xsl:with-param name="PAR_NO_IGNORE" select="$NO"/>
    </xsl:apply-templates>
  </mrow>
</xsl:template>

<xsl:template match = "sin | cos |
                       tan | sec | csc |
                       cot | sinh | cosh |
                       tanh | sech | csch |
                       coth | arcsin | arccos |
                       arctan | arcsec | arccsc |
                       arccot | arcsinh | arccosh |
                       arctanh | arcsech | arccsch |
                       arccoth">
  <xsl:apply-templates select = "." mode = "trigonometry"/>
</xsl:template>

<xsl:template match = "*" mode="trigonometry">
  <mo>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:choose>
      <xsl:when test="self::sin">
        <xsl:value-of select="'sin'"/>
      </xsl:when>
      <xsl:when test="self::cos">
        <xsl:value-of select="'cos'"/>
      </xsl:when>
      <xsl:when test="self::tan">
        <xsl:value-of select="'tan'"/>
      </xsl:when>
      <xsl:when test="self::sec">
        <xsl:value-of select="'sec'"/>
      </xsl:when>
      <xsl:when test="self::csc">
        <xsl:value-of select="'csc'"/>
      </xsl:when>
      <xsl:when test="self::cot">
        <xsl:value-of select="'cot'"/>
      </xsl:when>
      <xsl:when test="self::sinh">
        <xsl:value-of select="'sinh'"/>
      </xsl:when>
      <xsl:when test="self::cosh">
        <xsl:value-of select="'cosh'"/>
      </xsl:when>
      <xsl:when test="self::tanh">
        <xsl:value-of select="'tanh'"/>
      </xsl:when>
      <xsl:when test="self::sech">
        <xsl:value-of select="'sech'"/>
      </xsl:when>
      <xsl:when test="self::csch">
        <xsl:value-of select="'csch'"/>
      </xsl:when>
      <xsl:when test="self::coth">
        <xsl:value-of select="'coth'"/>
      </xsl:when>
      <xsl:when test="self::arcsin">
        <xsl:value-of select="'arcsin'"/>
      </xsl:when>
      <xsl:when test="self::arccos">
        <xsl:value-of select="'arccos'"/>
      </xsl:when>
      <xsl:when test="self::arctan">
        <xsl:value-of select="'arctan'"/>
      </xsl:when>
      <xsl:when test="self::arcsec">
        <xsl:value-of select="'arcsec'"/>
      </xsl:when>
      <xsl:when test="self::arccsc">
        <xsl:value-of select="'arccsc'"/>
      </xsl:when>
      <xsl:when test="self::arccot">
        <xsl:value-of select="'arccot'"/>
      </xsl:when>
      <xsl:when test="self::arcsinh">
        <xsl:value-of select="'arcsinh'"/>
      </xsl:when>
      <xsl:when test="self::arccosh">
        <xsl:value-of select="'arccosh'"/>
      </xsl:when>
      <xsl:when test="self::arctanh">
        <xsl:value-of select="'arctanh'"/>
      </xsl:when>
      <xsl:when test="self::arcsech">
        <xsl:value-of select="'arcsech'"/>
      </xsl:when>
      <xsl:when test="self::arccsch">
        <xsl:value-of select="'arccsch'"/>
      </xsl:when>
      <xsl:when test="self::arccoth">
        <xsl:value-of select="'arccot'"/>
      </xsl:when>
    </xsl:choose>
  </mo>
</xsl:template>



<!-- ***************** STATISTICS ***************** -->

<xsl:template match = "apply[mean[1]]">
  <mfenced open="&#x2329;" close="&#x232A;" separators=",">
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:for-each select = "*[position()>1]">
      <xsl:apply-templates select = "." mode="semantics"/>
    </xsl:for-each>
  </mfenced>
</xsl:template>


<xsl:template match = "apply[sdev[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x03C3;</xsl:text> </mo>
    <mfenced separators=",">
      <xsl:for-each select = "*[position()>1]">
        <xsl:apply-templates select = "." mode="semantics"/>
      </xsl:for-each>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[variance[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> <xsl:text disable-output-escaping='yes'>&amp;#x03C3;</xsl:text> </mo>
    <msup>
      <mfenced separators=",">
        <xsl:for-each select = "*[position()>1]">
          <xsl:apply-templates select = "." mode="semantics"/>
        </xsl:for-each>
      </mfenced>
      <mn> 2 </mn>
    </msup>
  </mrow>
</xsl:template>


<xsl:template match = "apply[median[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> median </mo>
    <mfenced separators=",">
      <xsl:for-each select = "*[position()>1]">
        <xsl:apply-templates select = "." mode="semantics"/>
      </xsl:for-each>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[mode[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> mode </mo>
    <mfenced separators=",">
      <xsl:for-each select = "*[position()>1]">
        <xsl:apply-templates select = "." mode="semantics"/>
      </xsl:for-each>
    </mfenced>
  </mrow>
</xsl:template>


<xsl:template match = "apply[moment[1]]">
  <mfenced open="&#x2329;" close="&#x232A;" separators="">
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:if test="*[2]=degree and not(*[3]=momentabout)">
      <msup>
        <xsl:apply-templates select="*[3]" mode = "semantics"/>
        <xsl:apply-templates select="*[2]" mode = "semantics"/>
      </msup>
    </xsl:if>
    <xsl:if test="*[2]=degree and *[3]=momentabout">
      <msup>
        <xsl:apply-templates select="*[4]" mode = "semantics"/>
        <xsl:apply-templates select="*[2]" mode = "semantics"/>
      </msup>
    </xsl:if>
    <xsl:if test="not(*[2]=degree) and *[2]=momentabout">
      <xsl:for-each select = "*[position()>2]">
        <xsl:apply-templates select = "." mode="semantics"/>
      </xsl:for-each>
    </xsl:if>
    <xsl:if test="not(*[2]=degree) and not(*[2]=momentabout)">
      <xsl:for-each select = "*[position()>1]">
        <xsl:apply-templates select = "." mode="semantics"/>
      </xsl:for-each>
    </xsl:if>
  </mfenced>
</xsl:template>



<!-- ***************** LINEAR ALGEBRA ***************** -->

<xsl:template match="vector">
  <mfenced separators="">
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mtable columnalign="center">
      <xsl:for-each select="*">
        <mtr>
          <mtd> <xsl:apply-templates select="." mode = "semantics"/> </mtd>
        </mtr>
      </xsl:for-each>
    </mtable>
  </mfenced>
</xsl:template>


<xsl:template match = "matrix">
  <mfenced separators="">
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mtable>
      <xsl:apply-templates mode = "semantics"/>
    </mtable>
  </mfenced>
</xsl:template>


<xsl:template match = "matrixrow">
  <mtr>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:for-each select="*">
      <mtd>
        <xsl:apply-templates select="." mode = "semantics"/>
      </mtd>
    </xsl:for-each>
  </mtr>
</xsl:template>


<xsl:template match = "apply[determinant[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <mo> det </mo>
    <mspace width="0.2em" linebreak="nobreak"/>
    <xsl:apply-templates select = "*[2]" mode = "semantics"/>
  </mrow>
</xsl:template>


<xsl:template match = "apply[transpose[1]]">
  <msup>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select = "*[2]" mode = "semantics"/>
    <mo> T </mo>
  </msup>
</xsl:template>


<xsl:template match = "apply[selector[1]]">
  <msub>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*[2]" mode = "semantics"/>
    <mfenced open="" close="">
      <xsl:for-each select="*[position()>2]">
        <xsl:apply-templates select="." mode = "semantics"/>
      </xsl:for-each>
    </mfenced>
  </msub>
</xsl:template>


<xsl:template match = "apply[vectorproduct[1] |
                                 scalarproduct[1] | outerproduct[1]]">
  <mrow>
    <xsl:if test="($SEM_SW=$SEM_XREF or $SEM_SW=$SEM_XREF_EXT) and @id">
      <xsl:attribute name="xref">
        <xsl:value-of select="@id"/>
      </xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="*[2]" mode = "semantics"/>
    <mo>
      <xsl:if test="vectorproduct[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x00D7;</xsl:text>
      </xsl:if>
      <xsl:if test="scalarproduct[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x22C5;</xsl:text>
      </xsl:if>
      <xsl:if test="outerproduct[1]">
        <xsl:text disable-output-escaping='yes'>&amp;#x2297;</xsl:text>
      </xsl:if>
    </mo>
    <xsl:apply-templates select="*[3]" mode = "semantics"/>
  </mrow>
</xsl:template>



<!-- ***************** CONSTANT and SYMBOL ELEMENTS ***************** -->

<xsl:template match="integers">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2124;</xsl:text> </mo>
</xsl:template>

<xsl:template match="reals">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x211D;</xsl:text> </mo>
</xsl:template>

<xsl:template match="rationals">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x211A;</xsl:text> </mo>
</xsl:template>

<xsl:template match="naturalnumbers">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2115;</xsl:text> </mo>
</xsl:template>

<xsl:template match="complexes">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2102;</xsl:text> </mo>
</xsl:template>

<xsl:template match="primes">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2119;</xsl:text> </mo>
</xsl:template>

<xsl:template match="exponentiale">
  <mn> <xsl:text disable-output-escaping='yes'>&amp;#x2147;</xsl:text> </mn>
</xsl:template>

<xsl:template match="imaginaryi">
  <mn> <xsl:text disable-output-escaping='yes'>&amp;#x2148;</xsl:text> </mn>
</xsl:template>

<xsl:template match="notanumber">
  <mo> NaN </mo>
</xsl:template>

<xsl:template match="true">
  <mo> true </mo>
</xsl:template>

<xsl:template match="false">
  <mo> false </mo>
</xsl:template>

<xsl:template match="emptyset">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x2205;</xsl:text> </mo>
</xsl:template>

<xsl:template match="pi">
  <mn> <xsl:text disable-output-escaping='yes'>&amp;#x03C0;</xsl:text> </mn>
</xsl:template>

<xsl:template match="eulergamma">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x213D;</xsl:text> </mo>
</xsl:template>

<xsl:template match="infinity">
  <mo> <xsl:text disable-output-escaping='yes'>&amp;#x221E;</xsl:text> </mo>
</xsl:template>

</xsl:stylesheet>
