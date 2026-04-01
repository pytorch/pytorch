Reverse Mode Autodiff (Out of Date)
==================================


This document serves as a design reference for reverse-mode auto-diff in the Slang compiler.

## Reverse-Mode Passes

Rather than implementing reverse-mode as a separate pass, Slang implements this as a series of independent passes:

If a function needs a reverse-mode version generated:
 - *Linearize* the function, and all dependencies.
 - *Propagate* differential types through the linearized code.
 - *Unzip* by moving primal insts to before differential insts.
 - *Transpose* the differential insts.


## Linearization (Forward-mode)

### Overview
(This is a incomplete section. More details coming soon)

Consider an arbitrary function `float f(float a, float b, float c, ..., z)` which takes in N inputs and generates one output `y`. Linearization aims to generate the first-order Taylor expansion of f about _all_ of it's inputs.

Mathematically, the forward derivative `fwd_f` represents `df/da * (a_0 - a)  + df/db * (b_0 - b) + ...`, where `a_0` is the value at which the Taylor expansion was produced. The quantity `a_0 - a` is known as the 'differential' (for brevity we'll denote them da, db, dc, etc..), and there is at-most one differential per input.

Thus, the new function's signature should be `fwd_f(float a, float da, float b, float db, float c, float dc, ...)`. For simplicity, we'll use *pairs* instead of interleaving the original and differential parameters. We use the intrinsic `DifferentialPair<T>` (or for short: `DP<T>`) to denote this.

The signature we use is then `fwd_f(DP<float> a, DP<float> b, DP<float> c)`

An example of linearization:
```C

float f(float a, float b)
{
    if (a > 0)
    {
        return a + b + 2.0 * a * b;
    }
    else
    {
        return sqrt(a);
    }
}
```

We'll write out the SSA form of this function.

```C
float f_SSA(float a, float b)
{
    bool _b1 = a > 0;
    if (_b1)
    {
        float _t1 = a + b;
        float _t2 = 2.0 * a;
        float _t3 = _t2 * b;
        float _t4 = _t1 + _t3;

        return _t4;
    }
    else
    {
        float _t1 = sqrt(a);
        return _t1;
    }
}

DP<float> f_SSA(DP<float> dpa, DP<float> dpb)
{

    bool _b1 = dpa.p > 0;
    if (_b1)
    {
        float _t1 = dpa.p + dpb.p;
        float _t1_d = dpa.d + dpb.d;

        float _t2 = 2.0 * dpa.p;
        float _t2_d = 0.0 * dpa.p + 2.0 * dpa.d;

        float _t3 = _t2 * dpb.p;
        float _t3_d = _t2_d * dpb.p + _t2 * dpb.d;

        float _t4 = _t1 + _t3;
        float _t4_d = _t1_d + _t3_d;

        return DP<float>(_t4, _t4_d);
    }
    else
    {
        DP<float> _t1_dp = sqrt_fwd(dpa);
        return DP<float>(_t1_dp.p, _t1_dp.d);
    }
}

```

In the result, the primal part of the pair holds the original computation, while the differential part computes the dot product of the differentials with the derivatives of the function's output w.r.t each input. 


## Propagation

This step takes a linearized function and propagates information about which instructions are computing a differential and which ones are part of the primal (original) computation.

Assuming first-order differentiation only:
The approach will be to mark any instructions that extract the differential from the differential pair as a differential. Then any instruction that uses the differential is itself marked as a differential and so on. The only exception is the call instruction which is either non-differentiable (do nothing) or differentiable and returns a pair (follow the same process)


Here's the above example with propagated type information (we use float.D to denote intermediaries that have been marked as differential, and also expand everything so that each line has a single operation)

```C

DP<float> f_SSA_Proped(DP<float> dpa, DP<float> dpb)
{
    bool _b1 = dpa.p > 0;
    if (_b1)
    {
        float _t1 = dpa.p + dpb.p;
        
        float.D _q1_d = dpa.d;
        float.D _q2_d = dpb.d;

        float.D _t1_d = _q1_d + _q2_d;

        float _t2 = 2.0 * dpa.p;
        
        float.D _q2_d = dpa.d;
        float.D _q3_d = 2.0 * dpa.d;

        float _q4 = dpa.p;
        float.D _q4_d = 0.0 * dpa.p;

        float.D _t2_d = _q4_d + _q3_d;

        float _t3 = _t2 * dpb.p;

        float _q5 = dpb.p;
        float.D _q6_d = _q5 * _t2_d;

        float.D _q7_d = dpb.d;
        float.D _q8_d = _t2 * _q7_d

        float _t3_d = _q6_d + _q8_d;

        float _t4 = _t1 + _t3;

        float.D _t4_d = _t1_d + _t3_d;

        return DP<float>(_t4, _t4_d);
    }
    else
    {
        DP<float> _t1_dp = sqrt_fwd(dpa);

        float _q1 = _t1_dp.p;
        float.D _q1_d = _t1_dp.d;

        return DP<float>(_q1, _q1_d);
    }
}

```

## Unzipping


This is a fairly simple process when there is no control flow. We simply move all non-differential instructions to before the first differential instruction.

When there is control flow, we need to be a bit more careful: the key is to *replicate* the control flow graph once for primal and once for the differential.

Here's the previous example unzipped:


```C

DP<float> f_SSA_Proped(DP<float> dpa, DP<float> dpb)
{
    bool _b1 = dpa.p > 0;

    float _t1, _t2, _q4, _t3, _q5, _t3_d, _t4, _q1;

    if (_b1)
    {
        _t1 = dpa.p + dpb.p;
        
        _t2 = 2.0 * dpa.p;
        
        _q4 = dpa.p;
        
        _t3 = _t2 * dpb.p;

        _q5 = dpb.p;

        _t4 = _t1 + _t3;

    }
    else
    {

        _q1 = sqrt_fwd(DP<float>(dpa.p, 0.0));
    }

    // Note here that we have to 'store' all the intermediaries 
    // _t1, _t2, _q4, _t3, _q5, _t3_d, _t4 and _q1. This is fundamentally
    // the tradeoff between fwd_mode and rev_mode

    if (_b1)
    {
        float.D _q1_d = dpa.d;
        float.D _q2_d = dpb.d;

        float.D _t1_d = _q1_d + _q2_d;

        float.D _q2_d = dpa.d;
        float.D _q3_d = 2.0 * dpa.d;

        float.D _q4_d = 0.0 * dpa.p;

        float.D _t2_d = _q4_d + _q3_d;

        float.D _q6_d = _q5 * _t2_d;

        float.D _q7_d = dpb.d;
        float.D _q8_d = _t2 * _q7_d

        float.D _t3_d = _q6_d + _q8_d;

        float.D _t4_d = _t1_d + _t3_d;

        return DP<float>(_t4, _t4_d);
    }
    else
    {
        DP<float> _t1_dp = sqrt_fwd(dpa);

        float.D _q1_d = _t1_dp.d;

        return DP<float>(_q1, _q1_d);
    }
}

```

## Transposition

### Overview

This transposition pass _assumes_ that provided function is linear in it's differentials.
It is out of scope of this project to attempt to enforce that constraint for user-defined differential code.

For transposition we walk all differential instructions in reverse starting from the return statement, and apply the following rules:

We'll have an accumulator dictionary `Dictionary<IRInst, IRInst> accMap` holding assignments for
intermediaries which don't have concrete variables. When we add a pair (A, C) and (A, B) already exists, this will form the pair (A, ADD(C, B)) in the dictionary. (ADD will be replaced with a call to `T.dadd` for a generic type T)

 - If `inst` is a `RETURN(A)`, add pair `(A, d_out)` to `accMap`
 - If an instruction is `MUL(P, D)` where D is the differential, add pair `(D, MUL(P, accMap[this_inst]))` to `accMap`
 - If an instruction is `ADD(D1, D2)`, where both D1 and D2 are differentials (this is the only config that should occur), then add pair `(D1, accMap[this_inst])` to `accMap`
 - If an instruction is `CALL(f_fwd, (P1, D1), (P2, D2), ...)`, create variables D1v, D2v, ... for D1, D2, ..., then replace with `CALL(f_rev, (P1, D1v), (P2, D2v), ..., accMap[this_inst])`, and finally add pairs `(D1, LOAD[D1v]), (D2, LOAD[D2v]), ...` to `accMap`

 ```C

void f_SSA_Rev(inout DP<float> dpa, inout DP<float> dpb, float dout)
{
    bool _b1 = dpa.p > 0;

    float _t1, _t2, _q4, _t3, _q5, _t3_d, _t4, _q1;

    if (_b1)
    {
        _t1 = dpa.p + dpb.p;
        
        _t2 = 2.0 * dpa.p;
        
        _q4 = dpa.p;
        
        _t3 = _t2 * dpb.p;

        _q5 = dpb.p;

        _t4 = _t1 + _t3;

    }
    else
    {

        _q1 = sqrt_fwd(DP<float>(dpa.p, 0.0));
    }

    // Note here that we have to 'store' all the intermediaries 
    // _t1, _t2, _q4, _t3, _q5, _t3_d, _t4 and _q1. This is fundamentally
    // the tradeoff between fwd_mode and rev_mode

    if (_b1)
    {

        float.D _t4_rev = d_out;

        float.D _t1_rev = _t4_rev;
        float.D _t3_rev = _t4_rev;

        float.D _q8_rev = _t3_rev;
        float.D _q6_rev = _t3_rev;

        float.D _q7_rev = _t2 * _q8_rev;

        dpb.d += _q7_rev;

        float.D _t2_rev = _q5 * _q6_rev;

        float.D _q4_rev = _t2_rev;
        float.D _q3_rev = _t2_rev;

        dpa.d += 2.0 * _q3_rev;

        float.D _q1_rev = _t1_rev;
        float.D _q2_rev = _t1_rev;

        dpb.d += _q2_rev;
        dpa.d += _q1_rev;
    }
    else
    {
        _q1_rev = d_out;

        DP<float> dpa_copy;
        sqrt_rev(dpa_copy, _q1_rev);

        dpa.d += dpa_copy.d;
    }
}

```
