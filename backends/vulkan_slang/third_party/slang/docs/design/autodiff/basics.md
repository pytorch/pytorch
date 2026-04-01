<!--The goal of this set of documents is to describe the design of Slang's automatic differentiation passes, along with the mechanisms & passes used to support various features. -->

This documentation is intended for Slang contributors and is written from a compiler engineering point of view. For Slang users, see the user-guide at this link: [https://shader-slang.com/slang/user-guide/autodiff.html](https://shader-slang.com/slang/user-guide/autodiff.html)

## What is Automatic Differentiation?

Before diving into the design of the automatic differentiation (for brevity, we will call it 'auto-diff') passes, it is important to understand the end goal of what auto-diff tries to achieve.

The over-arching goal of Slang's auto-diff is to enable the user to compute derivatives of a given shader program or function's output w.r.t its input parameters. This critical compiler feature enables users to quickly use their shaders with gradient-based parameter optimization algorithms, which forms the backbone of modern machine learning systems. It enables users to train and deploy graphics systems that contain ML primitives (like multi-layer perceptron's or MLPs) or use their shader programs as differentiable primitives within larger ML pipelines.

### More Resources
Here are some links to resources that talk more about differentiable programming from a more mathematical perspective:
1. UCSD CSE 291 (Spring 2024): https://cseweb.ucsd.edu/~tzli/cse291/sp2024/
2. UW CSE 5990 (Winter 2024): https://sites.google.com/cs.washington.edu/cse-599o-dppl

## Definition of Derivatives

This section is based off of these slides: https://cseweb.ucsd.edu/~tzli/cse291/sp2024/lectures/03_forward_mode.pdf.

Here, we establish the mathematical definition of derivatives, starting with a simple 1D case (function with a single input and output), and extending to the general case of functions mapping multiple inputs to multiple outputs.

To avoid confusion, we will denote mathematical functions using LaTeX italic script ($f$, $g$, etc..) and programs that compute these functions with markdown code (`f`, `g`, etc..)

### Derivatives of scalar (1D) functions

Consider the simplest case: a smooth scalar mathematical function that maps a real number to another real number:

$$f : \mathbb{R} \to \mathbb{R}$$

There are several definitions for a derivative, but we will use the definition that a derivative is the *closest linear approximation* of the output function at a given input location. 
Concretely, given a specific input $x$, we can create a linear approximation of the function $f$ around $x$ as follows:

$$ f(x + dx) \approx f(x) + Df(x) \cdot dx $$
<!--// TODO: Add image here.-->

This can also be understood as a geometric 'tangent' to the function at $x$. $Df(x)$ is the slope of $f$ at $x$, i.e. $\frac{\partial f}{\partial x}$, and $dx$ is the perturbation away from $x$. Our approximation is linear as a function of the perturbation $dx$. Note that no matter how non-linear or complex the underlying function $f(x)$ is, the approximation is always linear (this property becomes very important later).

### Forward-mode derivative functions

Now consider a concrete program `f` that computes some function.

```C
// Computes square of x
float f(float x)
{
    return x * x;
}
```

What should its derivative program look like? We the need the output $f(x)$ and the product of derivative at $x$, $Df(x)$ with the differential $dx$.

In Slang, we put both of these together into a single function, called the *forward-mode derivative* function, which takes in a pair $(x, dx)$ returns a pair $(f(x), Df(x)\cdot dx)$ Note that in auto-diff literature, this is also often referred to as the *total derivative* function. 

```C
DifferentialPair<float> fwd_f(DifferentialPair<float> dpx)
{
    float x = dpx.getPrimal(); // Can also be accessed via property dpx.p
    float dx = dpx.getDifferential(); // Can also be accessed via property dpx.d
    return makePair(x * x, (2 * x) * dx);
}
```

Note that `(2 * x)` is the multiplier corresponding to $Df(x)$. We refer to $x$ and $f(x)$ as "*primal*" values and the perturbations $dx$ and $Df(x)\cdot dx$ as "*differential*" values. The reason for this separation is that the "*differential*" output values are always linear w.r.t their "*differential*" inputs.

As the name implies, `DifferentialPair<T>` is a special pair type used by Slang to hold values and their corresponding differentials.


### Forward-mode derivatives for higher-dimensional functions
In practice, most functions tend to have multiple inputs and multiple outputs, i.e. $f: \mathbb{R}^N \to \mathbb{R}^M$

The definition above can be extended to higher dimensions, using the closest-linear-approximation idea. The main difference is that the derivative function represents a hyperplane rather than a line.

Effectively, we want our forward-mode derivative to compute the following:

$$ f(\mathbf{x} + \mathbf{dx}) \approx f(\mathbf{x}) + \langle Df(\mathbf{x}),\mathbf{dx}\rangle $$

Here, the input and its differential can be represented as a vector quantity $\mathbf{x}, \mathbf{dx} \in \mathbb{R}^N$ and the multiplier $Df(\mathbf{x})$ (also known as the *Jacobian* matrix) is a NxM matrix, and $\left\< \cdot,\cdot \right\>$ denotes the inner product (i.e. matrix-vector multiplication)

Here's an example of a Slang function taking in two inputs (N=2) and generating one output (M=1)

```C
// Compute length of hypotenuse.
float f(float x, float y)
{
    return sqrt(x * x + y * y);
}
```

and its forward-mode derivative:

```C
// Closest linear approximation at x, y
DifferentialPair<float> fwd_f(DifferentialPair<float> dpx, DifferentialPair<float> dpy)
{
    float x = dpx.p;
    float y = dpy.p;
    float dx = dpx.d;
    float dy = dpx.d;

    return DifferentialPair<float>(
        sqrt(x * x + y * y),                       // f(x, y)
        (x * dx + y * dy) / sqrt(x * x, y * y));   // <Df(x,y), dx>
}
```

Important note: the forward-mode function only needs to compute the inner product $\langle Df(\mathbf{x}),\mathbf{dx} \rangle$. The Jacobian matrix itself never needs to be fully materialized. This is a key design element of automatic differentiation, one which allows it to scale to huge input/output counts.

### Building Blocks: Forward-mode derivatives compose in forward order of execution.

In practice, we compute forward-mode derivatives of a complex function by decomposing them into constituent functions (or in compiler-speak: instructions) and composing the forward-mode derivative of each piece in the **same** order. 
This is because of each forward derivative is a 'right-side' product (or product of Jacobian matrix with a vector)

Here's an example of this in action (consider a complex function $h$ composed of $f$ and $g$):

$$ h(\mathbf{x}) = f(g(\mathbf{x})) $$

It's forward-mode derivative is then:

$$ \langle Dh(\mathbf{x}), \mathbf{dx}\rangle = \big\langle Df(\mathbf{x}), \langle Dg(\mathbf{x}), \mathbf{dx}\rangle\big\rangle $$

which is the forward-mode derivative of the outer function $f$ evaluated on the result of the forward-mode derivative of the inner function $g$. 

An example of this in Slang code:
```C
// Compute square.
float sqr(float x)
{
    return x * x;
}

// Compute length of hypotenuse.
float f(float x, float y)
{
    float x_sqr = sqr(x);
    float y_sqr = sqr(y)
    return sqrt(x_sqr + y_sqr);
}
```

The resulting derivative of `f` can be computed by composition:
```C
// Forward-mode derivative of sqr()
DifferentialPair<float> fwd_sqr(DifferentialPair<float> dpx)
{
    float x = dpx.getPrimal();
    float dx = dpx.getDifferential();

    return DifferentialPair<float>(x * x, 2 * x * dx);
}

// Forward-mode derivative of f()
DifferentialPair<float> fwd_f(DifferentialPair<float> dpx, DifferentialPair<float> dpy)
{
    DifferentialPair<float> dp_x_sqr = fwd_sqr(dpx);
    DifferentialPair<float> dp_y_sqr = fwd_sqr(dpy);

    float x_sqr = dp_x_sqr.getPrimal();
    float y_sqr = dp_y_sqr.getPrimal();
    float x_sqr_d = dp_x_sqr.getDifferential();
    float y_sqr_d = dp_y_sqr.getDifferential();

    return DifferentialPair<float>(
        sqrt(x_sqr + y_sqr),
        (x_sqr_d + y_sqr_d) / sqrt(x_sqr + y_sqr));
}
```

### Tip: Extracting partial derivatives from a forward-mode derivative (i.e. a 'total' derivative)

As we discussed above, forward-mode derivatives compute $\langle Df(\mathbf{x}),\mathbf{dx}\rangle$ rather than what you may be used to seeing in a calculus course (e.g. partial derivatives like $\frac{\partial f}{\partial x}$).

In fact, the forward-mode derivative is simply an product of the partial derivative w.r.t each input parameter multiplied by their differential perturbations $\frac{\partial f}{\partial x} * dx + \frac{\partial f}{\partial x} * dy$. This is the reason for the alternative name: *total derivative*.

Thus, partial derivative can be obtained by successively setting each input's differential to 1 (and 0 for everything else)
Example:
```C
// Compute partial derivative w.r.t x (pass dx=1.0)
float df_dx = fwd_f(DifferentialPair<float>(x, 1.0), DifferentialPair<float>(y, 0.0)).d;

// Compute partial derivaive w.r.t y (pass dy=1.0)
float df_dy = fwd_f(DifferentialPair<float>(x, 0.0), DifferentialPair<float>(y, 1.0)).d;
```

### Tip: Testing forward-mode derivatives using the first principles of calculus (i.e. the *finite difference* method)

In Calculus, partial derivatives of a function are often defined in a 'black box' manner using limits, by perturbing a single parameter by an infinitesimal amount:

$$ \frac{\partial f}{\partial x} = \lim_{dx\to 0} \frac{f(x + dx) - f(x - dx)}{2 * dx} $$

At the moment, we cannot leverage programming languages to compute true inifinitesimal limits, but we can replace $dx \to 0$ with a sufficiently small $\epsilon$ leading to the following 'test' to check if derivatives produced by automatic differentiation match with their true mathematical expected values.

Here's an example of using this idea to test functions (many autodiff tests were written this way)

```C
// Compute partial derivative w.r.t x analytically
float df_dx_ad = fwd_f(DifferentialPair<float>(x, 1.0), DifferentialPair<float>(y, 0.0))

// Compute partial derivative w.r.t x through the finite difference (FD) method.
float eps = 1e-4
float df_dx_fd = (f(x + eps, y) - f(x - eps, y)) / (2 * eps);

// If computed correctly, df_dx_ad and df_dx_fd are very close.
```

**Caveats:**
Since the finite difference method only produces a biased estimate of the derivative, the result is only numerically *close* to the auto-diff-based result. Poorly behaved functions (those that rapidly change, or are discontinuous or otherwise non-differentiable) will result in a (expected) mismatch between FD and AD results.

## Reverse-mode derivative functions

This section is based off of these slides: https://cseweb.ucsd.edu/~tzli/cse291/sp2024/lectures/05_reverse_mode.pdf.

### Motivation: Challenges with scaling forward-mode derivatives

A big problem with forward-mode derivatives is their inability to scale to great parameter counts.

Machine learning pipelines often compute derivatives of a large complex pipeline with millions or even billions of input parameters, but a single output value, i.e. the *loss* or *objective* function, frequently denoted by $\mathcal{L}$.
Computing $\frac{\partial \mathcal{L}}{\partial x_i}$ for $N$ inputs $x_i$ using the one-hot vector approach will involve invoking the forward-mode derivative function $N$ times.

The reason for this limitation is that forward-mode derivatives pass derivatives from the inputs through to the outputs by computing the dot-product $\left\< Df(\mathbf{x}),\mathbf{dx}\right\>$. 
Instead, we employ a different approach called the reverse-mode derivative, which propagates differentials *backwards* from outputs to inputs.

### Key Idea: Generate code to compute $\langle \frac{\partial \mathcal{L}}{\partial f}, Df(\mathbf{x})\rangle$ rather than $\langle Df(\mathbf{x}),\mathbf{dx}\rangle$

The fundamental building blocks of reverse-mode derivatives are the **left-side inner product**. That is, the product of a vector of derivatives of w.r.t outputs $\frac{\partial \mathcal{L}}{\partial f}$ with the Jacobian matrix $Df(\mathbf{x})$.

An important thing to keep in mind is that it does not necessarily matter what the scalar quantity $\mathcal{L}$ is. The goal of this product is to propagate the derivatives of any scalar value $\mathcal{L}$ w.r.t output vector $f(\mathbf{x})$ (i.e., $\frac{\partial \mathcal{L}}{\partial f}$) into derivatives of that same scalar value $\mathcal{L}$ w.r.t the input vector $\mathbf{x}$ (i.e., $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$).

Here's an example of a Slang function computing the `reverse-mode derivative`.

```C
// Compute length of hypotenuse
float f(float x, float y)
{
    return sqrt(x * x + y * y);
}

// Reverse-mode derivative of f. dOutput represents the derivative dL/dOutput of the output w.r.t scalar value.
void rev_f(inout DifferentialPair<float> dpx, inout DifferentialPair<float> dpy, float dOutput)
{
    float x = dpx.getPrimal();
    float y = dpy.getPrimal();

    float t = 1.0 / (sqrt(x * x + y * y));

    dpx = DifferentialPair<float>(
        x,                 // The primal part of the return value is *always* copied in from the input as-is.
        dOutput * x * t);  // The differential part for x is the derivative dL/dx computed as 
                           // (dL/dOutput) * (dOutput/dx), where dOutput/dx = x / sqrt(x*x+y*y).

    dpy = DifferentialPair<float>(
        y,                
        dOutput * y * t);  // The differential part for y is the derivative dL/dy computed as 
                           // (dL/dOutput) * (dOutput/dy), where dOutput/dy = y / sqrt(x*x+y*y).
}
```

Note that `rev_f` accepts derivatives w.r.t the output value as the input, and returns derivatives w.r.t inputs as its output (through `inout` parameters). `rev_f` still needs the primal values `x` and `y` to compute the derivatives, so those are still passed in as an input through the primal part of the differential pair. 

Also note that the reverse-mode derivative function does not have to compute the primal result value (its return is void). The reason for this is a matter of convenience: reverse-mode derivatives are often invoked after all the primal functions, and there is typically no need for these values. We go into more detail on this topic in the checkpointing chapter.

The reverse mode function can be used to compute both `dOutput/dx` and `dOutput/dy` with a single invocation (unlike the forward-mode case where we had to invoke `fwd_f` once for each input)

```C
DifferentialPair<float> dpx = makePair<float>(x, 0.f); // Initialize diff-value to 0 (not necessary)
DifferentialPair<float> dpx = makePair<float>(y, 0.f); // Initialize diff-value to 0 (not necessary)

rev_f(dpx, dpy, 1.0); // Pass 1.0 for dL/dOutput so that the results are (1.0 * dOutput/dx) and (1.0 * dOutput/dy)

float doutput_dx = dpx.getDifferential(); 
float doutput_dy = dpy.getDifferential();
```

### Extension to multiple outputs
The extension to multiple outputs is fairly natural. Each output gets a separate input for its derivative.
Here is an example:
```C
// Computation involving multiple inputs and outputs.
float2 f_multi_output(float x, float y)
{
    return float2(
        x * x,
        x + y);
}

// Reverse-mode derivative of 'f_multi_output'. The derivative of the outputs is also a vector quantity 
// (type follows from return type of f_multi_output)
void rev_f_multi_output(DifferentialPair<float> dpx, DifferentialPair<float> dpy, float2 dOut)
{
    float x = dpx.getPrimal();
    float y = dpy.getPrimal();

    dpx = DifferentialPair<float>(x, dOut[0] * 2 * x + dOut[1]);
    dpy = DifferentialPair<float>(x, dOut[1]);
}
```

### Jacobian method: Generate forward- and reverse-mode derivatives from first principles.
A simple way to figure out what the generated reverse (or forward) derivative function is supposed to compute is to write down the entire Jacobian function. That is, write down the partial derivative of each input w.r.t each output

$$
D\mathbf{f}(\mathbf{x}) = \begin{bmatrix} 
\partial f_0 / \partial x & \partial f_0 / \partial y \\  
\partial f_1 / \partial x & \partial f_1 / \partial y \\
\end{bmatrix} = 
\begin{bmatrix} 
2x    & 0.0 \\  
1.0   & 1.0 \\
\end{bmatrix}
$$

The **reverse-mode derivative**'s outputs should match the left-product of this matrix with the vector of derivatives w.r.t outputs:

$$ \left\langle \frac{\partial \mathcal{L}}{\partial \mathbf{f}}, D\mathbf{f}(\mathbf{x})\right\rangle  = 
\begin{bmatrix}
\frac{\partial \mathcal{L}}{\partial f_0} & \frac{\partial \mathcal{L}}{\partial f_1}
\end{bmatrix}
\begin{bmatrix} 
2x    & 0.0 \\  
1.0   & 1.0 \\
\end{bmatrix} = 
\begin{bmatrix} \left(\frac{\partial \mathcal{L}}{\partial f_0} \cdot 2x + \frac{\partial \mathcal{L}}{\partial f_1}\right) & \frac{\partial \mathcal{L}}{\partial f_1} \end{bmatrix}
$$

and the **forward-mode derivative**'s outputs should match the right-product of this matrix with the vector of differentials of the inputs:

$$ \langle D\mathbf{f}(\mathbf{x}), d\mathbf{x}\rangle  = 
\begin{bmatrix} 
2x    & 0.0 \\  
1.0   & 1.0 \\
\end{bmatrix}
\begin{bmatrix}
dx \\ dy
\end{bmatrix} = 
\begin{bmatrix} 2x \cdot dx & dx + dy \end{bmatrix}
$$

Note that when we generate derivative code in practice, we do not materialize the full Jacobian matrix, and instead use the composition property to chain together derivatives at the instruction level. 
However, the resulting code is equivalent to the Jacobian method (mathematically), and it is a good, analytical way to confirm that the generated code is indeed correct (or when thinking about what the derivative of a particular instruction/set of instructions should be)


### Building Blocks: Reverse-mode derivatives compose in reverse order of execution.
A consequence of using the 'left-side inner product' is that derivatives of a composite function must be computed in the reverse of the order of primal computation.

Here's an example of a composite function $h$ (similar to the example used in forward-mode building blocks):

$$ h(\mathbf{x}) = f(g(\mathbf{x})) $$

where (for brevity):

$$ \mathbf{y} = g(\mathbf{x}) $$

The reverse-mode derivative function for $h$ can be written as the composition of the reverse-mode derivatives of $f$ and $g$

$$ \left\langle \frac{\partial L}{\partial h}, Dh(\mathbf{x})\right\rangle  = \left\langle \left\langle \frac{\partial L}{\partial h}, Df(\mathbf{y})\right\rangle , Dg(\mathbf{x})\right\rangle $$

Note the 'backward' order here. We must first pass the derivatives through the outer function $f$, and then pass the result through the inner function $g$ to compute derivatives w.r.t inner-most inputs $\mathbf{x}$. This process of passing derivatives backwards is often referred to as *backpropagation*.

A more concrete Slang example of the same:

```C
// Compute square
float sqr(float x)
{
    return x * x;
}

// Compute length of hypotenuse
float f(float x, float y)
{
    return sqrt(sqr(x) + sqr(y));
}
```

The derivative functions are then:
```C
void rev_sqr(DifferentialPair<float> dpx, float dOutput)
{
    float x = dpx.getPrimal();

    dpx = DifferentialPair<float>(x, dOutput * 2 * x);
}

void rev_f(DifferentialPair<float> dpx, DifferentialPair<float> dpy, float dOut)
{
    float t = 0.5f / sqrt(x * x + y * y);
    
    float d_xsqr = t * dOut; // Calculate derivatives w.r.t output of sqr(x)
    float d_ysqr = t * dOut; // Calculate derivatives w.r.t output of sqr(y)

    rev_sqr(dpx, d_xsqr); // Propagate to x
    rev_sqr(dpx, d_ysqr); // Propagate to y
}
```

When comparing `rev_f`'s implementation to `fwd_f`, note the order of computing derivative w.r.t `sqr` (in `rev_f`, `rev_sqr` is called at the end, while in `fwd_f` it is called at the beginning)

