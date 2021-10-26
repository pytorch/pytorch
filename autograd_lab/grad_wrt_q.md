## Partial wrt q
So we want the partial of `L` with respect to `q`. Since we have `L` with respect to both `o` and `a`, we can separate out the partial for `q`
$$\frac{\partial L}{\partial q_{i,j}}
= \sum_m\sum_n \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial q_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial q_{i,j}} $$

$$\frac{\partial L}{\partial q_{i,j}}
= \sum_m\sum_n\sum_f\sum_g \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial a_{f, g}}\frac{\partial a_{f, g}}{\partial q_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial q_{i,j}} $$

We'll start by expanding the partial with respect to `o`
$$\frac{\partial L}{\partial q_{i,j}}
= \sum_m\sum_n\sum_f\sum_g \frac{\partial L}{\partial o_{m, n}} \frac{\partial \sum_z[a_{m, z}*v_{z, n}]}{\partial a_{f, g}}\frac{\partial a_{f, g}}{\partial q_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial q_{i,j}} $$

We can replace `o` with the matmul formula and use the partial as derived in the other colab (thanks Richard)
$$\frac{\partial L}{\partial q_{i,j}}
= \sum_n\sum_f\sum_g \frac{\partial L}{\partial o_{\alpha, n}} * v_{g, n} *\frac{\partial a_{f, g}}{\partial q_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial q_{i,j}} $$

### Noticing reuse
We can then notice that because we are summing across all elements in both terms, we are just looking for $\frac{\partial a_{b, c}}{\partial q_{i,j}}$. So, we'll be doing this in parallel:

$$\frac{\partial L}{\partial q_{i,j}}
= \sum_n\sum_f\sum_g\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}
  * v_{g, n}
  * \frac{\partial a_{f, g}}{\partial x_{\alpha,\beta}}
  \frac{\partial x_{\alpha,\beta}}{\partial q_{i,j}}
+ \sum_b\sum_c\sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{b, c}}
  \frac{\partial a_{b, c}}{\partial x_{\gamma,\delta}}
  \frac{\partial x_{\gamma,\delta}}{\partial q_{i,j}}$$

Since `tanh` is a pointwise operation
$$\frac{\partial L}{\partial q_{i,j}}
= \sum_n\sum_f\sum_g\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}
  * v_{g, n}
  * \frac{\partial [\tanh(x_{f, g})]}{\partial x_{\alpha,\beta}}
  \frac{\partial x_{\alpha,\beta}}{\partial q_{i,j}}
+ \sum_b\sum_c\sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{b, c}}
  * \frac{\partial [\tanh(x_{b, c})]}{\partial x_{\gamma,\delta}}
  \frac{\partial x_{\gamma,\delta}}{\partial q_{i,j}}$$

Continuing, we can replace $x_{\alpha, \beta} = \sum_z q_{\alpha,z} * k^T_{z,\beta}$ or similar for $\gamma$ and $\delta$.

Specifically here, we'll want to notice that $k^T_{\alpha, \beta} = k_{\beta, \alpha}$, so this becomes $x_{\alpha, \beta} = \sum_z q_{\alpha, z} * k_{\beta, z}$, again the same logic running for $\gamma$ and $\delta$. Plugging these in,

\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial q_{i,j}} &= \\
&= \sum_n\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}
  * v_{\beta, n}
  \left(1-\tanh^2\left(\sum_z q_{\alpha, z} * k_{\beta, z}\right)\right)
  \frac{\partial \left[\sum_z q_{\alpha,z} * k_{\beta, z}\right]}{\partial q_{i,j}}\\
&+ \sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{\gamma, delta}}
  \left(1-\tanh^2\left(\sum_z q_{\gamma, z} * k_{\delta, z}\right)\right)
  \frac{\partial \left[\sum_z q_{\gamma,z} * k_{\delta, z}\right]}{\partial q_{i,j}}
\end{aligned}
\end{equation}

Following the same logic from Richard's matmul colab, the second factor will only be zero when $\alpha,z$ != $i,j$ or  \\
$\gamma, z$ != $i,j$

$$
\frac{\partial L}{\partial q_{i,j}}
= \sum_n\sum_\beta \frac{\partial L}{\partial o_{i, n}}
  * v_{\beta, n}
  \left(1-\tanh^2\left(\sum_z q_{i, z} * k_{\beta, z}\right)\right)
  k_{\beta, j}
+ \sum_\delta \frac{\partial L}{\partial a_{i, \delta}}
  * \left(1-\tanh^2\left(\sum_z q_{i, z} * k_{\delta, z}\right)\right)
  * k_{\delta, j}
$$

Let's look at each term separately.

$$\sum_n\sum_\beta \frac{\partial L}{\partial o_{i, n}}
  * v_{\beta, n}
  \left(1-\tanh^2\left(\sum_z q_{i, z} * k_{\beta, z}\right)\right)
  k_{\beta, j}$$

First, we can notice that inside the `tanh` functions, we are computing the (i, $\beta$)th value of the matmul of `q` and `k.transpose(0, 1)`. Since n is only used in the first two terms, we can also see that that is the (i, $\beta$)th element of a matmul between the partial with respect to `o` and `v.transpose(0, 1)`. Then, since both compute the `(i, g)`th element and `tanh` and squaring are both pointwise operations, we can consider that we are doing a pointwise multiply between them. Finally, if we consider that pointwise multiply as a unit, the remaining summation over g is equivalent to doing a matmul bewteen that unit and `k`. So, our formula in code for the first term is

`matmul(matmul(grad_o, v^T) * (1-tanh(matmul(q, k.transpose(0, 1)))**2), k)`

Now the second term
$$\sum_\delta \frac{\partial L}{\partial a_{i, \delta}}
  * \left(1-\tanh^2\left(\sum_z q_{i, z} * k_{\delta, z}\right)\right)
  * k_{\delta, j}$$

Following the same logic as above for the second term, we get the operand of tanh is computing the (i, $\delta$)th element of the matmul of `q` and `k.transpose(0,1)`. Similarly, since `tanh` is a pointwise operation and the partial of `L` with respect to `a` is also getting the (i, $\delta$)th element, we are doing a pointwise multiply of the upstream gradients with `(1-tanh^2(matmul(q, k.transpose(0, 1))`. Finally, with the upstream partial and the tanh taken as a unit, we can notice that this is a matmul between that unit and k, untransposed. In code, this looks like:

 `matmul(grad_a * (1-tanh(matmul(q, k.transpose(0, 1))**2), k)`

 Phew! So the final formula is

```
grad_q = matmul(matmul(grad_o, v.transpose(0,1)) * (1-tanh(q, k.transpose(0,1))**2), k) +
    matmul(grad_a * (1-tanh(q, k.transpose(0,1))**2), k)
```
Noticing some final reuse here, the `tanh` term is just `a`. So we get

`grad_q = matmul(matmul(grad_o, v.transpose(0,1)) * (1-a**2), k) + matmul(grad_a * (1-a**2), k)`
