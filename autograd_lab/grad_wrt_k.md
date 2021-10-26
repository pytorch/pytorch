## Partial wrt k
Okay we're starting it off in the same way
$$\frac{\partial L}{\partial k_{i,j}}
= \sum_m\sum_n \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial k_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial k_{i,j}} $$

$$\frac{\partial L}{\partial k_{i,j}}
= \sum_m\sum_n\sum_f\sum_g \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial a_{f,g}}\frac{\partial a_{f,g}}{\partial k_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial k_{i,j}} $$

Oh, ho! We've seen part of this before. We can once again use Richard's derivation of the matmul partial
$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_f\sum_g \frac{\partial L}{\partial o_{\alpha, n}}* v_{g,n}*\frac{\partial a_{f,g}}{\partial k_{i,j}}
+ \sum_b\sum_c \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial k_{i,j}} $$

### Noticing reuse again
Okay, we once again notice that we're doing a full summation in both terms, so we will be doing substitution in parallel

$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_f\sum_g\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}* v_{g,n}*\frac{\partial a_{f, g}}{\partial x_{\alpha,\beta}}\frac{\partial x_{\alpha,\beta}}{\partial k_{i,j}}
+ \sum_b\sum_c\sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial x_{\gamma,\delta}}\frac{\partial x_{\gamma,\delta}}{\partial k_{i,j}} $$
Since `tanh` is a pointwise function,

$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_f\sum_g\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}* v_{g,n}*\frac{\partial [\tanh(x_{f, g})]}{\partial x_{\alpha,\beta}}\frac{\partial x_{\alpha,\beta}}{\partial k_{i,j}}
+ \sum_b\sum_c\sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{b, c}}\frac{\partial [\tanh(x_{b, c})]}{\partial x_{\gamma,\delta}}\frac{\partial x_{\gamma,\delta}}{\partial k_{i,j}} $$
We'll notice that this is only non-zero when $(f,g) = \alpha, \beta$ or $(b,c) = \gamma, \delta$.

$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}* v_{\beta,n}*\left(1-\tanh^2(x_{\alpha, \beta})\right)\frac{\partial x_{\alpha,\beta}}{\partial k_{i,j}}
+ \sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{\gamma, \delta}}\left(1 -\tanh^2(x_{\gamma, \delta})\right)\frac{\partial x_{\gamma,\delta}}{\partial k_{i,j}} $$
Since we transpose k, we get that
$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}* v_{\beta,n}*\left(1-\tanh^2(x_{\alpha, \beta})\right)\frac{\partial x_{\alpha,\beta}}{\partial k_{i,j}}
+ \sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{\gamma, \delta}}\left(1 -\tanh^2(x_{\gamma, \delta})\right)\frac{\partial x_{\gamma,\delta}}{\partial k_{i,j}} $$

We'll then notice that $x_{\alpha, \beta} = \sum_z q_{\alpha,z}*k_{\beta, z}$ and similarly for $\gamma$ and $\delta$.
$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_\alpha\sum_\beta \frac{\partial L}{\partial o_{\alpha, n}}* v_{\beta,n}*\left(1-\tanh^2\left(\sum_zq_{\alpha,z}*k_{\beta, z}\right)\right)\frac{\partial \sum_z[q_{\alpha, z}*k_{\beta, z}]}{\partial k_{i,j}}\\
+ \sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{\gamma, \delta}}\left(1 -\tanh^2\left(\sum_z q_{\gamma,z}*k_{\delta, z}\right)\right)\frac{\partial \sum_z[q_{\gamma, z}*k_{\delta, z}]}{\partial k_{i,j}} $$

We'll notice that this replaced terms are each now only non-zero when $(\beta, z) == (i, j)$ or $(\delta, z) == (i, j)$, respectively
$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_\alpha \frac{\partial L}{\partial o_{\alpha, n}}* v_{i,n}*\left(1-\tanh^2\left(\sum_zq_{\alpha,z}*k_{i, z}\right)\right)q_{\alpha, j}
+ \sum_\gamma \frac{\partial L}{\partial a_{\gamma, i}}\left(1 -\tanh^2\left(\sum_z q_{\gamma,z}*k_{i, z}\right)\right)q_{\gamma, j} $$

We'll break this up into two terms again. For the first one
$$\sum_n\sum_\alpha \frac{\partial L}{\partial o_{\alpha, n}}* v_{i,n}*\left(1-\tanh^2\left(\sum_zq_{\alpha,z}*k_{i, z}\right)\right)q_{\alpha, j}$$

The first two terms are the only ones that use $n$ and they are the $(\alpha, i)$th element of a matmul between o's upstream partial and `v.transpose(0,1)`. Then, inside the `tanh` we are computing the $(\alpha, i)$th element of a matmul between $q$ and `k.transpose(0,1)`. Since they are both computing the $(\alpha, i)$th element, we can see we are doing an elementwise multiply between them. Then, because we're summing over the alpha, if we consider the elementwise multiplied parts as one unit, we are doing a matmul between the unit, transposed and $q$. Giving us:
`matmul((matmul(grad_o, v.transpose(0,1))*(1-tanh(matmul(q, k.transpose(0,1)))**2)).transpose(0,1), q)`

For the second one

$$\sum_\gamma\frac{\partial L}{\partial a_{\gamma, i}}\left(1 -\tanh^2\left(\sum_z q_{\gamma,z}*k_{i, z}\right)\right)q_{\gamma, j}$$

We again see that the inside of the `tanh` is the $(\gamma, i)$th element of a matmul between `q` and `k.transpose(0,1)`. Since we are also getting the $(\gamma, i)$th element of the partial of L with respect to $a$, we can do an elementwise multiply between these and the matmul. If we consider this as a unit, the sum across the remaining $\gamma$ equates to the $(i,j)$th element of a matmul of the unit, transposed, and $q$. Giving us:\
`matmul((grad_a * (1-tanh(matmul(q,k.transpose(0,1)))**2)).transpose(0,1), q)`

For the second one

$$\sum_\gamma\frac{\partial L}{\partial a_{\gamma, i}}\left(1 -\tanh^2\left(\sum_z q_{\gamma,z}*k_{i, z}\right)\right)q_{\gamma, j}$$

We again see that the inside of the `tanh` is the $(\gamma, i)$th element of a matmul between `q` and `k.transpose(0,1)`. Since we are also getting the $(\gamma, i)$th element of the partial of L with respect to $a$, we can do an elementwise multiply between these and the matmul. If we consider this as a unit, the sum across the remaining $\gamma$ equates to the $(i,j)$th element of a matmul of the unit, transposed, and $q$. Giving us:\
`matmul((grad_a * (1-tanh(matmul(q,k.transpose(0,1)))**2)).transpose(0,1), q)`

So, overall, we have
```
grad_k =
  matmul((matmul(grad_o, v.transpose(0,1))*(1-tanh(matmul(q, k.transpose(0,1)))**2)).transpose(0,1), q)
  + matmul((grad_a * (1-tanh(matmul(q,k.transpose(0,1)))**2)).transpose(0,1), q)
```

Again we know that the `tanh` value is `a` so this can simplify to

`grad_k = matmul((matmul(grad_o, v.transpose(0,1))*(1-a**2)).transpose(0,1), q) + matmul((grad_a * (1-a**2)).transpose(0,1), q)`
