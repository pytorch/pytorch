## Partial wrt v
Okay we've done this twice, let's start again:
$$\frac{\partial L}{\partial v_{i,j}}
= \sum_m\sum_n \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial v_{i,j}}
+ \sum_b\sum_c\sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial v_{i,j}} $$

$$\frac{\partial L}{\partial v_{i,j}}
= \sum_m\sum_n \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial v_{i,j}}
+ \sum_b\sum_c\sum_\gamma\sum_\delta \frac{\partial L}{\partial a_{b, c}}\frac{\partial a_{b, c}}{\partial x_{\gamma, \delta}}\frac{\partial x_{\gamma, \delta}}{\partial v_{i,j}} $$

Oh! There's something interesting here. $x$ does not depend on $v$ at all, so that derivative is 0. So we only depend on one:
$$\frac{\partial L}{\partial v_{i,j}}
= \sum_m\sum_n \frac{\partial L}{\partial o_{m, n}} \frac{\partial o_{m, n}}{\partial v_{i,j}}$$

Okay so $o$ is just a matmul between $a$ and $v$ and as we saw earlier, $a$ does not depend on $v$. So,
$$\frac{\partial L}{\partial v_{i,j}}
= \sum_m\sum_n \frac{\partial L}{\partial o_{m, n}} \frac{\partial \sum_z[a_{m, z}*v_{z,n}]}{\partial v_{i,j}}$$

One final time, we notice that this is only non-zero when $(z,b) == i,j$ so
$$\frac{\partial L}{\partial v_{i,j}}
= \sum_m \frac{\partial L}{\partial o_{m, j}} * a_{m, i}$$
This is a matmul between $a$, transposed, and the upstream gradients with respect to $o$, or `matmul(a.transpose(0,1), grad_o)`
