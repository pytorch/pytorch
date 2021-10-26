For each, the full derivation is in the respective `grad_wrt_*.md` file.
## Derivative wrt q
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

## Derivative wrt k
$$\frac{\partial L}{\partial k_{i,j}}
= \sum_n\sum_\alpha \frac{\partial L}{\partial o_{\alpha, n}}* v_{i,n}*\left(1-\tanh^2\left(\sum_zq_{\alpha,z}*k_{i, z}\right)\right)q_{\alpha, j}
+ \sum_\gamma \frac{\partial L}{\partial a_{\gamma, i}}\left(1 -\tanh^2\left(\sum_z q_{\gamma,z}*k_{i, z}\right)\right)q_{\gamma, j} $$

## Derivative wrt v
$$\frac{\partial L}{\partial v_{i,j}}
= \sum_m \frac{\partial L}{\partial o_{m, j}} * a_{m, i}$$
