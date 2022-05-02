# AutoDiff Lab Derivatives
### dL/dq
$$
\frac{\partial L}{\partial q}  = \frac{\partial L}{\partial o} \frac{\partial o}{\partial q}  + \frac{\partial L}{\partial a}\frac{\partial a}{\partial q}
$$

$$
\frac{\partial L}{\partial q}  = \frac{\partial L}{\partial o} \frac{\partial o}{\partial a} \frac{\partial a}{\partial x}\frac{\partial x}{\partial q}  + \frac{\partial L}{\partial a} \frac{\partial a}{\partial x}\frac{\partial x}{\partial q}
$$

$$
\frac{\partial L}{\partial q}  = \frac{\partial L}{\partial o}\times v^\intercal \odot (1 - tanh^2(x)) \times k + \frac{\partial L}{\partial a}\odot (1 - tanh^2(x)) \times k
$$

### dL/dk

$$
\frac{\partial L}{\partial k}  = \frac{\partial L}{\partial o} \frac{\partial o}{\partial k}  + \frac{\partial L}{\partial a}\frac{\partial a}{\partial k}
$$

$$
\frac{\partial L}{\partial k}  = \frac{\partial L}{\partial o} \frac{\partial o}{\partial a} \frac{\partial a}{\partial x}\frac{\partial x}{\partial k}  + \frac{\partial L}{\partial a} \frac{\partial a}{\partial x}\frac{\partial x}{\partial k}
$$



$$
\frac{\partial L}{\partial k}  = (q^\intercal \times \frac{\partial L}{\partial o}\times v^\intercal \odot (1 - tanh^2(x)))^\intercal + (q^\intercal \times \frac{\partial L}{\partial a}\odot (1 - tanh^2(x)))^\intercal
$$

$$
\frac{\partial L}{\partial k}  = (\frac{\partial L}{\partial o}\times v^\intercal \odot (1 - tanh^2(x)))^\intercal \times q + (\frac{\partial L}{\partial a}\odot (1 - tanh^2(x)))^\intercal \times q
$$

$$
\frac{\partial L}{\partial k}  = (1 - tanh^2(x))^\intercal \odot v \times (\frac{\partial L}{\partial o})^\intercal \times q + (1 - tanh^2(x))^\intercal \odot (\frac{\partial L}{\partial a})^\intercal \ q
$$


### dL/dv
$$
\frac{\partial L}{\partial v}  = \frac{\partial L}{\partial o} \frac{\partial o}{\partial v}  + \frac{\partial L}{\partial a}\frac{\partial a}{\partial v}
$$

$$
\frac{\partial L}{\partial v}  = \frac{\partial L}{\partial o} \frac{\partial o}{\partial v}
$$

$$
\frac{\partial L}{\partial v}  = a^\intercal \cdot \frac{\partial L}{\partial o}
$$
