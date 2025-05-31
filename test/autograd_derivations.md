$$B = QK^T \quad A = \tanh(B) \quad O = AV \quad L = \text{sum}(O) + \text{sum}(A)$$

### $\frac{\partial L}{\partial Q}$

$$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial O} \cdot{} \frac{\partial O}{\partial A} \cdot{} \frac{\partial A}{\partial B} \cdot \frac{\partial B}{\partial Q} +  \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial B} \cdot \frac{\partial B}{\partial Q}$$

$$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial O} \cdot \frac{\partial AV}{\partial A} \cdot \frac{\partial \tanh(B)}{\partial B} \cdot \frac{\partial QK^T}{\partial Q} + \frac{\partial L}{\partial A} \cdot \frac{\partial \tanh(B)}{\partial B} \cdot \frac{\partial QK^T}{\partial Q}$$

$$\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial O} \cdot V^T \odot (1-\tanh^2(QK^T)) \cdot K + \frac{\partial L}{\partial A} \odot (1-\tanh^2(QK^T)) \cdot K$$

### $\frac{\partial L}{\partial K}$
$$\frac{\partial L}{\partial K} = \frac{\partial L}{\partial O} \cdot{} \frac{\partial O}{\partial A} \cdot{} \frac{\partial A}{\partial B} \cdot \frac{\partial B}{\partial K} +  \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial B} \cdot \frac{\partial B}{\partial K}$$

$$\frac{\partial L}{\partial K} = \frac{\partial L}{\partial O} \cdot \frac{\partial AV}{\partial A} \cdot \frac{\partial \tanh(B)}{\partial B} \cdot \frac{\partial QK^T}{\partial K} + \frac{\partial \tanh(B)}{\partial B} \cdot \frac{\partial QK^T}{\partial K}$$

$$\frac{\partial L}{\partial K} = \left[ \frac{\partial L}{\partial O} \cdot V^T \odot (1-\tanh^2(QK^T)) \right]^T \cdot Q + \left[ \frac{\partial L}{\partial A} \cdot (1-\tanh^2(QK^T)) \right]^T \cdot Q$$

### $\frac{\partial L}{\partial V}$

$$ \frac{\partial L}{\partial V} = \frac{\partial L}{\partial O} \cdot \frac{\partial O}{\partial V}$$

$$ \frac{\partial L}{\partial V} =  \frac{\partial L}{\partial O} \cdot \frac{\partial AV}{\partial V}$$

$$ \frac{\partial L}{\partial V} = A^T \cdot \frac{\partial L}{\partial O}$$

$$ \frac{\partial L}{\partial V} = \tanh(QK^T)^T \cdot \frac{\partial L}{\partial O}$$