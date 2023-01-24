## Differential Privacy with ResNet18

### Differential Privacy
Differential privacy is a way of training models that ensures no attacker can figure out the training
data from the gradient updates of the model. Recently, a paper was published comparing the performance of
Opacus to a JAX-based system.

[Original differential privacy paper](https://people.csail.mit.edu/asmith/PS/sensitivity-tcc-final.pdf)
[JAX-based differential privacy paper](https://arxiv.org/pdf/2010.09063.pdf)

### Opacus
Opacus is a differential privacy library built for PyTorch. They have added hooks to PyTorch's
autograd that compute per sample gradients and a differential privacy engine that computes
differentially private weight updates.

### Example
This example runs ResNet18 by either having Opacus compute the differentially private updates or
getting the per sample gradients using vmap and grad and computing the differentially private update
from those.

As a caveat, the transforms version may not be computing the exact same values as the opacus version.
No verification has been done yet for this.

### Requirements
These examples use Opacus version 1.0.1 and torchvision 0.11.2
