import torch
import numpy as np

t = torch.tensor([-4.8270-5.4215j, -7.5824-4.7842j, -0.6047-4.1508j, -1.5412+7.3808j,
        -1.9719+5.5761j, -4.1460-8.3157j], dtype=torch.complex128)

a = np.array([-4.8270-5.4215j, -7.5824-4.7842j, -0.6047-4.1508j, -1.5412+7.3808j,
        -1.9719+5.5761j, -4.1460-8.3157j], dtype=np.complex128)

print(t.numpy() == a)

print(torch.sin(t).numpy() == np.sin(a))

print(torch.sin(t.to(torch.complex64)).numpy() == np.sin(a.astype(np.complex64)))

print(torch.sin(t, dtype=torch.complex64).numpy() == np.sin(a, dtype=np.complex64))
