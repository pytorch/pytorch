import torch
import pickle
import pickletools
r = [1, 2, (3.4, 4.4, 4.4, 5.) , { 'a' : 3, 'b' : 400000 }, [3, 4, 'a']]

d = pickle.dumps(r, protocol=2)

pickletools.dis(d)

torch._C.print_pickle(d)
