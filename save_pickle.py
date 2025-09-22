# save_pickle.py
import pickle
from first import MyClass

obj = MyClass(42)

with open("obj.pkl", "wb") as f:
    pickle.dump(obj, f)
