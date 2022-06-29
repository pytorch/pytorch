

class A():
    def __init__(self):
        self.a = 1

    def printa(self):
        print(f"print {self.a}")


class B():
    def __init__(self, classA):
        self.a = 2
        self.classA = classA

    def printa(self):
        return self.classA.__class__.printa(self)

a = A()
b = B(a)
b.printa()
