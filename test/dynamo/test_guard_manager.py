from torch._C._dynamo import guards


const_guard = guards.PythonLambdaGuard(lambda x: x == 5)
print(const_guard(4))
print(const_guard(5))


guard_manager = guards.GuardManager()
guard_manager.add_lambda_guard(lambda x: isinstance(x, int))
guard_manager.add_lambda_guard(lambda x: x >= 5)

print([guard_manager.check(i) for i in range(4, 8)])


guard_manager.animesh.add_lambda_guard(lambda x: isinstance(x, int))
guard_manager.animesh.add_lambda_guard(lambda x: x <= 10)
guard_manager.animesh.add_lambda_guard(lambda x: x >= 5)
print(guard_manager.animesh.check(9), guard_manager.animesh.check(11))


guard_manager["foo"].add_lambda_guard(lambda x: isinstance(x, int))
guard_manager["foo"].add_lambda_guard(lambda x: x <= 10)
guard_manager["foo"].add_lambda_guard(lambda x: x >= 5)
print(guard_manager["foo"].check(9), guard_manager["foo"].check(11))


class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y


f_locals = {
    "foo": 5,
    "bar": Pair(1, 2),
}
guard_manager = guards.GuardManager()

guard_manager["foo"].add_lambda_guard(lambda x: isinstance(x, int))
guard_manager["foo"].add_lambda_guard(lambda x: x == 5)

guard_manager["bar"].add_lambda_guard(lambda x: isinstance(x, Pair))
guard_manager["bar"].x.add_lambda_guard(lambda x: isinstance(x, int))
guard_manager["bar"].x.add_lambda_guard(lambda x: x == 1)
guard_manager["bar"].y.add_lambda_guard(lambda x: isinstance(x, int))
guard_manager["bar"].y.add_lambda_guard(lambda x: x == 2)

f_locals1 = {
    "foo": 6,
    "bar": Pair(1, 2),
}

print(guard_manager.check(f_locals), guard_manager.check(f_locals1))
