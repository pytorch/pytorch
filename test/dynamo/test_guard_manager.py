
import torch

from torch._C._dynamo import guards



const_guard = guards.PythonLambdaGuard(lambda x: x == 5)
print(const_guard(4))
print(const_guard(5))


guard_mananger = guards.GuardManager()
guard_mananger.add_lambda_guard(lambda x: isinstance(x, int))
guard_mananger.add_lambda_guard(lambda x: x >= 5)

print([guard_mananger.check(i) for i in range(4, 8)])


guard_mananger.animesh.add_lambda_guard(lambda x: isinstance(x, int))
guard_mananger.animesh.add_lambda_guard(lambda x: x <= 10)
guard_mananger.animesh.add_lambda_guard(lambda x: x >= 5)
print(guard_mananger.animesh.check(9))
