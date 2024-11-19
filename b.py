import torch


y = 0


def test_pep_479():
    def generator_with_stop_iteration():
        yield 1
        # Explicitly raising StopIteration inside the generator
        raise StopIteration("StopIteration raised within generator")
        yield 2  # This should never be reached


    @torch.compile(backend="eager", fullgraph=True)
    def fn(t):
        global y
        try:
            # Try to consume the generator
            gen = generator_with_stop_iteration()
            y += next(gen)
            y += next(gen)
        except RuntimeError as e:
            # Check that StopIteration was converted to RuntimeError
            return 100
        except StopIteration:
            return 200
        return y

    t = torch.randn(3)
    y = fn(t)
    print(y)

# Run the test case
test_pep_479()
