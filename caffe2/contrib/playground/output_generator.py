




from caffe2.python import timeout_guard

def fun_conclude_operator(self):
    # Ensure the program exists. This is to "fix" some unknown problems
    # causing the job sometimes get stuck.
    timeout_guard.EuthanizeIfNecessary(600.0)


def assembleAllOutputs(self):
    output = {}
    output['train_model'] = self.train_model
    output['test_model'] = self.test_model
    output['model'] = self.model_output
    output['metrics'] = self.metrics_output
    return output
