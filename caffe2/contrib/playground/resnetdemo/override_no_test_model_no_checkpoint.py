from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def checkpoint(self, epoch):
    self.model_path = None
    pass

def prep_data_parallel_models(self):
    # only do train_model no test needed here
    self.prep_a_data_parallel_model(self.train_model,
                                    self.train_dataset, True)

def run_testing_net(self):
    pass
