from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
import glob
import json
import numpy as np

example_files = glob.glob('example_*.c2s')

for ex in example_files:
    print('Running example file', ex)
    with open(ex, 'r') as f:
        inits = json.loads(f.readline())
        net_name = f.readline().strip()
        outputs = json.loads(f.readline())

        CU = core.C.CompilationUnit()
        CU.define(f.read())

    # Initialize workspace with required inputs
    for name, shape, dt in inits:
        workspace.FeedBlob(name, np.random.rand(*shape).astype(np.dtype(dt)))

    net = CU.create_net(net_name)
    net.run()

    print('Success! Interesting outputs:')
    for output in outputs:
        print(output, workspace.FetchBlob(output))
