"""
This script will generate input-out plots for all of the activation
functions. These are for use in the documentation, and potentially in
online tutorials.
"""

import os.path
import torch.nn.modules.activation
import torch.autograd
import matplotlib

matplotlib.use('Agg')

import pylab


# Create a directory for the images, if it doesn't exist
ACTIVATION_IMAGE_PATH = os.path.join(
    os.path.realpath(os.path.join(__file__, "..")),
    "activation_images"
)

if not os.path.exists(ACTIVATION_IMAGE_PATH):
    os.mkdir(ACTIVATION_IMAGE_PATH)

# In a refactor, these ought to go into their own module or entry
# points so we can generate this list programmaticly
functions = [
    'ELU',
    'Hardshrink',
    'Hardtanh',
    'LeakyReLU',  # Perhaps we should add text explaining slight slope?
    'LogSigmoid',
    'PReLU',
    'ReLU',
    'ReLU6',
    'RReLU',
    'SELU',
    'SiLU',
    'CELU',
    'GELU',
    'Sigmoid',
    'Softplus',
    'Softshrink',
    'Softsign',
    'Tanh',
    'Tanhshrink'
    # 'Threshold'  Omit, pending cleanup. See PR5457
]


def plot_function(function, **args):
    """
    Plot a function on the current plot. The additional arguments may
    be used to specify color, alpha, etc.
    """
    xrange = torch.arange(-7.0, 7.0, 0.01)  # We need to go beyond 6 for ReLU6
    pylab.plot(
        xrange.numpy(),
        function(torch.autograd.Variable(xrange)).data.numpy(),
        **args
    )


# Step through all the functions
for function_name in functions:
    plot_path = os.path.join(ACTIVATION_IMAGE_PATH, function_name + ".png")
    if not os.path.exists(plot_path):
        function = torch.nn.modules.activation.__dict__[function_name]()

        # Start a new plot
        pylab.clf()
        pylab.grid(color='k', alpha=0.2, linestyle='--')

        # Plot the current function
        plot_function(function)

        # The titles are a little redundant, given context?
        pylab.title(function_name + " activation function")
        pylab.xlabel("Input")
        pylab.ylabel("Output")
        pylab.xlim([-7, 7])
        pylab.ylim([-7, 7])

        # And save it
        pylab.savefig(plot_path)
        print('Saved activation image for {} at {}'.format(function, plot_path))
