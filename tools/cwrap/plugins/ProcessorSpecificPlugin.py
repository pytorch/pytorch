from copy import deepcopy
from . import CWrapPlugin
import yaml


class ProcessorSpecificPlugin(CWrapPlugin):

    def process_declarations(self, declarations):
        # In order to move Torch's random functions into the same cwrap
        # declaration, we need to be able to handle the fact that on the CPU
        # these functions take a generator argument, while on the GPU, they
        # do not. As such, we would like to split those declarations at cwrap
        # runtime into two separate declarations, one for the CPU (unchanged),
        # and one for the GPU (with the generator argument removed).
        #
        # For example, the declaration arguments:
        # arguments:
        #   - THTensor* self
        #   - arg: THGenerator* generator
        #     default: THPDefaultGenerator->cdata
        #     kwarg_only: True
        #
        # Would have the generator argument removed when generating for the GPU
        # backend.

        def arg_contains_generator(arg):
            return (arg['type'] == 'THGenerator*' or (arg.get('default', None)
                    is not None and 'THPDefaultGenerator' in
                    str(arg.get('default', ""))))

        def split_candidate(declaration):
            # First, check and see if it is a declaration for both CPU/GPU
            if all([proc in declaration['backends'] for
                    proc in ['CPU', 'CUDA']]):
                for option in declaration['options']:
                    for argument in option['arguments']:
                        if arg_contains_generator(argument):
                            return True

            return False

        def can_we_handle_the_split(declaration):
            # hook into here if the split cannot happen for some reason
            return True

        def generator_split(declaration):
            # the split must make two changes: 1. remove the generator argument
            # for the GPU, and 2. assign the correct backends/types to the
            # split declaration
            dec_cpu = declaration
            dec_gpu = deepcopy(declaration)

            # Remove GPU backend and types from dec_cpu
            dec_cpu['backends'].remove('CUDA')
            if dec_cpu.get('backend_type_pairs', False):
                dec_cpu['backend_type_pairs'] = (
                    [pair for pair in dec_cpu['backend_type_pairs'] if
                     pair[1] == 'CPU'])
            # also need to reach into options
            for option in dec_cpu['options']:
                option['backends'].remove('CUDA')

            # Remove CPU backend and types from dec_gpu
            dec_gpu['backends'].remove('CPU')
            if dec_gpu.get('backend_type_pairs', False):
                dec_gpu['backend_type_pairs'] = (
                    [pair for pair in dec_gpu['backend_type_pairs'] if
                     pair[1] == 'CUDA'])
            # also need to reach into options
            for option in dec_gpu['options']:
                option['backends'].remove('CPU')

            # Remove generator arguments from dec_gpu options
            for option in dec_gpu['options']:
                option['arguments'] = (
                    [arg for arg in option['arguments'] if
                     not arg_contains_generator(arg)])

            return [dec_cpu, dec_gpu]

        decs = []
        for declaration in declarations:
            if split_candidate(declaration):
                assert(can_we_handle_the_split(declaration))
                newdecs = generator_split(declaration)
                decs.extend(newdecs)
            else:
                decs.append(declaration)

        return decs
