import sys
from tools.cwrap import cwrap
from tools.cwrap.plugins import CWrapPlugin
from string import Template
import sys
import torch
from torch.autograd import Variable

def transform_defined_if(defined_if):
    if defined_if != None:
        defined_if = defined_if.replace('defined(TH_REAL_IS_FLOAT)', 'Float')
        defined_if = defined_if.replace('defined(TH_REAL_IS_DOUBLE)', 'Double')
        defined_if = defined_if.replace('defined(TH_REAL_IS_BYTE)', 'Byte')
        defined_if = defined_if.replace('defined(TH_REAL_IS_CHAR)', 'Char')
        defined_if = defined_if.replace('defined(TH_REAL_IS_INT)', 'Int')
        defined_if = defined_if.replace('defined(TH_REAL_IS_LONG)', 'Long')
        defined_if = defined_if.replace('defined(NUMPY_TYPE_ENUM)', 
                                        'Byte // Short // Int // Long // Float // Double')
        defined_if = defined_if.replace('CUDA_INT', 'Cuda_Int')
        defined_if = defined_if.replace('CUDA_LONG', 'Cuda_Long')
        defined_if = defined_if.replace('CUDA_FLOAT', 'Cuda_Float')
        defined_if = defined_if.replace('CUDA_DOUBLE', 'Cuda_Double')
        defined_if = defined_if.replace('CUDA_HALF', 'Cuda_Half')
        defined_if = defined_if.replace('!IS_CUDA', 'All CPU Types')
    else:
        defined_if = "All Types (CPU and CUDA)"
    defined_if = defined_if.replace('||', '//')
    return defined_if

class DocGen(CWrapPlugin):
    def __init__(self):
        self.declarations = {}

    def process_declarations(self, declarations):
        self.declarations.update({declaration['name']: declaration for declaration in declarations})
        # self.declarations += declarations
        return declarations

    def get_wrapper_template(self, declaration):
        return Template("")

    def get_type_check(self, arg, option):
        return Template("")

    def get_type_unpack(self, arg, option):
        return Template("")

    def get_return_wrapper(self, option):
        return Template("")

    def print_declarations(self):  
        print(" # torch.Tensor")
        for name, declarations in sorted(self.declarations.items()):
            if name.endswith('_') and name[:-1] in self.declarations:
                continue
            if not name.endswith('_') and name + '_' in self.declarations:
                inplace = True
            else:
                inplace = False

            pname = declarations['options'][0].get('python_name', None)
            if pname != None:
                name = pname
            if name.startswith('_'):
                continue

            # START PRINTING MARKDOWN
            print("## " + name + " \n")
            print("|    %-25s |    %-8s |    %-25s |" % ("Name", "Autograd", "defined if"))
            print("| " + ('-' * 28) + " | " + ('-' * 11) + " | "+ ('-' * 28) + " |")
            if inplace:
                sys.stdout.write("|    %-25s" % (name + '  //  ' + name + "_"))
            else:
                sys.stdout.write("|    %-25s" % name)
            sys.stdout.write(' | ')
            if hasattr(Variable(torch.randn(10)), name):
                sys.stdout.write(' %9s ' % 'yes') # + '   ' + name)
            else:
                sys.stdout.write(' %9s ' % 'no') # + '   ' + name)
            defined_if = declarations.get('defined_if', None)
            defined_if = transform_defined_if(defined_if)
            sys.stdout.write(' | ')
            sys.stdout.write(defined_if)
            sys.stdout.write(' |')
            sys.stdout.write('\n\n')
            #if inplace:
            #    print('Inplace Exists : True')
            #sys.stdout.write('Arguments  : ')

            args = declarations['options'][0]['arguments']
            if len(args) == 0:
                print(    '**No Arguments**\n' )
            else:
                print(    '**Arguments**\n' )
                print("|    %-15s |    %-12s |    %-15s |" % ("Name", "Type", "Default"))
                print("| " + ('-' * 18) + " | " + ('-' * 15) + " | "+ ('-' * 18) + " |")

                for arg in args:
                    type_ = arg['type']
                    if type_ == 'THGenerator*':
                        continue
                    if type_ == 'THTensor*':
                        type_ = 'Tensor'
                    if type_ == 'THIndexTensor*':
                        type_ = 'LongTensor'
                    if type_ == 'THBoolTensor*':
                        type_ = 'ByteTensor'
                    if type_ == 'THLongTensor*':
                        type_ = 'LongTensor'
                    if type_ == 'THLongStorage*':
                        type_ = 'LongStorage'
                    default = arg.get('default', None)
                    allocated = arg.get('allocate', None)
                    if default == None and allocated == None:
                        default = "     [required]"
                    elif allocated != None:
                        default = "     [optional]"
                    else:
                        default = str(default)
                        import re
                        m = re.search('\s*AS_REAL\((.+)\)\s*', default)
                        if m:
                            default = m.group(1)
                            default = default

                    print('| %15s    |  %12s   |   %10s |' % (arg['name'], type_, default))
                    # print(    'Options    : ' )
                    # print(declarations['options'][0])
                print('')
            if declarations['return']:
                return_ = declarations['return']
                if return_ == 'THTensor*':
                    return_ = 'Tensor'
                if return_ == 'void':
                    return_ = 'nothing'
                print(    '**Returns        : ' + return_ + '**')
            print('')


docs = DocGen()
cwrap('../../torch/csrc/generic/TensorMethods.cwrap', plugins=[docs])

docs.print_declarations()
