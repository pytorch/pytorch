
class CWrapPlugin(object):

    def initialize(self, cwrap):
        pass

    def get_type_check(self, arg, option):
        pass

    def get_type_unpack(self, arg, option):
        pass

    def get_return_wrapper(self, option):
        pass

    def get_wrapper_template(self, declaration):
        pass

    def get_arg_accessor(self, arg, option):
        pass

    def process_full_file(self, code):
        return code

    def process_single_check(self, code, arg, arg_accessor):
        return code

    def process_all_checks(self, code, option):
        return code

    def process_single_unpack(self, code, arg, arg_accessor):
        return code

    def process_all_unpacks(self, code, option):
        return code

    def process_option_code(self, code, option):
        return code

    def process_wrapper(self, code, declaration):
        return code

    def process_declarations(self, declarations):
        return declarations

    def process_option_code_template(self, template, option):
        return template


from .StandaloneExtension import StandaloneExtension
from .NullableArguments import NullableArguments
from .OptionalArguments import OptionalArguments
from .ArgcountChecker import ArgcountChecker
from .ArgumentReferences import ArgumentReferences
from .BeforeAfterCall import BeforeAfterCall
from .ConstantArguments import ConstantArguments
from .ReturnArguments import ReturnArguments
from .GILRelease import GILRelease
from .AutoGPU import AutoGPU
from .CuDNNPlugin import CuDNNPlugin
from .GenericNN import GenericNN
