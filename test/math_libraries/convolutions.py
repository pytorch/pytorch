import torch
from torch.backends import cudnn
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    dtypes, dtypesIfCUDA, skipCUDAIfRocm, onlyOnCPUAndCUDA
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests
import os
import json
import warnings

class TestConvExt(NNTestCase):
    def _subtensor(self, tensor, dim, group, g):
        if tensor is None:
            return None
        group_size = int(tensor.size(dim) / group)
        return tensor.narrow(dim, group_size * g, group_size).contiguous()

    def _thnn_conv(self, input, weight, k, bias, s, p, d):
        if d[0] > 1 or d[1] > 1:
            return torch._C._nn.slow_conv_dilated2d(input, weight, k, bias, s, p, d)
        else:
            return torch._C._nn.thnn_conv2d(input, weight, k, bias, s, p)

    def _thnn_conv_group(self, input, weight, k, bias, s, p, d, group):
        if group == 1:
            return self._thnn_conv(input, weight, k, bias, s, p, d)
        else:
            outputs = []
            for g in range(group):
                input_g = self._subtensor(input, 1, group, g)
                weight_g = self._subtensor(weight, 0, group, g)
                bias_g = self._subtensor(bias, 0, group, g)
                outputs.append(self._thnn_conv(input_g, weight_g, k, bias_g, s, p, d))
            return torch.cat(outputs, 1)

    def _collect_cases(self):
        dir_name = 'convolutions_cases'
        cur_dir = os.path.split(os.path.realpath(__file__))[0]
        case_files = ['googlenet_v3', 
                      'maskrcnn_p1', 
                      'mobilenet', 
                      'resnet_50']
        for index, case_file in enumerate(case_files):
            case_files[index] = '{}/{}/shapes_{}.json'.format(cur_dir, dir_name, case_file)
        total_cases = list()
        for file_name in case_files:
            with open(file_name) as f:
                model_cases = json.loads(f.read())
            total_cases += model_cases
        return total_cases

    # Set multiple situations to improve convolution2d test coverage
    @onlyOnCPUAndCUDA
    @skipCUDAIfRocm
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    def test_conv2d_ext(self, device, dtype):
        # this list for save the cases which dh,dw == 0
        # and raise error in the end
        self.Fail = list()

        total_cases = self._collect_cases()
        for case in total_cases:
            case_name = case['case_name']
            bs = case['mb']
            group = case['g']
            ic, ih, iw = case['ic'], case['ih'], case['iw']
            oc = case['oc']
            kh, kw = case['kh'], case['kw']
            sh, sw = case['sh'], case['sw']
            ph, pw = case['ph'], case['pw']
            dh, dw = case['dh'], case['dw']
            has_bias = case['bias']
            if dh == 0 or dw == 0:
                self.Fail.append(case_name)
                continue

            ic_g = ic // group
            torch.manual_seed(1)
            input = torch.randn((bs, ic, ih, iw), device=device, dtype=dtype, requires_grad=True)
            weight = torch.randn((oc, ic_g, kh, kw), device=device, dtype=dtype) * 0.01
            bias = None if has_bias == 'False' else torch.randn((oc), device=device, dtype=dtype)

            k = [kh, kw]
            s = [sh, sw]
            p = [ph, pw]
            d = [dh, dw]
            thnn_output = self._thnn_conv_group(input, weight, k, bias, s, p, d, group)
            if self.device_type == 'cpu' and torch.backends.mkldnn.is_available():
                output = torch.mkldnn_convolution(input, weight, bias, p, s, d, group)
            elif self.device_type == 'cuda' and torch.backends.cudnn.is_available():
                output = torch.cudnn_convolution(input, weight, bias, p, s, d, group, True, True)
            else:
                output = torch.conv2d(input, weight, bias, s, p, d, group)

            msg = 'device:{}, dtype:{}, group:{}, batchsize:{}' \
                  'input channel:{}, output channel:{}, ' \
                  'bias:{}, padding:{}, dilation:{}, stride:{}, ' \
                  'kernel:{}'
            msg = msg.format(device, dtype, group, bs, ic, oc, has_bias, p, d, s, k)

            if self.device_type == 'cuda' and cudnn.is_available():
                self.assertEqual(
                    output, thnn_output,
                    msg=msg,
                    atol=1e-2, rtol=1e-2
                )
            else:
                self.assertEqual(
                    output, thnn_output,
                    msg=msg
                )

        if self.Fail != []: 
            warnings.warn('invalid cases dilation height or weight is 0: ' + ",".join(self.Fail))

instantiate_device_type_tests(TestConvExt, globals())

if __name__ == '__main__':
    run_tests()
