from torch.autograd import Function


class PixelShuffleFunction(Function):
    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor
        self.upscale_factor_squared = upscale_factor ** 2
        self._shuffle_out = None
        self._shuffle_in = None

    def forward(self, input):
        self.save_for_backward(input)
        shuffle_out = input.new()

        batch_size = input.size(0)
        channels = int(input.size(1) / self.upscale_factor_squared)
        in_height = input.size(2)
        in_width = input.size(3)

        input_view = input.view(batch_size, channels, self.upscale_factor,
                                self.upscale_factor, in_height, in_width)

        shuffle_out.resize_(input_view.size(0), input_view.size(1), input_view.size(4),
                            input_view.size(2), input_view.size(5), input_view.size(3))

        shuffle_out.copy_(input_view.permute(0, 1, 4, 2, 5, 3))

        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        output = shuffle_out.view(batch_size, channels, out_height, out_width)

        return output

    def backward(self, grad_output):
        input, = self.saved_tensors

        channels = int(input.size(1) / self.upscale_factor_squared)
        grad_output = grad_output.contiguous()
        go_view = grad_output.view(input.size(0), channels, input.size(2), self.upscale_factor,
                                   input.size(3), self.upscale_factor)

        shuffle_in = input.new().resize_(go_view.size(0), go_view.size(1), go_view.size(3),
                                         go_view.size(5), go_view.size(2), go_view.size(4))

        shuffle_in.copy_(go_view.permute(0, 1, 3, 5, 2, 4))

        return shuffle_in.view_as(input)
