import torch

from ..utils import has_triton

if has_triton():

    import triton

    class _conv1x1:
        @staticmethod
        def _call(
            x,
            w,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ):
            # Q: should we check x, w, bias dtypes?
            device = x.device
            # input shapes
            shape_x = x.shape
            shape_w = w.shape
            shape_bias = bias.shape if bias is not None else None

            # indicies for the layout
            xn, xc, xh, xw = 0, 1, 2, 3
            yn, yc, yh, yw = 0, 1, 2, 3
            wn, wc, wh, ww = 0, 1, 2, 3

            # out_channel, in_channel, kernel_height, kernel_width
            kernel_size = [shape_w[wh], shape_w[ww]]
            input_size = [shape_x[xh], shape_x[xw]]
            assert (
                not shape_bias or shape_bias[0] == shape_w[wn]
            ), f"bias shape did not match{shape_bias} != {shape_w[wn]}"
            in_channel = shape_w[wc] * groups

            assert shape_x[xc] % groups == 0, "in_channels must be divisible by groups"
            assert shape_w[wn] % groups == 0, "out_channels must be divisible by groups"
            assert (
                shape_x[xc] == in_channel
            ), f"in_channel did not match {shape_x[xc]} != {in_channel}"

            assert (
                len(stride)
                == len(padding)
                == len(dilation)
                == len(output_padding)
                == len(kernel_size)
                == len(input_size)
            )

            # output shape
            shape_y = [0] * 4
            shape_y[yn] = shape_x[xn]
            shape_y[yc] = shape_w[wn]
            shape_y[yh] = (
                input_size[0]
                + 2 * padding[0]
                - dilation[0] * (kernel_size[0] - 1)
                - 1
                + stride[0]
            ) // stride[0] + 2 * output_padding[0]
            shape_y[yw] = (
                input_size[1]
                + 2 * padding[1]
                - dilation[1] * (kernel_size[1] - 1)
                - 1
                + stride[1]
            ) // stride[1] + 2 * output_padding[1]

            BATCH = shape_x[xn]
            IN_C = shape_x[xc]
            # IN_H = shape_x[xh]
            # IN_W = shape_x[xw]
            KERNEL_N = shape_w[wn]
            KERNEL_H = shape_w[wh]
            KERNEL_W = shape_w[ww]
            OUT_H = shape_y[yh]
            OUT_W = shape_y[yw]

            assert KERNEL_H == 1 and KERNEL_W == 1, "only support 1x1 conv"
            channels_last = x.stride()[1] == 1

            if padding == (0, 0):
                # nchw -> nhwc
                x = x.permute(0, 2, 3, 1)
                # select every stride's element (for stride > 1)
                x = x[:, :: stride[0], :: stride[1], :]
                # 2d matrix
                mat_x = x.reshape(-1, IN_C)
                # 2d matrix
                mat_w = w.view(KERNEL_N, IN_C)
                mat_w = mat_w.permute(1, 0)
                # 2d matrix y, (BATCH * OUT_H * OUT_W, KERNEL_N)
                mat_y = triton.ops.matmul(mat_x, mat_w)
                # mat_y = torch.empty((BATCH * OUT_H * OUT_W, KERNEL_N), device=device, dtype=x.dtype,)
                y = mat_y.view(BATCH, OUT_H, OUT_W, KERNEL_N)
                if bias is not None:
                    y += bias
                # convert back to the original layout of y
                # nhwc -> nchw
                y = y.permute(0, 3, 1, 2)
                if not channels_last:
                    y = y.to(memory_format=torch.contiguous_format)
                return y

            else:
                y = torch.empty(
                    (shape_y[yn], shape_y[yh], shape_y[yw], shape_y[yc]),
                    device=device,
                    dtype=x.dtype,
                )
                if channels_last:
                    y = y.to(memory_format=torch.channels_last)
                # y = bias.repeat((shape_y[yn], shape_y[yh], shape_y[yw], 1)).to(device).type(x.dtype)
                # convert x to channel-last layout;
                # don't care w layout since kernel size is 1
                x = x.permute(0, 2, 3, 1)
                # select every stride"s element (for stride > 1)
                x = x[:, :: stride[0], :: stride[1], :]
                # 2d matrix
                mat_x = x.view(-1, IN_C)
                # 2d matrix
                mat_w = w.view(KERNEL_N, IN_C)
                mat_w = mat_w.permute(1, 0)
                # 2d matrix y, (BATCH * (OUT_H-2*padding) * (OUT_W-2*padding), KERNEL_N)
                mat_y = triton.ops.matmul(mat_x, mat_w)
                mat_y = mat_y.view(
                    BATCH, OUT_H - 2 * padding[0], OUT_W - 2 * padding[1], KERNEL_N
                )
                # consider padding > 0
                if bias is not None:
                    y[
                        :,
                        padding[0] : OUT_H - padding[0],
                        padding[1] : OUT_W - padding[1],
                        :,
                    ] = (
                        mat_y + bias
                    )
                    y[:, : padding[0], :, :] = bias
                    y[:, :, : padding[1], :] = bias
                    y[:, OUT_H - padding[0] :, :, :] = bias
                    y[:, :, OUT_W - padding[1] :, :] = bias
                else:
                    y[
                        :,
                        padding[0] : OUT_H - padding[0],
                        padding[1] : OUT_W - padding[1],
                        :,
                    ] = mat_y
                    y[:, : padding[0], :, :] = 0
                    y[:, :, : padding[1], :] = 0
                    y[:, OUT_H - padding[0] :, :, :] = 0
                    y[:, :, OUT_W - padding[1] :, :] = 0
                # convert back to the original layout of y
                # nhwc -> nchw
                y = y.permute(0, 3, 1, 2)
                return y

        @staticmethod
        def forward(
            x,
            w,
            bias,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
        ):
            if groups != 1:
                print(f"Do not support groups = {groups}")
                return
            if transposed:
                print("Do not support transposed")
            return _conv1x1._call(
                x,
                w,
                bias,
                stride,
                padding,
                dilation,
                transposed,
                output_padding,
                groups,
            )

    conv1x1 = _conv1x1.forward
