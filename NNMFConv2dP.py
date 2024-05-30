import torch
from non_linear_weigth_function import non_linear_weigth_function


class NNMFConv2dP(torch.nn.Module):

    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    weight: torch.Tensor
    bias: None | torch.Tensor
    output_size: None | torch.Tensor = None
    convolution_contribution_map: None | torch.Tensor = None
    iterations: int
    convolution_contribution_map_enable: bool
    epsilon: float | None
    init_min: float
    init_max: float
    beta: torch.Tensor | None
    positive_function_type: int
    use_convolution: bool
    local_learning: bool
    local_learning_kl: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: str | tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        device=None,
        dtype=None,
        iterations: int = 20,
        convolution_contribution_map_enable: bool = False,
        epsilon: float | None = None,
        init_min: float = 0.0,
        init_max: float = 1.0,
        beta: float | None = None,
        positive_function_type: int = 0,
        use_convolution: bool = False,
        local_learning: bool = False,
        local_learning_kl: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        self.positive_function_type = positive_function_type
        self.init_min = init_min
        self.init_max = init_max

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.iterations = iterations
        self.convolution_contribution_map_enable = convolution_contribution_map_enable

        self.local_learning = local_learning
        self.local_learning_kl = local_learning_kl

        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_channels, in_channels, *kernel_size), **factory_kwargs)
        )

        if beta is not None:
            self.beta = torch.nn.parameter.Parameter(torch.empty((1), **factory_kwargs))
            self.beta.data[0] = beta
        else:
            self.beta = None

        self.reset_parameters()
        self.functional_nnmf_conv2d = FunctionalNNMFConv2dP.apply

        self.epsilon = epsilon
        self.use_convolution = use_convolution

        assert self.use_convolution is False

    def extra_repr(self) -> str:
        s: str = f"{self.in_channels}, {self.out_channels}"

        s += f", kernel_size={self.kernel_size}"
        s += f", stride={self.stride}, iterations={self.iterations}"
        s += f", epsilon={self.epsilon}"
        s += f", use_convolution={self.use_convolution}"

        if self.use_convolution:
            s += f", ccmap={self.convolution_contribution_map_enable}"

        s += f", pfunctype={self.positive_function_type}"
        s += f", local_learning={self.local_learning}"

        if self.local_learning:
            s += f", local_learning_kl={self.local_learning_kl}"

        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"

        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"

        return s

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, a=self.init_min, b=self.init_max)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if input.ndim == 2:
            input = input.unsqueeze(-1)
        if input.ndim == 3:
            input = input.unsqueeze(-1)

        if self.output_size is None:
            self.output_size = torch.tensor(
                torch.nn.functional.conv2d(
                    torch.zeros(
                        1,
                        input.shape[1],
                        input.shape[2],
                        input.shape[3],
                        device=self.weight.device,
                        dtype=self.weight.dtype,
                    ),
                    torch.zeros_like(self.weight),
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                ).shape,
                requires_grad=False,
            )
        assert self.output_size is not None

        input = torch.nn.functional.fold(
            torch.nn.functional.unfold(
                input.requires_grad_(True),
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            ),
            output_size=self.output_size[-2:],
            kernel_size=(1, 1),
            dilation=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )

        positive_weights = non_linear_weigth_function(
            self.weight, self.beta, self.positive_function_type
        )
        positive_weights = positive_weights / (
            positive_weights.sum((1, 2, 3), keepdim=True) + 10e-20
        )

        positive_weights = positive_weights.reshape(
            positive_weights.shape[0],
            positive_weights.shape[1]
            * positive_weights.shape[2]
            * positive_weights.shape[3],
        )

        # Prepare input
        input = input / (input.sum(dim=1, keepdim=True) + 10e-20)

        h_dyn = self.functional_nnmf_conv2d(
            input,
            positive_weights,
            self.output_size,
            self.iterations,
            self.stride,
            self.padding,
            self.dilation,
            self.epsilon,
            self.use_convolution,
            self.local_learning,
            self.local_learning_kl,
        )
        self.reco = False
        if self.reco:
            print(h_dyn.shape)
            print(positive_weights.shape)
            print(input.shape)
            exit()
            output = torch.cat((h_dyn, input), dim=1)
        else:
            output = torch.cat((h_dyn, input), dim=1)
        return output


class FunctionalNNMFConv2dP(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        output_size: torch.Tensor,
        iterations: int,
        stride: tuple[int, int],
        padding: str | tuple[int, int],
        dilation: tuple[int, int],
        epsilon: float | None,
        use_convolution: bool,
        local_learning: bool,
        local_learning_kl: bool,
    ) -> torch.Tensor:

        # Prepare h
        output_size[0] = input.shape[0]
        h = torch.full(
            output_size.tolist(),
            1.0 / float(output_size[1]),
            device=input.device,
            dtype=input.dtype,
        )

        if use_convolution:
            for _ in range(0, iterations):
                factor_x_div_r: torch.Tensor = input / (
                    torch.nn.functional.conv_transpose2d(
                        h,
                        weight,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                    + 10e-20
                )

                if epsilon is None:
                    h *= torch.nn.functional.conv2d(
                        factor_x_div_r,
                        weight,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )
                else:
                    h *= 1 + epsilon * torch.nn.functional.conv2d(
                        factor_x_div_r,
                        weight,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                    )

                h /= h.sum(1, keepdim=True) + 10e-20
        else:
            h = h.movedim(1, -1)
            input = input.movedim(1, -1)
            for _ in range(0, iterations):
                reconstruction = torch.nn.functional.linear(h, weight.T)
                reconstruction += 1e-20
                if epsilon is None:
                    h *= torch.nn.functional.linear((input / reconstruction), weight)
                else:
                    h *= 1 + epsilon * torch.nn.functional.linear(
                        (input / reconstruction), weight
                    )
                h /= h.sum(-1, keepdim=True) + 10e-20
            h = h.movedim(-1, 1)
            input = input.movedim(-1, 1)

        # ###########################################################
        # Save the necessary data for the backward pass
        # ###########################################################
        ctx.save_for_backward(input, weight, h)

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.use_convolution = use_convolution
        ctx.local_learning = local_learning
        ctx.local_learning_kl = local_learning_kl

        assert torch.isfinite(h).all()
        return h

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[  # type: ignore
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:

        # ##############################################
        # Default values
        # ##############################################
        grad_input: torch.Tensor | None = None
        grad_weight: torch.Tensor | None = None

        # ##############################################
        # Get the variables back
        # ##############################################
        (input, weight, h) = ctx.saved_tensors

        if ctx.use_convolution:
            big_r: torch.Tensor = torch.nn.functional.conv_transpose2d(
                h,
                weight,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
            )
            big_r_div = 1.0 / (big_r + 1e-20)

            factor_x_div_r: torch.Tensor = input * big_r_div

            grad_input = (
                torch.nn.functional.conv_transpose2d(
                    (h * grad_output),
                    weight,
                    stride=ctx.stride,
                    padding=ctx.padding,
                    dilation=ctx.dilation,
                )
                * big_r_div
            )

            del big_r_div
            if ctx.local_learning is False:
                del big_r
                grad_weight = -torch.nn.functional.conv2d(
                    (factor_x_div_r * grad_input).movedim(0, 1),
                    h.movedim(0, 1),
                    stride=ctx.dilation,
                    padding=ctx.padding,
                    dilation=ctx.stride,
                )

                grad_weight += torch.nn.functional.conv2d(
                    factor_x_div_r.movedim(0, 1),
                    (h * grad_output).movedim(0, 1),
                    stride=ctx.dilation,
                    padding=ctx.padding,
                    dilation=ctx.stride,
                )

            else:
                if ctx.local_learning_kl:
                    grad_weight = -torch.nn.functional.conv2d(
                        factor_x_div_r.movedim(0, 1),
                        h.movedim(0, 1),
                        stride=ctx.dilation,
                        padding=ctx.padding,
                        dilation=ctx.stride,
                    )
                else:
                    grad_weight = -torch.nn.functional.conv2d(
                        (2 * (input - big_r)).movedim(0, 1),
                        h.movedim(0, 1),
                        stride=ctx.dilation,
                        padding=ctx.padding,
                        dilation=ctx.stride,
                    )
            grad_weight = grad_weight.movedim(0, 1)
        else:
            h = h.movedim(1, -1)
            grad_output = grad_output.movedim(1, -1)
            input = input.movedim(1, -1)
            big_r = torch.nn.functional.linear(h, weight.T)
            big_r_div = 1.0 / (big_r + 1e-20)

            factor_x_div_r = input * big_r_div

            grad_input = (
                torch.nn.functional.linear(h * grad_output, weight.T) * big_r_div
            )

            del big_r_div

            if ctx.local_learning is False:
                del big_r

                grad_weight = -torch.nn.functional.linear(
                    h.reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        h.shape[3],
                    ).T,
                    (factor_x_div_r * grad_input)
                    .reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        grad_input.shape[3],
                    )
                    .T,
                )

                grad_weight += torch.nn.functional.linear(
                    (h * grad_output)
                    .reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        h.shape[3],
                    )
                    .T,
                    factor_x_div_r.reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        grad_input.shape[3],
                    ).T,
                )

            else:
                if ctx.local_learning_kl:
                    grad_weight = -torch.nn.functional.linear(
                        h.reshape(
                            grad_input.shape[0]
                            * grad_input.shape[1]
                            * grad_input.shape[2],
                            h.shape[3],
                        ).T,
                        factor_x_div_r.reshape(
                            grad_input.shape[0]
                            * grad_input.shape[1]
                            * grad_input.shape[2],
                            grad_input.shape[3],
                        ).T,
                    )
                else:
                    grad_weight = -torch.nn.functional.linear(
                        h.reshape(
                            grad_input.shape[0]
                            * grad_input.shape[1]
                            * grad_input.shape[2],
                            h.shape[3],
                        ).T,
                        (2 * (input - big_r))
                        .reshape(
                            grad_input.shape[0]
                            * grad_input.shape[1]
                            * grad_input.shape[2],
                            grad_input.shape[3],
                        )
                        .T,
                    )
            grad_input = grad_input.movedim(-1, 1)
        assert torch.isfinite(grad_input).all()
        assert torch.isfinite(grad_weight).all()

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
