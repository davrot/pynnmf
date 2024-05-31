import torch
from append_input_conv2d import append_input_conv2d
from L1NormLayer import L1NormLayer
from NNMF2d import NNMF2d
from Y import Y


def append_nnmf_block(
    network: torch.nn.Sequential,
    out_channels: int,
    test_image: torch.tensor,
    list_other_id: list[int],
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
    kernel_size: list[int] = [5, 5],
    epsilon: float | None = None,
    positive_function_type: int = 0,
    beta: float | None = None,
    iterations: int = 20,
    local_learning: bool = False,
    local_learning_kl: bool = False,
    use_reconstruction: bool = False,
    skip_connection: bool = False,
) -> torch.Tensor:

    kernel_size_internal: list[int] = list(kernel_size)

    if kernel_size[0] < 1:
        kernel_size_internal[0] = test_image.shape[-2]

    if kernel_size[1] < 1:
        kernel_size_internal[1] = test_image.shape[-1]

    test_image = append_input_conv2d(
        network=network,
        test_image=test_image,
        dilation=dilation,
        padding=padding,
        stride=stride,
        kernel_size=kernel_size_internal,
    )

    network.append(L1NormLayer())
    test_image = network[-1](test_image)

    list_other_id.append(len(network))
    if skip_connection:
        network.append(
            Y(
                torch.nn.Sequential(
                    torch.nn.Sequential(
                        NNMF2d(
                            in_channels=test_image.shape[1],
                            out_channels=out_channels,
                            epsilon=epsilon,
                            positive_function_type=positive_function_type,
                            beta=beta,
                            iterations=iterations,
                            local_learning=local_learning,
                            local_learning_kl=local_learning_kl,
                            use_reconstruction=use_reconstruction,
                            skip_connection=skip_connection,
                        )
                    ),
                    torch.nn.Sequential(torch.nn.Identity()),
                )
            )
        )
    else:
        network.append(
            NNMF2d(
                in_channels=test_image.shape[1],
                out_channels=out_channels,
                epsilon=epsilon,
                positive_function_type=positive_function_type,
                beta=beta,
                iterations=iterations,
                local_learning=local_learning,
                local_learning_kl=local_learning_kl,
                use_reconstruction=use_reconstruction,
                skip_connection=skip_connection,
            )
        )
    test_image = network[-1](test_image)

    return test_image
