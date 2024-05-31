import torch
from SplitOnOffLayer import SplitOnOffLayer
from append_nnmf_block import append_nnmf_block


def make_network(
    use_nnmf: bool,
    cnn_top: bool,
    input_dim_x: int,
    input_dim_y: int,
    input_number_of_channel: int,
    iterations: int,
    epsilon: bool | None = None,
    positive_function_type: int = 0,
    beta: float | None = None,
    # Conv:
    number_of_output_channels: list[int] = [32, 64, 96, 10],
    kernel_size_conv: list[tuple[int, int]] = [
        (5, 5),
        (5, 5),
        (-1, -1),  # Take the whole input image x and y size
        (1, 1),
    ],
    stride_conv: list[tuple[int, int]] = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    padding_conv: list[tuple[int, int]] = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    dilation_conv: list[tuple[int, int]] = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    # Pool:
    kernel_size_pool: list[tuple[int, int]] = [
        (2, 2),
        (2, 2),
        (-1, -1),  # No pooling layer
        (-1, -1),  # No pooling layer
    ],
    stride_pool: list[tuple[int, int]] = [
        (2, 2),
        (2, 2),
        (-1, -1),
        (-1, -1),
    ],
    padding_pool: list[tuple[int, int]] = [
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    dilation_pool: list[tuple[int, int]] = [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    local_learning: list[bool] = [False, False, False, False],
    skip_connection: list[bool] = [False, False, False, False],
    local_learning_kl: bool = True,
    use_reconstruction: bool = False,
    max_pool: bool = True,
    enable_onoff: bool = False,
) -> tuple[torch.nn.Sequential, list[int], list[int]]:

    assert len(number_of_output_channels) == len(kernel_size_conv)
    assert len(number_of_output_channels) == len(stride_conv)
    assert len(number_of_output_channels) == len(padding_conv)
    assert len(number_of_output_channels) == len(dilation_conv)
    assert len(number_of_output_channels) == len(kernel_size_pool)
    assert len(number_of_output_channels) == len(stride_pool)
    assert len(number_of_output_channels) == len(padding_pool)
    assert len(number_of_output_channels) == len(dilation_pool)
    assert len(number_of_output_channels) == len(local_learning)
    assert len(number_of_output_channels) == len(skip_connection)

    if enable_onoff:
        input_number_of_channel *= 2

    list_cnn_top_id: list[int] = []
    list_other_id: list[int] = []

    test_image = torch.ones((1, input_number_of_channel, input_dim_x, input_dim_y))

    network = torch.nn.Sequential()

    if enable_onoff:
        network.append(SplitOnOffLayer())
        test_image = network[-1](test_image)

    for block_id in range(0, len(number_of_output_channels)):
        if use_nnmf:
            test_image = append_nnmf_block(
                network=network,
                out_channels=number_of_output_channels[block_id],
                test_image=test_image,
                list_other_id=list_other_id,
                dilation=dilation_conv[block_id],
                padding=padding_conv[block_id],
                stride=stride_conv[block_id],
                kernel_size=kernel_size_conv[block_id],
                epsilon=epsilon,
                positive_function_type=positive_function_type,
                beta=beta,
                iterations=iterations,
                local_learning=local_learning[block_id],
                local_learning_kl=local_learning_kl,
                use_reconstruction=use_reconstruction,
                skip_connection=skip_connection[block_id],
            )
        else:
            list_other_id.append(len(network))

            kernel_size_conv_internal = list(kernel_size_conv[block_id])

            if kernel_size_conv[block_id][0] == -1:
                kernel_size_conv_internal[0] = test_image.shape[-2]

            if kernel_size_conv[block_id][1] == -1:
                kernel_size_conv_internal[1] = test_image.shape[-1]

            network.append(
                torch.nn.Conv2d(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels[block_id],
                    kernel_size=kernel_size_conv_internal,
                    stride=1,
                    padding=0,
                )
            )
            test_image = network[-1](test_image)
            if cnn_top or block_id < len(number_of_output_channels) - 1:
                network.append(torch.nn.ReLU())
                test_image = network[-1](test_image)

        if cnn_top:
            list_cnn_top_id.append(len(network))
            network.append(
                torch.nn.Conv2d(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels[block_id],
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=True,
                )
            )
            test_image = network[-1](test_image)
            if block_id < len(number_of_output_channels) - 1:
                network.append(torch.nn.ReLU())
                test_image = network[-1](test_image)

        if (kernel_size_pool[block_id][0] > 0) and (kernel_size_pool[block_id][1] > 0):
            if max_pool:
                network.append(
                    torch.nn.MaxPool2d(
                        kernel_size=kernel_size_pool[block_id],
                        stride=stride_pool[block_id],
                        padding=padding_pool[block_id],
                    )
                )
            else:
                network.append(
                    torch.nn.AvgPool2d(
                        kernel_size=kernel_size_pool[block_id],
                        stride=stride_pool[block_id],
                        padding=padding_pool[block_id],
                    )
                )
            test_image = network[-1](test_image)

    network.append(torch.nn.Flatten())
    test_image = network[-1](test_image)

    network.append(torch.nn.Softmax(dim=1))
    test_image = network[-1](test_image)

    return network, list_cnn_top_id, list_other_id
