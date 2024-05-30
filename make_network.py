import torch
from NNMFConv2d import NNMFConv2d
from NNMFConv2dP import NNMFConv2dP
from SplitOnOffLayer import SplitOnOffLayer


def make_network(
    use_nnmf: bool,
    cnn_top: bool,
    input_dim_x: int,
    input_dim_y: int,
    input_number_of_channel: int,
    iterations: int,
    init_min: float = 0.0,
    init_max: float = 1.0,
    use_convolution: bool = False,
    convolution_contribution_map_enable: bool = False,
    epsilon: bool | None = None,
    positive_function_type: int = 0,
    beta: float | None = None,
    number_of_output_channels_conv1: int = 32,
    number_of_output_channels_conv2: int = 64,
    number_of_output_channels_flatten2: int = 96,
    number_of_output_channels_full1: int = 10,
    kernel_size_conv1: tuple[int, int] = (5, 5),
    kernel_size_pool1: tuple[int, int] = (2, 2),
    kernel_size_conv2: tuple[int, int] = (5, 5),
    kernel_size_pool2: tuple[int, int] = (2, 2),
    stride_conv1: tuple[int, int] = (1, 1),
    stride_pool1: tuple[int, int] = (2, 2),
    stride_conv2: tuple[int, int] = (1, 1),
    stride_pool2: tuple[int, int] = (2, 2),
    padding_conv1: int = 0,
    padding_pool1: int = 0,
    padding_conv2: int = 0,
    padding_pool2: int = 0,
    enable_onoff: bool = False,
    local_learning_0: bool = False,
    local_learning_1: bool = False,
    local_learning_2: bool = False,
    local_learning_3: bool = False,
    local_learning_kl: bool = True,
    p_mode_0: bool = False,
    p_mode_1: bool = False,
    p_mode_2: bool = False,
    p_mode_3: bool = False,
) -> tuple[torch.nn.Sequential, list[int], list[int]]:

    if enable_onoff:
        input_number_of_channel *= 2

    list_cnn_top_id: list[int] = []
    list_other_id: list[int] = []

    test_image = torch.ones((1, input_number_of_channel, input_dim_x, input_dim_y))

    network = torch.nn.Sequential()

    if enable_onoff:
        network.append(SplitOnOffLayer())
        test_image = network[-1](test_image)

    list_other_id.append(len(network))
    if use_nnmf:
        if p_mode_0:
            network.append(
                NNMFConv2dP(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_conv1,
                    kernel_size=kernel_size_conv1,
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_0,
                    local_learning_kl=local_learning_kl,
                )
            )
        else:
            network.append(
                NNMFConv2d(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_conv1,
                    kernel_size=kernel_size_conv1,
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_0,
                    local_learning_kl=local_learning_kl,
                )
            )
        test_image = network[-1](test_image)
    else:
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_conv1,
                kernel_size=kernel_size_conv1,
                stride=stride_conv1,
                padding=padding_conv1,
            )
        )
        test_image = network[-1](test_image)
        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

    if cnn_top:
        list_cnn_top_id.append(len(network))
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_conv1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )
        )
        test_image = network[-1](test_image)
        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

    network.append(
        torch.nn.MaxPool2d(
            kernel_size=kernel_size_pool1, stride=stride_pool1, padding=padding_pool1
        )
    )
    test_image = network[-1](test_image)

    list_other_id.append(len(network))
    if use_nnmf:
        if p_mode_1:
            network.append(
                NNMFConv2dP(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_conv2,
                    kernel_size=kernel_size_conv2,
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_1,
                    local_learning_kl=local_learning_kl,
                )
            )
        else:
            network.append(
                NNMFConv2d(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_conv2,
                    kernel_size=kernel_size_conv2,
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_1,
                    local_learning_kl=local_learning_kl,
                )
            )
        test_image = network[-1](test_image)
    else:
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_conv2,
                kernel_size=kernel_size_conv2,
                stride=stride_conv2,
                padding=padding_conv2,
            )
        )
        test_image = network[-1](test_image)
        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

    if cnn_top:
        list_cnn_top_id.append(len(network))
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_conv2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )
        )
        test_image = network[-1](test_image)
        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

    network.append(
        torch.nn.MaxPool2d(
            kernel_size=kernel_size_pool2, stride=stride_pool2, padding=padding_pool2
        )
    )
    test_image = network[-1](test_image)

    list_other_id.append(len(network))
    if use_nnmf:
        if p_mode_2:
            network.append(
                NNMFConv2dP(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_flatten2,
                    kernel_size=(test_image.shape[2], test_image.shape[3]),
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_2,
                    local_learning_kl=local_learning_kl,
                )
            )
        else:
            network.append(
                NNMFConv2d(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_flatten2,
                    kernel_size=(test_image.shape[2], test_image.shape[3]),
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_2,
                    local_learning_kl=local_learning_kl,
                )
            )
        test_image = network[-1](test_image)
    else:
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_flatten2,
                kernel_size=(test_image.shape[2], test_image.shape[3]),
            )
        )
        test_image = network[-1](test_image)
        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

    if cnn_top:
        list_cnn_top_id.append(len(network))
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_flatten2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )
        )
        test_image = network[-1](test_image)
        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

    list_other_id.append(len(network))
    if use_nnmf:
        if p_mode_3:
            network.append(
                NNMFConv2dP(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_full1,
                    kernel_size=(test_image.shape[2], test_image.shape[3]),
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_3,
                    local_learning_kl=local_learning_kl,
                )
            )
        else:
            network.append(
                NNMFConv2d(
                    in_channels=test_image.shape[1],
                    out_channels=number_of_output_channels_full1,
                    kernel_size=(test_image.shape[2], test_image.shape[3]),
                    convolution_contribution_map_enable=convolution_contribution_map_enable,
                    epsilon=epsilon,
                    positive_function_type=positive_function_type,
                    init_min=init_min,
                    init_max=init_max,
                    beta=beta,
                    use_convolution=use_convolution,
                    iterations=iterations,
                    local_learning=local_learning_3,
                    local_learning_kl=local_learning_kl,
                )
            )
        test_image = network[-1](test_image)
    else:
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_full1,
                kernel_size=(test_image.shape[2], test_image.shape[3]),
            )
        )
        test_image = network[-1](test_image)
        if cnn_top:
            network.append(torch.nn.ReLU())
            test_image = network[-1](test_image)

    if cnn_top:
        list_cnn_top_id.append(len(network))
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_output_channels_full1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )
        )
        test_image = network[-1](test_image)

    network.append(torch.nn.Flatten())
    test_image = network[-1](test_image)

    network.append(torch.nn.Softmax(dim=1))
    test_image = network[-1](test_image)

    return network, list_cnn_top_id, list_other_id
