import torch


def append_input_conv2d(
    network: torch.nn.Sequential,
    test_image: torch.tensor,
    dilation: int = 1,
    padding: int = 0,
    stride: int = 1,
    kernel_size: list[int] = [5, 5],
) -> torch.Tensor:

    mock_output = (
        torch.nn.functional.conv2d(
            torch.zeros(
                1,
                1,
                test_image.shape[2],
                test_image.shape[3],
            ),
            torch.zeros((1, 1, kernel_size[0], kernel_size[1])),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        .squeeze(0)
        .squeeze(0)
    )

    network.append(
        torch.nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
        )
    )
    test_image = network[-1](test_image)

    network.append(
        torch.nn.Fold(
            output_size=mock_output.shape,
            kernel_size=(1, 1),
            dilation=1,
            padding=0,
            stride=1,
        )
    )
    test_image = network[-1](test_image)

    return test_image
