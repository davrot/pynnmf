import torch
import torchvision  # type: ignore
from data_loader import data_loader


def get_the_data(
    dataset: str,
    batch_size_train: int,
    batch_size_test: int,
    torch_device: torch.device,
    input_dim_x: int,
    input_dim_y: int,
    flip_p: float = 0.5,
    jitter_brightness: float = 0.5,
    jitter_contrast: float = 0.1,
    jitter_saturation: float = 0.1,
    jitter_hue: float = 0.15,
) -> tuple[
    data_loader,
    data_loader,
    torchvision.transforms.Compose,
    torchvision.transforms.Compose,
]:
    if dataset == "MNIST":
        tv_dataset_train = torchvision.datasets.MNIST(
            root="data", train=True, download=True
        )
        tv_dataset_test = torchvision.datasets.MNIST(
            root="data", train=False, download=True
        )
    elif dataset == "FashionMNIST":
        tv_dataset_train = torchvision.datasets.FashionMNIST(
            root="data", train=True, download=True
        )
        tv_dataset_test = torchvision.datasets.FashionMNIST(
            root="data", train=False, download=True
        )
    elif dataset == "CIFAR10":
        tv_dataset_train = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True
        )
        tv_dataset_test = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True
        )
    else:
        raise NotImplementedError("This dataset is not implemented.")

    if dataset == "MNIST" or dataset == "FashionMNIST":

        train_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_train,
            pattern=tv_dataset_train.data,
            labels=tv_dataset_train.targets,
            shuffle=True,
        )

        test_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_test,
            pattern=tv_dataset_test.data,
            labels=tv_dataset_test.targets,
            shuffle=False,
        )

        # Data augmentation filter
        test_processing_chain = torchvision.transforms.Compose(
            transforms=[torchvision.transforms.CenterCrop((input_dim_x, input_dim_y))],
        )

        train_processing_chain = torchvision.transforms.Compose(
            transforms=[torchvision.transforms.RandomCrop((input_dim_x, input_dim_y))],
        )
    else:

        train_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_train,
            pattern=torch.tensor(tv_dataset_train.data).movedim(-1, 1),
            labels=torch.tensor(tv_dataset_train.targets),
            shuffle=True,
        )

        test_dataloader = data_loader(
            torch_device=torch_device,
            batch_size=batch_size_test,
            pattern=torch.tensor(tv_dataset_test.data).movedim(-1, 1),
            labels=torch.tensor(tv_dataset_test.targets),
            shuffle=False,
        )

        # Data augmentation filter
        test_processing_chain = torchvision.transforms.Compose(
            transforms=[torchvision.transforms.CenterCrop((input_dim_x, input_dim_y))],
        )

        train_processing_chain = torchvision.transforms.Compose(
            transforms=[
                torchvision.transforms.RandomCrop((input_dim_x, input_dim_y)),
                torchvision.transforms.RandomHorizontalFlip(p=flip_p),
                torchvision.transforms.ColorJitter(
                    brightness=jitter_brightness,
                    contrast=jitter_contrast,
                    saturation=jitter_saturation,
                    hue=jitter_hue,
                ),
            ],
        )

    return (
        train_dataloader,
        test_dataloader,
        test_processing_chain,
        train_processing_chain,
    )
