import torch
from NNMF2d import NNMF2d


def make_optimize(
    network: torch.nn.Sequential,
    list_cnn_top_id: list[int],
    list_other_id: list[int],
    lr_initial_nnmf: float = 0.01,
    lr_initial_cnn: float = 0.001,
    lr_initial_cnn_top: float = 0.001,
    eps=1e-10,
) -> tuple[
    torch.optim.Adam | None,
    torch.optim.Adam | None,
    torch.optim.Adam | None,
    torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    torch.optim.lr_scheduler.ReduceLROnPlateau | None,
]:

    list_cnn_top: list = []
    # Init the cnn top layers 1x1 conv2d layers
    for layerid in list_cnn_top_id:
        for netp in network[layerid].parameters():
            with torch.no_grad():
                if netp.ndim == 1:
                    netp.data *= 0
                if netp.ndim == 4:
                    assert netp.shape[-2] == 1
                    assert netp.shape[-1] == 1
                    netp[: netp.shape[0], : netp.shape[0], 0, 0] = torch.eye(
                        netp.shape[0], dtype=netp.dtype, device=netp.device
                    )
                    netp[netp.shape[0] :, :, 0, 0] = 0
                    netp[:, netp.shape[0] :, 0, 0] = 0

            list_cnn_top.append(netp)

    list_cnn: list = []
    list_nnmf: list = []
    for layerid in list_other_id:
        if isinstance(network[layerid], torch.nn.Conv2d):
            for netp in network[layerid].parameters():
                list_cnn.append(netp)

        if isinstance(network[layerid], NNMF2d):
            for netp in network[layerid].parameters():
                list_nnmf.append(netp)

    # The optimizer
    if len(list_nnmf) > 0:
        optimizer_nnmf: torch.optim.Adam | None = torch.optim.Adam(
            list_nnmf, lr=lr_initial_nnmf
        )
    else:
        optimizer_nnmf = None

    if len(list_cnn) > 0:
        optimizer_cnn: torch.optim.Adam | None = torch.optim.Adam(
            list_cnn, lr=lr_initial_cnn
        )
    else:
        optimizer_cnn = None

    if len(list_cnn_top) > 0:
        optimizer_cnn_top: torch.optim.Adam | None = torch.optim.Adam(
            list_cnn_top, lr=lr_initial_cnn_top
        )
    else:
        optimizer_cnn_top = None

    # The LR Scheduler
    if optimizer_nnmf is not None:
        lr_scheduler_nnmf: torch.optim.lr_scheduler.ReduceLROnPlateau | None = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_nnmf, eps=eps)
        )
    else:
        lr_scheduler_nnmf = None

    if optimizer_cnn is not None:
        lr_scheduler_cnn: torch.optim.lr_scheduler.ReduceLROnPlateau | None = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, eps=eps)
        )
    else:
        lr_scheduler_cnn = None

    if optimizer_cnn_top is not None:
        lr_scheduler_cnn_top: torch.optim.lr_scheduler.ReduceLROnPlateau | None = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn_top, eps=eps)
        )
    else:
        lr_scheduler_cnn_top = None

    return (
        optimizer_nnmf,
        optimizer_cnn,
        optimizer_cnn_top,
        lr_scheduler_nnmf,
        lr_scheduler_cnn,
        lr_scheduler_cnn_top,
    )
