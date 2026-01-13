import torch


def pearsoncor(
    x: torch.Tensor, y: torch.Tensor, reduction: str = "mean", eps: float = 1e-4
) -> torch.Tensor:
    """
    Args:
        x: tensor of shape (B, T, S)
        y: tensor of shape (B, T, S)
        reduction: str of "mean" or "sum"
    Returns:
        corr: tensor of shape (1,)
    """

    mux = x.mean(dim=1, keepdim=True)
    muy = y.mean(dim=1, keepdim=True)

    u = torch.sum((x - mux) * (y - muy), dim=1)
    d = torch.sqrt(
        torch.sum((x - mux) ** 2, dim=1) * torch.sum((y - muy) ** 2, dim=1)
    )

    corr = u / (d + eps)

    if reduction == "sum":
        corr = corr.sum()
    elif reduction == "mean":
        corr = corr.mean()
    else:
        raise ValueError("Uknown reduction mode")
    return corr
