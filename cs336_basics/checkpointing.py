import os
import typing
import torch


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    total_state_dict = {}
    total_state_dict["iteration"] = iteration
    total_state_dict["model"] = model.state_dict()
    total_state_dict["optimizer"] = optimizer.state_dict()
    torch.save(total_state_dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    total_state_dict = torch.load(src)
    model.load_state_dict(total_state_dict["model"])
    optimizer.load_state_dict(total_state_dict["optimizer"])
    return total_state_dict["iteration"]