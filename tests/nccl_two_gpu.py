#!/usr/bin/env python3
"""Two local ranks, one GPU from each underlying driver, one ordinary NCCL all-reduce."""
import datetime
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, port: int) -> None:
    def stage(message: str) -> None:
        print(f"rank {rank}: {message}", flush=True)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    stage("set device")
    torch.cuda.set_device(rank)
    stage("initialize NCCL")
    dist.init_process_group("nccl", rank=rank, world_size=world, timeout=datetime.timedelta(seconds=45))
    stage("allocate")
    value = torch.empty(1, dtype=torch.float32, device=f"cuda:{rank}")
    value.copy_(torch.tensor([float(rank + 1)], dtype=torch.float32))
    stage("all-reduce")
    dist.all_reduce(value)
    stage("copy result")
    got = value.cpu().item()
    if got != world * (world + 1) / 2:
        raise RuntimeError(f"rank {rank}: all-reduce returned {got}")
    print(f"rank {rank} {torch.cuda.get_device_name(rank)}: NCCL all-reduce {got} OK", flush=True)
    dist.destroy_process_group()


def main() -> None:
    if torch.cuda.device_count() < 2:
        raise SystemExit("need at least two merged CUDA devices")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
