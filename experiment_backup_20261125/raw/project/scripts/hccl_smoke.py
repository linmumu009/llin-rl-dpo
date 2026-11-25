import os

import torch
import torch.distributed as dist
import torch_npu


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    tensor = torch.tensor([float(rank + 1)]).npu()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    result = tensor.cpu().item()

    if rank == 0:
        expected = world_size * (world_size + 1) / 2
        print(f"world_size={world_size} all_reduce_sum={result} expected={expected}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

