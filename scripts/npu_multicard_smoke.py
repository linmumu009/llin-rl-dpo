import torch
import torch_npu


def main() -> None:
    count = torch_npu.npu.device_count()
    print(f"device_count={count}")
    for index in range(count):
        torch_npu.npu.set_device(index)
        value = torch.ones(4).npu().sum().cpu().item()
        print(f"device={index} sum={value}")


if __name__ == "__main__":
    main()

