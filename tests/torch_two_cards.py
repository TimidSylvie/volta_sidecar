"""One PyTorch process, both cards: cuda:0 (open driver) and cuda:1 (sidecar), through the CUDA proxy."""
import time
import torch

n = torch.cuda.device_count()
print("device_count:", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f} GiB")
assert n >= 2, "expected both cards"
required_arches = {f"sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]}" for i in range(n)}
built_arches = set(torch.cuda.get_arch_list())
missing = sorted(required_arches - built_arches)
assert not missing, (
    f"this PyTorch build has no kernels for {', '.join(missing)}; the proxy can expose both GPUs, but full Torch "
    'operation requires one build containing every device architecture (for V100 + RTX 5060 Ti, build with '
    'CUDA 12.8 and TORCH_CUDA_ARCH_LIST="7.0;12.0")'
)
cpu = torch.randn(512, 512, generator=torch.Generator().manual_seed(0))
ref = cpu @ cpu
for i in range(n):
    t0 = time.time()
    g = cpu.to(f"cuda:{i}")
    out = (g @ g).cpu()                                     # cuBLAS on that card
    print(f"  cuda:{i} matmul max err {(out - ref).abs().max().item():.2e} ({time.time() - t0:.2f}s)")
a = torch.arange(1 << 20, dtype=torch.float32, device="cuda:1") * 3
b = a.to("cuda:0")                                          # V100 -> 5060
c = (b + 1).to("cuda:1")                                    # 5060 -> V100
ok = torch.equal(c.cpu(), torch.arange(1 << 20, dtype=torch.float32) * 3 + 1)
print("cross-card copies:", "OK" if ok else "WRONG")
with torch.cuda.device(1):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        x = torch.randn(2048, 2048, device="cuda:1"); y = torch.nn.functional.softmax(x @ x, dim=-1).sum()
    s.synchronize()
print("stream + softmax on cuda:1:", round(y.item()), "(expect 2048)")
print("done")
