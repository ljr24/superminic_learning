"""Verify CUDA execution and the compiler used by training."""
import json
import torch
assert torch.cuda.is_available()
x = torch.randn(1024, device='cuda', requires_grad=True)
f = torch.compile(lambda t: (t.sin() ** 2).sum())
y = f(x)
y.backward()
assert torch.isfinite(x.grad).all()
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(), 'compiled_backward': True}))
