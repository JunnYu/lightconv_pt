# lightconv_layer
lightconv_layer fairseq

# Generate cuda_function_gen
```bash
cd csrc
python cuda_function_gen.py
```

# Install 
```bash
git clone https://github.com/JunnYu/lightconv_pt.git
python setup.py build install
or
pip install git+https://github.com/JunnYu/lightconv_pt.git
```

# Usage
```python
import torch
from lightconv_layer import LightweightConv, LightweightConv1d

# T x B x C
x = torch.randn(16, 1, 32).cuda()
model1 = LightweightConv(32, kernel_size=3, padding_l=1).cuda()
model1.eval()
with torch.no_grad():
    o1 = model1(x)

# B x C x T
model2 = LightweightConv1d(32, kernel_size=3, padding=1).cuda()
model2.eval()
model2.weight.data = model1.weight.data.unsqueeze(1)
x2 = x.permute(1, 2, 0)
with torch.no_grad():
    o2 = model2(x2).permute(2, 0, 1)

# T x B x C
print(o1.shape, o2.shape)

dif = (o1 - o2).abs().mean()
print(dif)
# torch.Size([16, 1, 32]) torch.Size([16, 1, 32])
# tensor(0., device='cuda:0')
```

# Reference
https://github.com/pytorch/fairseq/tree/master/fairseq/modules/lightconv_layer