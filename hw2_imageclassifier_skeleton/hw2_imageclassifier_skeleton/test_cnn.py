import torch
from models import CNN

size = 28
channels = [1, 100, 32]
shapes = [300, 100, 10]
PADDING = 3
kernel_size_conv = 5
stride_size_conv = 1
kernel_size_pool = 2
stride_size_pool = 2

buf_conv = []
for (ch_ins, ch_outs) in zip(channels[:-1], channels[1:]):
    buf_conv.append(torch.nn.Conv2d(ch_ins, ch_outs, kernel_size=kernel_size_conv, stride=stride_size_conv, padding=PADDING))
convs = torch.nn.ModuleList(buf_conv)
pool = torch.nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)

# Compute shape dynamically
with torch.no_grad():
    x = torch.zeros(1, channels[0], size, size)
    for conv in convs:
        x = conv(x)
        x = torch.nn.functional.relu(x)
    x = pool(x)
    shape_in = x.numel()

print("Calculated shape_in:", shape_in)
