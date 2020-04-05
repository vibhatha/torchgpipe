import torch
from torch import nn
import torchvision
import torchbenchmark.estimator.model_estimator as est
import click
from unet import unet

model: nn.Module = unet(depth=5, num_convs=5, base_channels=64,
                        input_channels=3, output_channels=1)



batch_size = 1
image_w = 192
image_h = 192
input_ = torch.randn(batch_size, 3, image_w, image_h)

input_size = (3, 192, 192)

ms = est.ModelEstimator(name='alexnet', model=model, input_size=input_size, batch_size=batch_size, save=False,
                                  save_path='unet_info_dummy.info', console=False)

ms.generate_summary()