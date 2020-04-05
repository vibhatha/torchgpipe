from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import time
from unet import unet
import click

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 64
dataset_size = 640

######################################################################
BASE_TIME: float = 0
epochs = 5
skip_epochs = 1
title = 'data_parallel'
save_file = 'unet_data_parallel_stats.csv'
experiment = title
cmd_params = sys.argv
if len(cmd_params) < 2:
    raise ValueError("Expected Command Line parameter is 1, given None.")
parallelism = cmd_params[1]
parallelism = 8
######################################################################
# Device
#
def max_gpu_devices():
    devices = []
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    return devices

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
devices = []
in_device = None
out_device = None

max_devices = max_gpu_devices()
if parallelism > len(max_devices):
    raise ValueError("Parallelism {} exceeds the maximum number of devices {}".format(parallelism, max_devices))
else:
    devices = list(range(parallelism))
in_device = devices[0]
out_device = devices[-1]


######################################################################
# Dummy DataSet
# -------------
#
# Make a dummy (random) dataset. You just need to implement the
# getitem
#


class RandomDataset(Dataset):

    def __init__(self, batch_size, dataset_size):
        input = torch.rand(3, 192, 192, device=in_device)
        target = torch.ones(1, 192, 192, device=out_device)
        data = [(input, target)] * dataset_size

        self.len = dataset_size
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


data = DataLoader(dataset=RandomDataset(batch_size, dataset_size),
                  batch_size=batch_size, shuffle=True)

######################################################################
# Simple Model
# ------------
#
# For the demo, our model just gets an input, performs a linear operation, and
# gives an output. However, you can use ``DataParallel`` on any model (CNN, RNN,
# Capsule Net etc.)
#
# We've placed a print statement inside the model to monitor the size of input
# and output tensors.
# Please pay attention to what is printed at batch rank 0.
#


model: nn.Module = unet(depth=5, num_convs=5, base_channels=64,
                        input_channels=3, output_channels=1)

optimizer = SGD(model.parameters(), lr=0.1)


######################################################################
# Create Model and DataParallel
# -----------------------------
#
# This is the core part of the tutorial. First, we need to make a model instance
# and check if we have multiple GPUs. If we have multiple GPUs, we can wrap
# our model using ``nn.DataParallel``. Then we can put our model on GPUs by
# ``model.to(device)``
#


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(module=model, device_ids=devices)
#
  model.to(device)


######################################################################
# Run the Model
# -------------
#
# Now we can see the sizes of input and output tensors.
#

# for i, (input, target) in enumerate(data):
#     print(i, input.shape, target.shape)
#
#


def hr() -> None:
    """Prints a horizontal line."""
    width, _ = click.get_terminal_size()
    click.echo('-' * width)


def log(msg: str, clear: bool = False, nl: bool = True) -> None:
    """Prints a message with elapsed time."""
    if clear:
        # Clear the output line to overwrite.
        width, _ = click.get_terminal_size()
        click.echo('\b\r', nl=False)
        click.echo(' ' * width, nl=False)
        click.echo('\b\r', nl=False)

    t = time.time() - BASE_TIME
    h = t // 3600
    t %= 3600
    m = t // 60
    t %= 60
    s = t

    click.echo('%02d:%02d:%02d | ' % (h, m, s), nl=False)
    click.echo(msg, nl=nl)


def synch_for_all_devices():
    for dv in devices:
        torch.cuda.synchronize(dv)


def run_epoch(epoch: int) -> Tuple[float, float]:
    synch_for_all_devices()
    tick = time.time()

    data_trained = 0
    forward_time = []
    backward_time = []
    loss_time = []
    opt_time = []
    for i, (input, target) in enumerate(data):
        input = input.to(device)
        target = target.to(device)
        data_trained += input.size(0)

        synch_for_all_devices()
        t1 = time.time()
        output = model(input)
        synch_for_all_devices()
        forward_time.append(time.time() - t1)

        synch_for_all_devices()
        t1 = time.time()
        loss = F.binary_cross_entropy_with_logits(output, target)
        synch_for_all_devices()
        loss_time.append(time.time() - t1)

        synch_for_all_devices()
        t1 = time.time()
        loss.backward()
        synch_for_all_devices()
        backward_time.append(time.time() - t1)

        synch_for_all_devices()
        t1 = time.time()
        optimizer.step()
        optimizer.zero_grad()
        synch_for_all_devices()
        opt_time.append(time.time() - t1)

        # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
        percent = (i + 1) / len(data) * 100
        throughput = data_trained / (time.time() - tick)
        log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
            '' % (epoch + 1, epochs, percent, throughput), clear=True,
            nl=False)

    synch_for_all_devices()
    tock = time.time()

    # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
        '' % (epoch + 1, epochs, throughput, elapsed_time), clear=True)

    return throughput, elapsed_time, sum(forward_time), \
           sum(backward_time), sum(loss_time), sum(opt_time)


#global BASE_TIME
BASE_TIME = time.time()

# for id, data in enumerate(rand_loader):
#     input = data[0]
#     target = data[1]
#     input = input.to(device)
#     target = target.to(device)

# for data in rand_loader:
#     input = data.to(device)
#     output = model(input)
#     print("Outside: input size", input.size(),
#           "output_size", output.size())


throughputs = []
elapsed_times = []
forward_times = []
backward_times = []
loss_times = []
opt_times = []

hr()
synch_for_all_devices()
t1 = time.time()
for epoch in range(epochs):
    throughput, elapsed_time, forward_time, backward_time, loss_time, opt_time = run_epoch(epoch)

    if epoch < skip_epochs:
        continue

    throughputs.append(throughput)
    elapsed_times.append(elapsed_time)
    forward_times.append(forward_time)
    backward_times.append(backward_time)
    loss_times.append(loss_time)
    opt_times.append(opt_time)
hr()
synch_for_all_devices()
t2 = time.time()

# RESULT ======================================================================================

# pipeline-4, 2-10 epochs | 200.000 samples/sec, 123.456 sec/epoch (average)
n = len(throughputs)
throughput = sum(throughputs) / n
elapsed_time = sum(elapsed_times) / n
forward_avg_time = sum(forward_times) / n
backward_avg_time = sum(backward_times) / n
loss_avg_time = sum(loss_times) / n
opt_avg_time = sum(opt_times) / n
click.echo('%s | %.3f samples/sec, %.3f sec/epoch (average)'
           '' % (title, throughput, elapsed_time))
click.echo('Average Time Per Epoch {}'.format((t2-t1)/epochs))

if save_file is not None:
    with open(save_file, "a+") as fp:
        fp.write(
            "{},{},{},{},{},{},{},{},{},{}\n".format(experiment, parallelism, dataset_size,
                                                     batch_size, throughput,
                                                     elapsed_time, forward_avg_time,
                                                     backward_avg_time, loss_avg_time, opt_avg_time))