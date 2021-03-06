"""U-Net Speed Benchmark"""
import platform
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import click
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
import torch.utils.data

import torchgpipe
from torchgpipe import GPipe
from unet import unet

Stuffs = Tuple[
    nn.Module, int, List[torch.device]]  # (model, batch_size, devices)
Experiment = Callable[[nn.Module, List[int]], Stuffs]


class Experiments:

    @staticmethod
    def baseline(model: nn.Module, devices: List[int], batch_size: int,
                 chunks: int, checkpointing: str) -> Stuffs:
        device = devices[0]
        model.to(device)
        return model, batch_size, [torch.device(device)]

    @staticmethod
    def pipeline1(model: nn.Module, devices: List[int], batch_size: int,
                  chunks: int, checkpointing: str) -> Stuffs:
        balance = [241]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint=checkpointing)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline2(model: nn.Module, devices: List[int], batch_size: int,
                  chunks: int, checkpointing: str) -> Stuffs:
        balance = [104, 137]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint=checkpointing)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline4(model: nn.Module, devices: List[int], batch_size: int,
                  chunks: int, checkpointing: str) -> Stuffs:
        balance = [30, 66, 84, 61]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint=checkpointing)
        return model, batch_size, list(model.devices)

    @staticmethod
    def pipeline8(model: nn.Module, devices: List[int], batch_size: int,
                  chunks: int, checkpointing: str) -> Stuffs:
        balance = [16, 27, 31, 44, 22, 57, 27, 17]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint=checkpointing)
        return model, batch_size, list(model.devices)

    @staticmethod
    def dataparallel(model: nn.Module, devices: List[int], batch_size: int,
                  chunks: int, checkpointing: str) -> Stuffs:
        balance = [16, 27, 31, 44, 22, 57, 27, 17]

        model = cast(nn.Sequential, model)
        model = GPipe(model, balance, devices=devices, chunks=chunks, checkpoint=checkpointing)
        return model, batch_size, list(model.devices)

EXPERIMENTS: Dict[str, Experiment] = {
    'baseline': Experiments.baseline,
    'pipeline-1': Experiments.pipeline1,
    'pipeline-2': Experiments.pipeline2,
    'pipeline-4': Experiments.pipeline4,
    'pipeline-8': Experiments.pipeline8,
}

BASE_TIME: float = 0


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


def parse_devices(ctx: Any, param: Any, value: Optional[str]) -> List[int]:
    if value is None:
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in value.split(',')]


@click.command()
@click.pass_context
@click.argument(
    'experiment',
    type=click.Choice(sorted(EXPERIMENTS.keys())),
)
@click.option(
    '--epochs', '-e',
    type=int,
    default=10,
    help='Number of epochs (default: 10)',
)
@click.option(
    '--batch_size', '-e',
    type=int,
    default=10,
    help='Batch Size (default: 10)',
)
@click.option(
    '--chunks', '-e',
    type=int,
    default=10,
    help='Chunks (default: 2)',
)
@click.option(
    '--save_file', '-e',
    type=str,
    default=None,
    help='Save File (default: None)',
)
@click.option(
    '--checkpointing', '-e',
    type=str,
    default='except_last',
    help='checkpoint mode (default: except_last)',
)
@click.option(
    '--dataset_size', '-e',
    type=int,
    default=8,
    help='Datasize (default: 10000)',
)
@click.option(
    '--skip-epochs', '-k',
    type=int,
    default=1,
    help='Number of epochs to skip in result (default: 1)',
)
@click.option(
    '--devices', '-d',
    metavar='0,1,2,3',
    callback=parse_devices,
    help='Device IDs to use (default: all CUDA devices)',
)


def cli(ctx: click.Context,
        experiment: str,
        epochs: int,
        batch_size: int,
        chunks: int,
        dataset_size: int,
        skip_epochs: int,
        devices: List[int],
        save_file: str,
        checkpointing: str
        ) -> None:
    """U-Net Speed Benchmark"""
    if skip_epochs >= epochs:
        ctx.fail('--skip-epochs=%d must be less than --epochs=%d' % (
            skip_epochs, epochs))

    model: nn.Module = unet(depth=5, num_convs=5, base_channels=64,
                            input_channels=3, output_channels=1)

    hr()
    click.echo(message="Experiment: {}".format(experiment), color="red")
    click.echo(
        message="Configs: Dataset Size {}, Batch Size {}, Chunk Size (Micro-batch size) {}, Save File {} ,Checkpointing {}".format(
            dataset_size,
            batch_size,
            chunks,
            save_file,
            checkpointing),
        color="teal")
    hr()

    f: Experiment = EXPERIMENTS[experiment]
    try:
        model, batch_size, _devices = f(model, devices, batch_size, chunks, checkpointing)

    except ValueError as exc:
        # Examples:
        #   ValueError: too few devices to hold given partitions (devices: 1, paritions: 2)
        ctx.fail(str(exc))

    optimizer = SGD(model.parameters(), lr=0.1)

    in_device = _devices[0]
    out_device = _devices[-1]
    torch.cuda.set_device(in_device)

    input = torch.rand(batch_size, 3, 192, 192, device=in_device)
    target = torch.ones(batch_size, 1, 192, 192, device=out_device)
    data = [(input, target)] * (dataset_size // batch_size)

    if dataset_size % batch_size != 0:
        last_input = input[:dataset_size % batch_size]
        last_target = target[:dataset_size % batch_size]
        data.append((last_input, last_target))

    # HEADER ======================================================================================

    title = f'{experiment}, {skip_epochs + 1}-{epochs} epochs'
    click.echo(title)

    if isinstance(model, GPipe):
        click.echo(f'batch size: {batch_size}, chunks: {model.chunks}, '
                   f'balance: {model.balance}, checkpoint: {model.checkpoint}')
    else:
        click.echo(f'batch size: {batch_size}')

    click.echo(
        'torchgpipe: %s, python: %s, torch: %s, cudnn: %s, cuda: %s, gpu: %s' % (
            torchgpipe.__version__,
            platform.python_version(),
            torch.__version__,
            torch.backends.cudnn.version(),
            torch.version.cuda,
            torch.cuda.get_device_name(in_device)))

    # TRAIN =======================================================================================

    global BASE_TIME
    BASE_TIME = time.time()

    def run_epoch(epoch: int) -> Tuple[float, float]:
        for dv in _devices:
            torch.cuda.synchronize(dv)
        tick = time.time()

        data_trained = 0
        forward_time = []
        backward_time = []
        loss_time = []
        opt_time = []

        for i, (input, target) in enumerate(data):
            data_trained += input.size(0)

            for dv in _devices:
                torch.cuda.synchronize(dv)
            t1 = time.time()
            output = model(input)
            for dv in _devices:
                torch.cuda.synchronize(dv)
            forward_time.append(time.time() - t1)

            for dv in _devices:
                torch.cuda.synchronize(dv)
            t1 = time.time()
            loss = F.binary_cross_entropy_with_logits(output, target)
            for dv in _devices:
                torch.cuda.synchronize(dv)
            loss_time.append(time.time() - t1)

            for dv in _devices:
                torch.cuda.synchronize(dv)
            t1 = time.time()
            loss.backward()
            for dv in _devices:
                torch.cuda.synchronize(dv)
            backward_time.append(time.time() - t1)

            for dv in _devices:
                torch.cuda.synchronize(dv)
            t1 = time.time()
            optimizer.step()
            optimizer.zero_grad()
            for dv in _devices:
                torch.cuda.synchronize(dv)
            opt_time.append(time.time() - t1)

            # 00:01:02 | 1/20 epoch (42%) | 200.000 samples/sec (estimated)
            percent = (i + 1) / len(data) * 100
            throughput = data_trained / (time.time() - tick)
            log('%d/%d epoch (%d%%) | %.3f samples/sec (estimated)'
                '' % (epoch + 1, epochs, percent, throughput), clear=True,
                nl=False)

        for dv in _devices:
            torch.cuda.synchronize(dv)
        tock = time.time()

        # 00:02:03 | 1/20 epoch | 200.000 samples/sec, 123.456 sec/epoch
        elapsed_time = tock - tick
        throughput = data_trained / elapsed_time
        log('%d/%d epoch | %.3f samples/sec, %.3f sec/epoch'
            '' % (epoch + 1, epochs, throughput, elapsed_time), clear=True)

        return throughput, elapsed_time, sum(forward_time), \
               sum(backward_time), sum(loss_time), sum(opt_time)

    throughputs = []
    elapsed_times = []
    forward_times = []
    backward_times = []
    loss_times = []
    opt_times = []

    hr()
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
                "{},{},{},{},{},{},{},{},{},{}\n".format(experiment, dataset_size,
                                                         batch_size, chunks, throughput,
                                                         elapsed_time, forward_avg_time,
                                                         backward_avg_time, loss_avg_time, opt_avg_time))


if __name__ == '__main__':
    cli()
