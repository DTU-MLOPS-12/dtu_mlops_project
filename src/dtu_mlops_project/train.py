#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import copy
import importlib
import json
import logging
import os
import time
import typer
from typing import List, Dict, Optional
from types import SimpleNamespace
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml

from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

import wandb
has_wandb = True

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')


# Helper function to convert key=value list into a dictionary
def parse_key_value_pair(pair: str) -> Dict[str, str]:
    return dict(kv.split('=') for kv in pair)

@app.command()
def main(
    # Config
    config: str = typer.Option(None, "--config", "-c", help="YAML config file specifying default arguments"),

    # Dataset parameters
    data: str = typer.Option(None, help='path to dataset (positional is *deprecated*, use --data-dir)'),
    data_dir: str = typer.Option(None, help="Path to dataset (root dir)"),
    dataset: str = typer.Option("", help="Dataset type + name ('<type>/<name>')"),
    train_split: str = typer.Option("train", help="Dataset train split (default: train)"),
    val_split: str = typer.Option("validation", help="Dataset validation split (default: validation)"),
    train_num_samples: int = typer.Option(None, help="Number of samples in train split for IterableDatasets"),
    val_num_samples: int = typer.Option(None, help="Number of samples in validation split for IterableDatasets"),
    dataset_download: bool = typer.Option(False, help="Allow download of dataset"),
    class_map: str = typer.Option('', help="Path to class-to-index mapping file"),
    input_img_mode: str = typer.Option(None, help="Dataset image conversion mode"),
    input_key: str = typer.Option(None, help="Dataset key for input images"),
    target_key: str = typer.Option(None, help="Dataset key for target labels"),
    dataset_trust_remote_code: bool = typer.Option(False, help="Allow remote code execution for dataset import"),

    # Model parameters
    model: str = typer.Option("resnet50", help="Name of model to train (default: resnet50)"),
    pretrained: bool = typer.Option(False, help="Use pretrained version of specified network"),
    pretrained_path: str = typer.Option(None, help="Path to pretrained weights"),
    initial_checkpoint: str = typer.Option('', help="Load checkpoint after initialization"),
    resume: str = typer.Option('', help="Resume model and optimizer state from checkpoint"),
    no_resume_opt: bool = typer.Option(False, help="Prevent resuming optimizer state"),
    num_classes: int = typer.Option(None, help="Number of label classes"),
    gp: str = typer.Option(None, help="Global pool type"),
    img_size: int = typer.Option(None, help="Image size"),
    in_chans: int = typer.Option(None, help="Image input channels"),
    input_size: Optional[List[int]] = typer.Option(None, help="Input dimensions (d h w)"),
    crop_pct: float = typer.Option(None, help="Input image center crop percent"),
    mean: List[float] = typer.Option(None, help="Override mean pixel value of dataset"),
    std: List[float] = typer.Option(None, help="Override std deviation of dataset"),
    interpolation: str = typer.Option("", help="Image resize interpolation type"),
    batch_size: int = typer.Option(128, "-b", help="Batch size for training"),
    validation_batch_size: int = typer.Option(None, "-vb", help="Validation batch size override"),
    channels_last: bool = typer.Option(False, help="Use channels_last memory layout"),
    fuser: str = typer.Option("", help="Select JIT fuser"),
    grad_accum_steps: int = typer.Option(1, help="Number of steps to accumulate gradients"),
    grad_checkpointing: bool = typer.Option(False, help="Enable gradient checkpointing"),
    fast_norm: bool = typer.Option(False, help="Enable experimental fast normalization"),
    model_kwargs: List[str] = typer.Option([], help="Additional model kwargs in key=value format"),
    head_init_scale: float = typer.Option(None, help="Head initialization scale"),
    head_init_bias: float = typer.Option(None, help="Head initialization bias value"),
    torchcompile_mode: str = typer.Option(None, help="Torch.compile mode"),

    # scripting / codegen
    torchscript: bool = typer.Option(False, "--torchscript", help="torch.jit.script the full model"),
    torchcompile: str = typer.Option(None, "--torchcompile", show_default=False, help="Enable compilation w/ specified backend (default: inductor)."),

    # Device & distributed
    device: str = typer.Option("cuda", help="Device to use"),
    amp: bool = typer.Option(False, help="Use AMP for mixed precision training"),
    amp_dtype: str = typer.Option("float16", help="AMP dtype"),
    amp_impl: str = typer.Option("native", help="AMP implementation (native/apex)"),
    model_dtype: str = typer.Option(None, help="Model dtype override"),
    no_ddp_bb: bool = typer.Option(False, help="Disable broadcast buffers in DDP"),
    synchronize_step: bool = typer.Option(False, help="Synchronize CUDA at the end of each step"),
    local_rank: int = typer.Option(0, help="Local rank for distributed training"),
    device_modules: List[str] = typer.Option(None, help="Python imports for device backend modules"),

    # Optimizer parameters
    opt: str = typer.Option("sgd", help="Optimizer (default: sgd)"),
    opt_eps: float = typer.Option(None, help="Optimizer epsilon"),
    opt_betas: List[float] = typer.Option(None, help="Optimizer betas"),
    momentum: float = typer.Option(0.9, help="Optimizer momentum"),
    weight_decay: float = typer.Option(2e-5, help="Weight decay"),
    clip_grad: float = typer.Option(None, help="Clip gradient norm"),
    clip_mode: str = typer.Option("norm", help="Gradient clipping mode"),
    layer_decay: float = typer.Option(None, help="Layer-wise learning rate decay"),
    opt_kwargs: List[str] = typer.Option([], help="Additional optimizer kwargs in key=value format"),

    # Learning rate schedule parameters
    sched: str = typer.Option("cosine", help='LR scheduler (default: "cosine")'),
    sched_on_updates: bool = typer.Option(False, help="Apply LR scheduler step on update instead of epoch end."),
    lr: float = typer.Option(None, help="Learning rate, overrides lr-base if set (default: None)"),
    lr_base: float = typer.Option(0.1, help="Base learning rate: lr = lr_base * global_batch_size / base_size"),
    lr_base_size: int = typer.Option(256, help="Base learning rate batch size (divisor, default: 256)."),
    lr_base_scale: str = typer.Option("", help='Base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)'),
    lr_noise: List[float] = typer.Option(None, help="Learning rate noise on/off epoch percentages"),
    lr_noise_pct: float = typer.Option(0.67, help="Learning rate noise limit percent (default: 0.67)"),
    lr_noise_std: float = typer.Option(1.0, help="Learning rate noise std-dev (default: 1.0)"),
    lr_cycle_mul: float = typer.Option(1.0, help="Learning rate cycle len multiplier (default: 1.0)"),
    lr_cycle_decay: float = typer.Option(0.5, help="Amount to decay each learning rate cycle (default: 0.5)"),
    lr_cycle_limit: int = typer.Option(1, help="Learning rate cycle limit, cycles enabled if > 1"),
    lr_k_decay: float = typer.Option(1.0, help="Learning rate k-decay for cosine/poly (default: 1.0)"),
    warmup_lr: float = typer.Option(1e-5, help="Warmup learning rate (default: 1e-5)"),
    min_lr: float = typer.Option(0, help="Lower LR bound for cyclic schedulers that hit 0 (default: 0)"),
    epochs: int = typer.Option(300, help="Number of epochs to train (default: 300)"),
    epoch_repeats: float = typer.Option(0.0, help="Epoch repeat multiplier (number of times to repeat dataset epoch per train epoch)."),
    start_epoch: int = typer.Option(None, help="Manual epoch number (useful on restarts)"),
    decay_milestones: List[int] = typer.Option([90, 180, 270], help="List of decay epoch indices for multistep LR. Must be increasing"),
    decay_epochs: float = typer.Option(90, help="Epoch interval to decay LR"),
    warmup_epochs: int = typer.Option(5, help="Epochs to warmup LR, if scheduler supports"),
    warmup_prefix: bool = typer.Option(False, help="Exclude warmup period from decay schedule."),
    cooldown_epochs: int = typer.Option(0, help="Epochs to cooldown LR at min_lr, after cyclic schedule ends"),
    patience_epochs: int = typer.Option(10, help="Patience epochs for Plateau LR scheduler (default: 10)"),
    decay_rate: float = typer.Option(0.1, help="LR decay rate (default: 0.1)"),

    # Augmentation & regularization parameters
    no_aug: bool = typer.Option(False, help="Disable all training augmentation, override other train aug args"),
    train_crop_mode: str = typer.Option(None, help="Crop-mode in train"),
    scale: List[float] = typer.Option([0.08, 1.0], help="Random resize scale (default: 0.08 1.0)"),
    ratio: List[float] = typer.Option([3. / 4., 4. / 3.], help="Random resize aspect ratio (default: 0.75 1.33)"),
    hflip: float = typer.Option(0.5, help="Horizontal flip training aug probability"),
    vflip: float = typer.Option(0.0, help="Vertical flip training aug probability"),
    color_jitter: float = typer.Option(0.4, help="Color jitter factor (default: 0.4)"),
    color_jitter_prob: float = typer.Option(None, help="Probability of applying any color jitter."),
    grayscale_prob: float = typer.Option(None, help="Probability of applying random grayscale conversion."),
    gaussian_blur_prob: float = typer.Option(None, help="Probability of applying gaussian blur."),
    aa: str = typer.Option(None, help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    aug_repeats: float = typer.Option(0, help="Number of augmentation repetitions (distributed training only) (default: 0)"),
    aug_splits: int = typer.Option(0, help="Number of augmentation splits (default: 0, valid: 0 or >=2)"),
    jsd_loss: bool = typer.Option(False, help="Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`."),
    bce_loss: bool = typer.Option(False, help="Enable BCE loss w/ Mixup/CutMix use."),
    bce_sum: bool = typer.Option(False, help="Sum over classes when using BCE loss."),
    bce_target_thresh: float = typer.Option(None, help="Threshold for binarizing softened BCE targets (default: None, disabled)."),
    bce_pos_weight: float = typer.Option(None, help="Positive weighting for BCE loss."),
    reprob: float = typer.Option(0.0, help="Random erase prob (default: 0.)"),
    remode: str = typer.Option("pixel", help='Random erase mode (default: "pixel")'),
    recount: int = typer.Option(1, help="Random erase count (default: 1)"),
    resplit: bool = typer.Option(False, help="Do not random erase first (clean) augmentation split"),
    mixup: float = typer.Option(0.0, help="Mixup alpha, mixup enabled if > 0. (default: 0.)"),
    cutmix: float = typer.Option(0.0, help="Cutmix alpha, cutmix enabled if > 0. (default: 0.)"),
    cutmix_minmax: List[float] = typer.Option(None, help="Cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)"),
    mixup_prob: float = typer.Option(1.0, help="Probability of performing mixup or cutmix when either/both is enabled"),
    mixup_switch_prob: float = typer.Option(0.5, help="Probability of switching to cutmix when both mixup and cutmix enabled"),
    mixup_mode: str = typer.Option("batch", help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'),
    mixup_off_epoch: int = typer.Option(0, help="Turn off mixup after this epoch, disabled if 0 (default: 0)"),
    smoothing: float = typer.Option(0.1, help="Label smoothing (default: 0.1)"),
    train_interpolation: str = typer.Option("random", help='Training interpolation (random, bilinear, bicubic default: "random")'),
    drop: float = typer.Option(0.0, help="Dropout rate (default: 0.)"),
    drop_connect: float = typer.Option(None, help="Drop connect rate, DEPRECATED, use drop-path (default: None)"),
    drop_path: float = typer.Option(None, help="Drop path rate (default: None)"),
    drop_block: float = typer.Option(None, help="Drop block rate (default: None)"),

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    bn_momentum: float = typer.Option(None, help="BatchNorm momentum override (if not None)"),
    bn_eps: float = typer.Option(None, help="BatchNorm epsilon override (if not None)"),
    sync_bn: bool = typer.Option(False, help="Enable NVIDIA Apex or Torch synchronized BatchNorm."),
    dist_bn: str = typer.Option("reduce", help='Distribute BatchNorm stats between nodes ("broadcast", "reduce", or "")'),
    split_bn: bool = typer.Option(False, help="Enable separate BN layers per augmentation split"),

    # Model Exponential Moving Average
    model_ema: bool = typer.Option(False, help="Enable tracking moving average of model weights."),
    model_ema_force_cpu: bool = typer.Option(False, help="Force EMA to be tracked on CPU, rank=0 node only."),
    model_ema_decay: float = typer.Option(0.9998, help="Decay factor for model weights moving average (default: 0.9998)"),
    model_ema_warmup: bool = typer.Option(False, help="Enable warmup for model EMA decay"),

    # Misc
    seed: int = typer.Option(42, help="Random seed (default: 42)"),
    worker_seeding: str = typer.Option("all", help="Worker seed mode (default: all)"),
    log_interval: int = typer.Option(50, help="Number of batches to wait before logging training status"),
    recovery_interval: int = typer.Option(0, help="Number of batches to wait before writing recovery checkpoint"),
    checkpoint_hist: int = typer.Option(10, help="Number of checkpoints to keep (default: 10)"),
    workers: int = typer.Option(4, help="Number of training processes to use (default: 4)"),
    save_images: bool = typer.Option(False, help="Save images of input batches every log interval for debugging"),
    pin_mem: bool = typer.Option(False, help="Pin CPU memory in DataLoader for more efficient transfer to GPU."),
    no_prefetcher: bool = typer.Option(False, help="Disable fast prefetcher"),
    output: str = typer.Option("", help="Path to output folder (default: current dir)"),
    experiment: str = typer.Option("", help="Name of train experiment, sub-folder for output"),
    eval_metric: str = typer.Option("top1", help="Best metric (default: 'top1')"),
    tta: int = typer.Option(0, help="Test/inference time augmentation factor (default: 0)"),
    use_multi_epochs_loader: bool = typer.Option(False, help="Use multi-epochs-loader to save time at the beginning of every epoch"),
    log_wandb: bool = typer.Option(False, help="Log training and validation metrics to wandb"),
    wandb_project: str = typer.Option(None, help="Wandb project name"),
    wandb_tags: List[str] = typer.Option([], help="Wandb tags"),
    wandb_resume_id: str = typer.Option("", help="If resuming a run, the id of the run in wandb"),
    ):

    # Initialize an empty dictionary for arguments
    args_dict = {}

    # If there's a config file, load it and update the defaults
    if config:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
            args_dict.update(cfg)

    # Capture the command-line arguments
    for key, value in locals().items():
        # Add them to the args_dict
        if key != "config":
            args_dict[key] = value

    # Cache the arguments as a text string to save them later
    args_text = yaml.safe_dump(args_dict, default_flow_style=False)

    args = SimpleNamespace(**args_dict)

    args.model_kwargs = {key:value for kv in args.model_kwargs for key, value in [kv.split('=')]}
    args.opt_kwargs = {key:value for kv in args.opt_kwargs for key, value in [kv.split('=')]}

    utils.setup_default_logging()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning('float16 is not recommended for training, for half precision bfloat16 is recommended.')

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )


    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    )
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.weight_decay  # this isn't stored in optimizer.defaults
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
        )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type in ('cuda',) and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)

    if args.data and not args.data_dir:
        args.data_dir = args.data
    # create the train and eval datasets
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
        #trust_remote_code=args.dataset_trust_remote_code,
    )

    if args.val_split:
        dataset_eval = create_dataset(
            args.dataset,
            root=args.data_dir,
            split=args.val_split,
            is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            input_img_mode=input_img_mode,
            input_key=args.input_key,
            target_key=args.target_key,
            num_samples=args.val_num_samples,
            #trust_remote_code=args.dataset_trust_remote_code,
        )

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        use_prefetcher=args.prefetcher,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    loader_eval = None
    if args.val_split:
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=eval_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            img_dtype=model_dtype or torch.float32,
            device=device,
            use_prefetcher=args.prefetcher,
        )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,

            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        if args.log_wandb:
            if has_wandb:
                assert not args.wandb_resume_id or args.resume
                wandb.init(
                    project=args.wandb_project,
                    name=exp_name,
                    config=args,
                    tags=args.wandb_tags,
                    resume="must" if args.wandb_resume_id else None,
                    id=args.wandb_resume_id if args.wandb_resume_id else None,
                )
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        if args.warmup_prefix:
            sched_explain = '(warmup_epochs + epochs + cooldown_epochs). Warmup added to total when warmup_prefix=True'
        else:
            sched_explain = '(epochs + cooldown_epochs). Warmup within epochs when warmup_prefix=False'
        _logger.info(
            f'Scheduled epochs: {num_epochs} {sched_explain}. '
            f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                    model_dtype=model_dtype,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, latest_metric)

            latest_results = {
                'epoch': epoch,
                'train': train_metrics,
            }
            if eval_metrics is not None:
                latest_results['validation'] = eval_metrics
            results.append(latest_results)

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        # log best metric as tracked by checkpoint saver
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if utils.is_primary(args):
        # for parsable results display, dump top-10 summaries to avoid excess console spam
        display_results = sorted(
            results,
            key=lambda x: x.get('validation', x.get('train')).get(eval_metric, 0),
            reverse=decreasing_metric,
        )
        print(f'--result\n{json.dumps(display_results[-10:], indent=4)}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        model_ema=None,
        mixup_fn=None,
        num_updates_total=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device=device, dtype=model_dtype), target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            if args.distributed:
                # synchronize current step and avg loss, each process keeps its own running avg
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        # synchronize avg loss, each process keeps its own running avg
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()
    return OrderedDict([('loss', loss_avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        model_dtype=None,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == "npu":
                torch.npu.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics

if __name__ == '__main__':
    app()
