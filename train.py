import numpy as np
import os
import pytorch_lightning as pl
import random
import torch
import pandas as pd
from yacs.config import CfgNode
from samplers import *
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import distributed as dist
from dataPipeline import DataPipeline
from losses import Loss
from litmodel import LightModel
import timm
import json

samples_per_class = [19997, 14712, 10823, 7962, 5857, 4306, 3167,
                     2331, 1715, 1261, 928, 682, 502, 369, 271, 200]


def main(args):

    checkpoint_val_loss = ModelCheckpoint(
        monitor="val_loss", mode='min')
    checkpoint_val_acc = ModelCheckpoint(
        monitor="val_acc", mode='max')

    focal_loss = Loss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class,
        class_balanced=True
    )
    base_model = timm.create_model(
        args.model_name, pretrained=True, num_classes=args.num_class)

    model = LightModel(args, base_model, focal_loss)

    tb_logger = TensorBoardLogger(save_dir='logs', name='lightning_log')

    data_module = DataPipeline(args.root_path, args.batch_size)

    # sync_batchnorm = False
    # trainer = pl.Trainer.from_argparse_args(
    #     args,
    #     plugins=DDPPlugin(find_unused_parameters=True,
    #                       num_nodes=args.num_nodes,
    #                       sync_batchnorm=sync_batchnorm),
    #     gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
    #     callbacks=callbacks,
    #     logger=logger,
    #     sync_batchnorm=sync_batchnorm,
    #     replace_sampler_ddp=False,  # use custom sampler
    #     reload_dataloaders_every_epoch=False,  # avoid repeated samples!
    #     weights_summary='full',
    #     profiler=profiler)

    trainer = pl.Trainer(
        callbacks=[checkpoint_val_acc, checkpoint_val_loss],
        devices=1, accelerator="gpu",
        logger=tb_logger,
        max_epochs=args.epochs,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    with open(args.RUN_NAME + 'result.json', 'w') as fp:
        json.dump(model.history, fp)

    save_path = args.RUN_NAME + '.ckpt'
    trainer.save_checkpoint(save_path)


# def init_dist_slurm():
#     proc_id = int(os.environ['SLURM_PROCID'])
#     ntasks = int(os.environ['SLURM_NTASKS'])
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(proc_id % num_gpus)
#     os.environ['WORLD_SIZE'] = str(ntasks)
#     os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
#     os.environ['RANK'] = str(proc_id)
#     dist.init_process_group(backend='nccl')


if __name__ == "__main__":
    # option = parse_arguments()
    # args = update_args(cfg_file=option.cfg_file,
    #                    run_name=option.run_name, seed=option.seed)
    # fix_seed(args.TRAIN.SEED)

    args = CfgNode()
    # args.LOAD_WEIGHT_PATH = False
    args.NUM_GPUS = 1
    args.lr = 4e-4
    args.weight_decay = 1e-4
    # args.TRAIN.MOMENTUM = 0.9
    args.RUN_NAME = 'focal12'
    args.model_name = 'efficientnetv2_s'
    args.epochs = 12
    args.root_path = "/home/minouei/Downloads/datasets/rvl"
    args.CLASSES = ("letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication",
                    "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo")
    args.num_class = 16
    args.batch_size = 32

    main(args)

    print("-------------------------")
    print("     Finish All")
    print("-------------------------")
