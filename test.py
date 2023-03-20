import argparse
import numpy as np
import os
import pytorch_lightning as pl
import random
import torch
import pandas as pd
from samplers import *
from litmodel import LightModel
import timm
from rvlcdip import RvlDataset
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from losses import Loss

samples_per_class = [19997, 14712, 10823, 7962, 5857, 4306, 3167,
                     2331, 1715, 1261, 928, 682, 502, 369, 271, 200]


def main(args):

    # base_model = timm.create_model(
    #     args.model_name, pretrained=True, num_classes=args.num_class)
    # focal_loss = Loss(
    #     loss_type="focal_loss",
    #     samples_per_class=samples_per_class,
    #     class_balanced=True
    # )

    # model = LightModel(args, base_model, focal_loss)
    model = LightModel.load_from_checkpoint(
        checkpoint_path=args.ckpt_path)

    tar_path = os.path.join(args.root, 'test.tar')
    dataset = RvlDataset(tar_path=tar_path)
    data_loader_test = DataLoader(
        dataset,
        batch_size=32,
        drop_last=False,
        num_workers=32
    )
    trainer = pl.Trainer(devices=1, accelerator="gpu")

    trainer.test(model, ckpt_path=args.ckpt_path, dataloaders=data_loader_test)
    # dfhistory = pd.DataFrame(model.history)
    # dfhistory.to_csv(os.path.join('logs', 'dfhistory.csv'), index=None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None,)
    args = parser.parse_args()
    # args = CfgNode()
    args.root = '/home/minouei/Downloads/datasets/rvl'
    # args.ckpt_path = '/home/minouei/Documents/models/rvl/mycode/eff_dis/logs/lightning_log/version_17/checkpoints/epoch7.ckpt'
    args.model_name = 'efficientnetv2_s'
    args.epochs = 12
    args.num_class = 16
    args.batch_size = 32
    args.lr = 1e-4
    args.weight_decay = 1e-4
    args.CLASSES = ("letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication",
                    "specification", "file folder", "news article", "budget", "invoice", "presentation", "questionnaire", "resume", "memo")

    main(args)

    print("-------------------------")
    print("     Finish All")
    print("-------------------------")
