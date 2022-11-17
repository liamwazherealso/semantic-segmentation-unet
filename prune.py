import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np

import model
import utils
import train
import wandb


MODEL_NAME = "22_10_16_unet_seal"

meta_data = utils.load_metadata(MODEL_NAME)


def prune_modules(modules, amount=0.3):
    weight_module_types = [nn.BatchNorm2d, nn.Conv2d, torch.nn.modules.conv.ConvTranspose2d]
    bias_module_types = [nn.BatchNorm2d]

    for module in modules:
        if type(module) in weight_module_types:
            prune.random_unstructured(module, name="weight", amount=amount)
        elif type(module) is model.DoubleConv:
            for _module in module.conv:
                if type(_module) in weight_module_types:
                    prune.random_unstructured(_module, name="weight", amount=amount)

    for module in modules:
        if type(module) in bias_module_types:
            prune.random_unstructured(module, name="bias", amount=amount)
        elif type(module) is model.DoubleConv:
            for _module in module.conv:
                if type(_module) in bias_module_types:
                    prune.random_unstructured(_module, name="bias", amount=amount)


if __name__ == "__main__":
    UPSAMPLE = 'upsample'
    DOWNSAMPLE = 'downsample'

    for samp in [UPSAMPLE]:
        wandb.init(project="semantic-segmentation-unet",
                   group=MODEL_NAME,
                   job_type="prune",
                   tags=[f"prune_{samp}"])

        conf = utils.Config()
        wandb.config.learning_rate = conf.LEARNING_RATE
        wandb.config.batch_size = conf.BATCH_SIZE
        wandb.config.num_workers = conf.NUM_WORKERS
        wandb.config.image_height = conf.IMAGE_HEIGHT
        wandb.config.image_width = conf.IMAGE_WIDTH
        wandb.config.prune_sample = samp

        train_transforms = train.get_train_transforms()
        val_transforms = train.get_val_transforms()

        # the original code both loaders are together.
        _, val_loader = train._get_loaders(train_transforms, val_transforms, True)

        prune_wandb = {
            "dice_score": meta_data["dice_score"]
        }

        for amount in np.linspace(0.1, 0.3, num=3):
            prune_wandb["amount"] = amount
            print(f"=> Pruning {samp}")
            unet = model.UNET(in_channels=3, out_channels=1).to(conf.DEVICE)
            print("=> Loading model")
            utils.load_checkpoint(torch.load(f'models/{MODEL_NAME}.pth.tar'), unet)

            print("=> Pruning models")
            modules = unet.ups if samp is UPSAMPLE else unet.downs
            prune_modules(modules, amount=amount)

            print("=> checking_accuracy")
            utils.check_accuracy(val_loader, unet, device=conf.DEVICE, wandb=wandb, prune=prune_wandb)

        wandb.finish()