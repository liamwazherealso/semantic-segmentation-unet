import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

import utils
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import click


conf = utils.Config()


def train_fn(loader, model, optimizer, loss_fn, scaler, wandb=False):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=conf.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=conf.DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            if wandb:
                wandb.log({'loss': loss})

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(height=conf.IMAGE_HEIGHT, width=conf.IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(height=conf.IMAGE_HEIGHT, width=conf.IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


def _get_loaders(train_transform, val_transforms, pin_memory):
    return get_loaders(
        conf.TRAIN_IMG_DIR,
        conf.TRAIN_MASK_DIR,
        conf.VAL_IMG_DIR,
        conf.VAL_MASK_DIR,
        conf.BATCH_SIZE,
        train_transform,
        val_transforms,
        conf.NUM_WORKERS,
        pin_memory,
    )


@click.command()
@click.option('-w', '--wandb', is_flag=True, default=False, help="Use wandb as experiment tracker")
@click.option('-l', '--load-model', is_flag=True, default=False, help="Load previous checkpoint")
@click.option('--pin-memory',  is_flag=True, default=True, help="Pin memory for data loader")
def main(wandb, load_model, pin_memory):
    if wandb:
        import wandb
        wandb.init(project="semantic-segmentation-unet")

        wandb.config.learning_rate = conf.LEARNING_RATE
        wandb.config.batch_size = conf.BATCH_SIZE
        wandb.config.num_workers = conf.NUM_WORKERS
        wandb.config.image_height = conf.IMAGE_HEIGHT
        wandb.config.image_width = conf.IMAGE_WIDTH
        wandb.config.loaded = load_model

    train_transform = get_train_transforms()

    val_transforms = get_val_transforms()

    model = UNET(in_channels=3, out_channels=1).to(conf.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf.LEARNING_RATE)

    train_loader, val_loader = _get_loaders(train_transform, val_transforms)
    if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=conf.DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(conf.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, wandb=wandb)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=conf.DEVICE, wandb=wandb)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=conf.DEVICE
        )


if __name__ == "__main__":
    main()