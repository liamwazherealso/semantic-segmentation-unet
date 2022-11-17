import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import yaml


def load_yaml(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def load_metadata(model_name):
    with open(f"/home/lpieri/ml/semantic_segmentation_unet/models/{model_name}.yaml", 'r') as stream:
        return yaml.safe_load(stream)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


class Config:
    def __init__(self):
        # Hyperparameters etc.
        self.LEARNING_RATE = 1e-4
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 3
        self.NUM_WORKERS = 2
        self.IMAGE_HEIGHT = 160  # 1280 originally
        self.IMAGE_WIDTH = 240  # 1918 originally
        self.TRAIN_IMG_DIR = "/home/lpieri/ml/semantic_segmentation_unet/data/train_images/"
        self.TRAIN_MASK_DIR = "/home/lpieri/ml/semantic_segmentation_unet/data/train_masks/"
        self.VAL_IMG_DIR = "/home/lpieri/ml/semantic_segmentation_unet/data/val_images/"
        self.VAL_MASK_DIR = "/home/lpieri/ml/semantic_segmentation_unet/data/val_masks/"


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda", wandb=False, prune=False):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    model.train()
    if wandb:
        if prune:
            wandb.log({'dice_score_delta': dice_score/len(loader) - prune['dice_score'],
                                 'prune_amount': prune['amount']
                                 })
        else:
            wandb.log({'accuracy': num_correct/num_pixels, 'dice_score': dice_score/len(loader)})

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()