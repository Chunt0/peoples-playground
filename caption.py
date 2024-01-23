import argparse
import os
import random
import sys

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # sys.path.append(os.path.dirname(__file__))
from blip.blip import blip_decoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_FILE_TYPES = (".png", ".jpg", ".jpeg",)

#IMAGE_SIZE = 384
IMAGE_SIZE = 1024

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
)


class ImageLoadingTransformDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor = IMAGE_TRANSFORM(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)

def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]
def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths)) 
    image_paths.sort()
    return image_paths

def run_BLIP(img_dir_path, seed, batch_size, top_p, max_len, min_len):
    recursive = False
    if seed == -1:
        seed = random.randint(0,10e7)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"load images from {img_dir_path}")
    train_data_dir_path = Path(img_dir_path)
    image_paths = glob_images_pathlib(train_data_dir_path, recursive)
    print(f"found {len(image_paths)} images.")

    caption_weights = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"
    print(f"loading BLIP caption: {caption_weights}")
    model = blip_decoder(pretrained=caption_weights, image_size=IMAGE_SIZE, vit="large", med_config="./blip/med_config.json")
    model.eval()
    model = model.to(DEVICE)
    print("BLIP loaded")

    beam_search = False
    num_beams = 1
    caption_extension = ".txt"
    max_data_loader_n_workers = None    
    
    # captioning
    def run_batch(path_imgs):
        imgs = torch.stack([im for _, im in path_imgs]).to(DEVICE)

        with torch.no_grad():
            if beam_search:
                captions = model.generate(
                    imgs, sample=False, num_beams=num_beams, max_length=max_len, min_length=min_len
                )
            else:
                captions = model.generate(
                    imgs, sample=True, top_p=top_p, max_length=max_len, min_length=min_len
                )

        for (image_path, _), caption in zip(path_imgs, captions):
            with open(os.path.splitext(image_path)[0] + caption_extension, "wt", encoding="utf-8") as f:
                f.write(caption + "\n")
                print(image_path, caption)

    if max_data_loader_n_workers is not None:
        dataset = ImageLoadingTransformDataset(image_paths)
        data = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max_data_loader_n_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
        )
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            img_tensor, image_path = data
            if img_tensor is None:
                try:
                    raw_image = Image.open(image_path)
                    if raw_image.mode != "RGB":
                        raw_image = raw_image.convert("RGB")
                    img_tensor = IMAGE_TRANSFORM(raw_image)
                except Exception as e:
                    print(f"Could not load image path: {image_path}, error: {e}")
                    continue

            b_imgs.append((image_path, img_tensor))
            if len(b_imgs) >= batch_size:
                run_batch(b_imgs)
                b_imgs.clear()
    if len(b_imgs) > 0:
        run_batch(b_imgs)

    print("done!")

def load_caption(img_dir_path, img_dict):
    for file in os.listdir(img_dir_path):
        if file.endswith((IMAGE_FILE_TYPES)):
            key = file.split(".")[0]
            image = img_dir_path+file
            caption = img_dir_path+key+".txt"
            img_dict[key] = (image, caption)
    keys = list(img_dict)
    current_key = keys[0]
    image = Image.open(img_dict[current_key][0])
    with open(img_dict[current_key][1], 'r') as file:
        caption = file.readline()
    return img_dict, current_key, image, caption

def update_caption(caption, img_dict, current_key):
    try:
        # Save new caption file
        with open(img_dict[current_key][1], "w") as file:
            file.write(caption)
        
        # Remove previous image/caption pair from dict and repopulate image and caption variables
        img_dict.pop(current_key)
    except:
        current_key = ""
        print("Image Dictionary is empty")
    keys = list(img_dict)
    if len(keys) > 0:
        current_key = keys[0]
        image = Image.open(img_dict[current_key][0])
        with open(img_dict[current_key][1], 'r') as file:
            caption = file.readline()
    else: return None, "No more images", img_dict, current_key
    return image, caption, img_dict, current_key