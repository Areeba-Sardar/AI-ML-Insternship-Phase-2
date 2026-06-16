import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms

class HousingDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.img_dir, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor([
            row["area"],
            row["bedrooms"],
            row["bathrooms"],
            row["garage"]
        ], dtype=torch.float32)

        price = torch.tensor(row["price"], dtype=torch.float32)

        return image, tabular, price