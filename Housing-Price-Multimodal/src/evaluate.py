import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import HousingDataset
from model import HousingModel

dataset = HousingDataset("data/housing.csv", "data/images")
loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = HousingModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

preds = []
actuals = []

with torch.no_grad():
    for img, tab, price in loader:
        output = model(img, tab)

        preds.extend(output.numpy())
        actuals.extend(price.numpy())

preds = np.array(preds)
actuals = np.array(actuals)

mae = np.mean(np.abs(preds - actuals))
rmse = np.sqrt(np.mean((preds - actuals) ** 2))

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")