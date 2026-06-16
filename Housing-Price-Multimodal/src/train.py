import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import HousingDataset
from model import HousingModel

dataset = HousingDataset("data/housing.csv", "data/images")
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = HousingModel()
criterion = nn.L1Loss()  # MAE
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for img, tab, price in loader:
        optimizer.zero_grad()

        output = model(img, tab)
        loss = criterion(output, price)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")

torch.save(model.state_dict(), "model.pth")
print("Model Saved!")