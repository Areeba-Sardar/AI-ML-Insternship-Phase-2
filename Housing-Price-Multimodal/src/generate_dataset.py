import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

os.makedirs("data/images", exist_ok=True)

records = []

for i in range(200):

    area = np.random.randint(800, 4000)
    bedrooms = np.random.randint(1, 6)
    bathrooms = np.random.randint(1, 5)
    garage = np.random.randint(0, 3)

    price = (
        area * 120
        + bedrooms * 15000
        + bathrooms * 10000
        + garage * 8000
        + np.random.randint(-20000, 20000)
    )

    img = Image.new("RGB", (128, 128), color=(255, 255, 255))

    draw = ImageDraw.Draw(img)

    width = area // 40

    draw.rectangle(
        [20, 128-width, 108, 108],
        fill=(150, 75, 0)
    )

    image_name = f"house_{i}.png"

    img.save(f"data/images/{image_name}")

    records.append([
        image_name,
        area,
        bedrooms,
        bathrooms,
        garage,
        price
    ])

df = pd.DataFrame(
    records,
    columns=[
        "image_name",
        "area",
        "bedrooms",
        "bathrooms",
        "garage",
        "price"
    ]
)

df.to_csv("data/housing.csv", index=False)

print("Dataset Created Successfully")