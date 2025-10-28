"""
Produces:
- data/sample_texts.json
- data/images/ (sample colored rectangles saved as png)
- data/gt_subset.csv (optional)
"""
import json, os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import pandas as pd

SAMPLE_DIR = Path("data")
SAMPLE_DIR.mkdir(exist_ok=True)

texts = [
    {"id":"001", "text":"I'm so happy we won the battle!"},
    {"id":"002", "text":"Why did it all go so wrong?"},
    {"id":"003", "text":"I will destroy you!"},
    {"id":"004", "text":"I am not sure how I feel."}
]

with open(SAMPLE_DIR / "sample_texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, indent=2)

img_dir = SAMPLE_DIR / "images"
img_dir.mkdir(exist_ok=True)
colors = [(255,200,200),(50,50,50),(255,50,50),(200,200,200)]
for rec, col in zip(texts, colors):
    img = Image.new("RGB", (256,256), color=col)
    img.save(img_dir / f"{rec['id']}.png")

# ground truth subset
gt = pd.DataFrame([
    {"id":"001","gt_emotion":"Happy"},
    {"id":"002","gt_emotion":"Sad"},
    {"id":"003","gt_emotion":"Angry"},
    {"id":"004","gt_emotion":"Mixed"},
])
gt.to_csv(SAMPLE_DIR / "gt_subset.csv", index=False)
print("[+] Sample data generated in data/")
