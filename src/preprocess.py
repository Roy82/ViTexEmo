"""
preprocess.py
- Minimal preprocessing for text and images
- Expects dataset structure: data/sample_texts.json and data/images/
"""
import os
import json
import argparse
from pathlib import Path
import pandas as pd
import cv2

def normalize_text(s: str) -> str:
    s = s.strip()
    s = " ".join(s.split())
    return s

def preprocess_texts(input_json, out_csv):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for rec in data:
        cid = rec.get("id")
        txt = normalize_text(rec.get("text",""))
        rows.append({"id": cid, "text": txt})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved {len(df)} preprocessed texts to {out_csv}")

def preprocess_images(img_dir, out_dir, size=224):
    os.makedirs(out_dir, exist_ok=True)
    paths = list(Path(img_dir).glob("*"))
    count = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_resized = cv2.resize(img, (size, size))
        out_path = Path(out_dir) / p.name
        cv2.imwrite(str(out_path), img_resized)
        count += 1
    print(f"[+] Processed {count} images into {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="data/sample_texts.json")
    parser.add_argument("--out_csv", default="data/preprocessed_texts.csv")
    parser.add_argument("--img_dir", default="data/images")
    parser.add_argument("--img_out", default="data/images_preprocessed")
    args = parser.parse_args()
    preprocess_texts(args.input_json, args.out_csv)
    preprocess_images(args.img_dir, args.img_out)
