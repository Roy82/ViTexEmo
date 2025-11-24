# src/extract_image_emotion.py
import argparse, os, glob
import pandas as pd
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

LABELS = ["Happy","Sad","Angry","Fearful","Disgust","Surprised","Neutral","Mixed"]

def list_images(image_dir):
    exts = ("*.png","*.jpg","*.jpeg")
    files = []
    for e in exts:
        files.extend(sorted(glob.glob(os.path.join(image_dir, e))))
    return files

def predict_images(image_paths, model_name="google/vit-base-patch16-224", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=len(LABELS))
    model.to(device)
    model.eval()
    preds = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        inputs = feature_extractor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = int(torch.argmax(logits, dim=-1).cpu().numpy()[0])
        preds.append({"image": os.path.basename(img_path), "pred": LABELS[pred_idx]})
    return preds

def main(image_dir, out_metrics, out_preds):
    imgs = list_images(image_dir)
    if not imgs:
        print("No images found in", image_dir)
        pd.DataFrame().to_csv(out_metrics, index=False)
        return
    preds = predict_images(imgs)
    os.makedirs(os.path.dirname(out_preds) or ".", exist_ok=True)
    pd.DataFrame(preds).to_csv(out_preds, index=False)
    # metrics: write empty file unless gold labels provided
    pd.DataFrame().to_csv(out_metrics, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True, help="visual metrics csv")
    parser.add_argument("--preds", required=True, help="visual predictions csv")
    args = parser.parse_args()
    main(args.image_dir, args.output, args.preds)
