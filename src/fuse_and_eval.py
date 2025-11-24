# src/fuse_and_eval.py
import argparse, os, pandas as pd
from sklearn.metrics import precision_recall_fscore_support

LABELS = ["Happy","Sad","Angry","Fearful","Disgust","Surprised","Neutral","Mixed"]

def load_preds(path):
    return pd.read_csv(path)

def fuse_labels(text_df, vision_df):
    fused = []
    n = min(len(text_df), len(vision_df))
    for i in range(n):
        trow = text_df.iloc[i]
        vrow = vision_df.iloc[i]
        tlabel = trow.get("pred")
        vlabel = vrow.get("pred")
        if tlabel == vlabel:
            fused_label = tlabel
        else:
            # fallback: prefer textual label (dialogue-heavy dataset)
            fused_label = tlabel if pd.notna(tlabel) else vlabel
        fused.append({"index": i, "text_pred": tlabel, "vision_pred": vlabel, "fused": fused_label})
    return pd.DataFrame(fused)

def compute_overall_metrics(golds, preds, labels):
    p, r, f, _ = precision_recall_fscore_support(golds, preds, labels=labels, zero_division=0)
    metrics = pd.DataFrame({"Metric": ["Precision","Recall","F1 Score"]})
    for i,l in enumerate(labels):
        metrics[l] = [round(p[i]*100,2), round(r[i]*100,2), round(f[i]*100,2)]
    return metrics

def main(text_preds, vision_preds, out_metrics):
    tdf = load_preds(text_preds)
    vdf = load_preds(vision_preds)
    fused_df = fuse_labels(tdf, vdf)
    os.makedirs(os.path.dirname(out_metrics) or ".", exist_ok=True)
    fused_df.to_csv("results/fused_predictions.csv", index=False)

    # If gold labels are available in text preds (same length), compute metrics
    if "gold" in tdf.columns or "emotion" in tdf.columns:
        golds = tdf.get("gold") if "gold" in tdf.columns else tdf.get("emotion")
        preds = fused_df["fused"].tolist()[:len(golds)]
        metrics = compute_overall_metrics(golds.tolist(), preds, LABELS)
        metrics.to_csv(out_metrics, index=False)
    else:
        # placeholder to keep pipeline consistent
        pd.DataFrame().to_csv(out_metrics, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_preds", required=True)
    parser.add_argument("--vision_preds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.text_preds, args.vision_preds, args.output)
