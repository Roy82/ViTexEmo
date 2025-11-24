# src/extract_text_emotion.py
import argparse, json, os
import pandas as pd
from textblob import TextBlob
from map_polarity_to_emotion import map_polarity_to_emotion
from sklearn.metrics import precision_recall_fscore_support

LABELS = ["Happy","Sad","Angry","Fearful","Disgust","Surprised","Neutral","Mixed"]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_predictions(preds, out_path):
    df = pd.DataFrame(preds)
    df.to_csv(out_path, index=False)

def compute_metrics(golds, preds, labels):
    p, r, f, _ = precision_recall_fscore_support(golds, preds, labels=labels, zero_division=0)
    # AUC/MCC/Kappa require probabilities or special computations; we provide Precision/Recall/F1 table as in Table 15
    metrics = pd.DataFrame({
        "Metric": ["Precision","Recall","F1 Score"],
    })
    for i, l in enumerate(labels):
        metrics[l] = [round(p[i]*100,2), round(r[i]*100,2), round(f[i]*100,2)]
    return metrics

def main(input_json, out_metrics_csv, out_preds_csv):
    data = load_json(input_json)
    preds = []
    golds = []
    gold_ids = []
    for entry in data:
        tid = entry.get("id")
        text = entry.get("text","")
        gold = entry.get("emotion")  # may be None in sample
        polarity = TextBlob(text).sentiment.polarity
        pred = map_polarity_to_emotion(polarity, text)
        preds.append({"id": tid, "text": text, "polarity": polarity, "pred": pred})
        if gold is not None:
            golds.append(gold)
            gold_ids.append(tid)
    os.makedirs(os.path.dirname(out_preds_csv) or ".", exist_ok=True)
    save_predictions(preds, out_preds_csv)

    if golds:
        # align preds with golds by id order
        id_to_pred = {p["id"]: p["pred"] for p in preds}
        aligned_preds = [id_to_pred[iid] for iid in gold_ids]
        metrics = compute_metrics(golds, aligned_preds, LABELS)
        metrics.to_csv(out_metrics_csv, index=False)
    else:
        # create an empty metrics file to keep pipeline consistent
        pd.DataFrame().to_csv(out_metrics_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input json (preprocessed/sample)")
    parser.add_argument("--output", required=True, help="metrics csv output")
    parser.add_argument("--preds", required=True, help="predictions csv output")
    args = parser.parse_args()
    main(args.input, args.output, args.preds)
