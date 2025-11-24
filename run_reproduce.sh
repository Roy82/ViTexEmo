#!/usr/bin/env bash
set -e

echo "============================================"
echo "        ViTexEmo Reproducibility Run        "
echo "============================================"

# Optional: create & activate virtual environment (uncomment if needed)
# python3 -m venv vitexemo_env
# source vitexemo_env/bin/activate
# pip install -r requirements.txt

echo "[1/5] Running preprocessing..."
python src/preprocess.py \
    --input data/sample_texts.json \
    --out data/preprocessed_texts.json

echo "[2/5] Running textual emotion extraction (TextBlob + rule-based mapping)..."
python src/extract_text_emotion.py \
    --input data/preprocessed_texts.json \
    --output results/textual_metrics.csv \
    --preds results/textual_predictions.csv

echo "[3/5] Running visual emotion extraction (ViT)..."
python src/extract_visual_emotion.py \
    --image_dir data/sample_images \
    --output results/visual_metrics.csv \
    --preds results/visual_predictions.csv

echo "[4/5] Running multimodal fusion & evaluation..."
python src/fuse_and_evaluate.py \
    --text_preds results/textual_predictions.csv \
    --vision_preds results/visual_predictions.csv \
    --output results/overall_metrics.csv

echo "[5/5] Verifying results against expected outputs..."
python verify_results.py \
    --expected expected_results \
    --observed results \
    --tol 0.5

echo "============================================"
echo "      Reproduction Completed Successfully!  "
echo "============================================"
echo "If the above step printed 'REPRODUCTION PASSED',"
echo "then the outputs match the published results."
