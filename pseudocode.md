# ViTexEmo Pseudocode

See the main paper for theory. This pseudocode reproduces the core pipeline:

1. Load textual and visual data (JSON + images)
2. Preprocess textual data and images
3. Extract textual emotions using TextBlob (polarity -> emotion mapping)
4. Extract visual features/emotions using ViT (pretrained)
5. For each instance, compute fused scores: fused = wt * text_vector + wv * image_vector
6. Select fused emotion = argmax(fused)
7. Evaluate performance against ground truth (precision/recall/F1/AUC/MCC)
8. Grid-search weights (wt,wv) on validation set
9. Export predictions and evaluation results
