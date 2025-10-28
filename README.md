# ViTexEmo: A Multimodal Metadata Framework for Textual, Visual, and Emotional Enrichment in Digital Comic Archives

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17468178.svg)](https://doi.org/10.5281/zenodo.17468178)

This repository contains the core pseudocode, scripts, and documentation for the **ViTexEmo** framework proposed in the research paper:
> Sharma, R., Kukreja, V. & Bhattacharjee, A. (2025). *ViTexEmo: A Multimodal Metadata Framework for Textual, Visual, and Emotional Enrichment in Digital Comic Archives.*

## 📘 Overview
ViTexEmo is an AI-powered metadata framework designed to integrate textual, visual, and emotional information for enhanced management of digital comic archives.  
It automates metadata enrichment using **TextBlob** for textual emotion extraction and **ViT (Vision Transformer)** for visual emotion analysis.  
The framework supports **weighted multimodal fusion**, **semantic enrichment**, and **scalable evaluation** across large datasets.

## 🧩 Core Components
- **Data Preprocessing** – Cleaning and structuring textual and visual inputs (JSON + image files).  
- **Textual Emotion Extraction (TextBlob)** – Sentiment and emotion tagging from comic dialogues.  
- **Visual Emotion Extraction (ViT)** – Emotion recognition from character expressions and scene colors.  
- **Weighted Fusion** – Combines textual and visual emotion scores using optimal weights.  
- **Evaluation** – Computes precision, recall, F1, AUC-ROC, MCC, and consistency metrics.

## 🧠 Methodology Pseudocode
See [`pseudocode.md`](pseudocode.md) for a detailed step-by-step workflow.

## ⚙️ Requirements
See [`requirements.txt`](requirements.txt) for a detailed requirements.

## License

This project is released under the MIT License — see the [LICENSE](LICENSE) file for details.

> **Note:** The `src/` folder is provided as a ZIP archive (`src.zip`) for convenience.  
> Unzip it locally to access all source scripts (`preprocess.py`, `extract_text_emotion.py`, `extract_image_emotion.py`, `fuse_and_eval.py`).




