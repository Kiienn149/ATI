# 62FIT4ATI – Project 4: Tweet Sentiment Phrase Extraction (Span Extraction)

**Course:** 62FIT4ATI – Deep Learning for NLP  
**Topic:** Project 4 – Tweet Sentiment Phrase Extraction  
**Group:** Group <11>  
**Members:** <Nguyen Mai Phuong> – <2201140066>, <Nguyen Trung Kien> – <2201140042>, <Nguyen Xuan Huy> – <2201140036>

---

## 1. Problem Overview

This project solves **Tweet Sentiment Phrase Extraction**, an **extractive question-answering / span extraction** task.

Given:
- **Input:** tweet text + sentiment label (`positive`, `negative`, `neutral`)
- **Output:** a substring (*selected_text*) that best justifies the sentiment

We fine-tune a Transformer model (RoBERTa) to predict the **start/end span positions** of the selected phrase.

---

## 2. Dataset

- Training set: ~27,480 samples  
- Test set: ~3,534 samples  
- CSV format:
  - `textID`, `text`, `sentiment`, `selected_text` (train)
  - `textID`, `text`, `sentiment` (test)

---

## 3. Method

### 3.1 Modeling as Span Extraction (QA-style)
We encode:
- **Query:** sentiment label
- **Context:** tweet text

The model predicts:
- `start_logits`, `end_logits` over token positions  
and we decode the best span to generate `selected_text`.

### 3.2 Pretrained Backbone
We use:
- `cardiffnlp/twitter-roberta-base` (RoBERTa pretrained on Twitter-style text)

---

## 4. Optimization Techniques Used

In our final run, we applied:
- **Warmup** learning rate scheduling (`warmup_ratio`)
- **Weight decay** regularization
- **Gradient accumulation** (to simulate larger batch sizes on Colab GPU)
- **Layer-wise learning rate decay (LLRD)** to stabilize fine-tuning
- **Smart decoding** (top-k span search, temperature scaling, length penalties)
- **Neutral rule for submission:** if sentiment is `neutral`, output full text (based on dataset insight)

> Note: We report metrics with **Neutral rule OFF** (more honest about model capability) and **Neutral rule ON** (matches the submission policy).

---

## 5. Results (Validation)

**Macro metrics (Neutral rule ON):**
- Word Jaccard: ~0.7257  
- Char F1: ~0.7874  

Per-sentiment (macro):
- Positive Jaccard: ~0.55  
- Negative Jaccard: ~0.56  
- Neutral Jaccard: ~0.98 (neutral often equals full text)

We also include:
- EDA plots (length distributions, selection ratios)
- Training/validation loss curves
- Character-level confusion matrix (IN-span vs OUT-span)

---

## 6. How to Run (Recommended: Google Colab)

### 6.1 Install dependencies
The notebook installs required packages:
- `transformers`, `datasets`, `accelerate`, `evaluate`, `sentencepiece`, etc.

### 6.2 Put dataset files
Place `train.csv` and `test.csv` in one of these locations:
- `/content/train.csv` and `/content/test.csv` (upload directly to Colab), or
- Google Drive, e.g. `/content/drive/MyDrive/ATI_Project4/`

The notebook auto-detects file paths.

### 6.3 Run
Open the notebook and **Run all cells** in order.
Outputs:
- `/content/submission.csv`
- `/content/figs/*.png`
- saved artifacts folder (optionally copied to Drive)

---

## 7. Trained Model

Because model checkpoints are large, we provide them via:
- Google Drive / GitHub Release link: <PASTE_MODEL_LINK_HERE>

Saved checkpoint directory example:
- `/content/ati_p4_seed55_final`

---

## 8. Inference Demo

The notebook contains an interactive inference loop:
- Type a tweet
- Type sentiment label (`positive/negative/neutral`)
- Type `Q` to quit

This is used for the presentation demo (Step 7: inference on new data).

---

## 9. Submission Notes 

We submit:
- Slides: https://www.canva.com/design/DAG7pQe38iY/yU8TPcs96X9kJhmJH3FFEg/edit?utm_content=DAG7pQe38iY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

- Github: https://github.com/Kiienn149/ATI

- YouTube video link: 

- Notebook: https://colab.research.google.com/drive/1YEZXZAiQplrhp39M2zhdL3qPXjg27VAO?usp=sharing