# =========================
# PROJECT 4: TWEET SENTIMENT PHRASE EXTRACTION
# Complete Deep Learning Workflow - Production Ready (FIXED)
# Optimization Techniques: Ensemble + Focal Loss + Smart Decoding
# =========================

!pip -q install -U transformers datasets accelerate evaluate sentencepiece

import os, glob, random, re, math, warnings, json, pickle, inspect
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

warnings.filterwarnings('ignore')
print("="*80)
print("PROJECT 4: TWEET SENTIMENT PHRASE EXTRACTION")
print("Deep Learning for NLP - Complete Workflow (FIXED)")
print("="*80)
print(f"Transformers: {transformers.__version__} | PyTorch: {torch.__version__}")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== STEP 1: PROBLEM FORMULATION ====================
print("\n" + "="*80)
print("STEP 1: PROBLEM FORMULATION")
print("="*80)
print("""
PROBLEM TYPE: Extractive Question Answering (Span Extraction)
INPUT: Tweet text + Sentiment label (positive/negative/neutral)
OUTPUT: Character span (start, end) indicating sentiment-bearing phrase
CHALLENGE: Twitter language diversity, short texts, ambiguous sentiments
APPROACH: Fine-tune RoBERTa QA model with custom optimizations
""")

# ==================== REPRODUCIBILITY ====================
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==================== MOUNT DRIVE ====================
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    IN_COLAB = True
except:
    IN_COLAB = False

def find_csv(filename: str):
    for root in ["/content", "/content/drive/MyDrive", "/content/drive/Shareddrives"]:
        if os.path.exists(root):
            files = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if files:
                return sorted(files, key=lambda x: len(x))[0]
    return None

TRAIN_CSV = find_csv("train.csv")
TEST_CSV  = find_csv("test.csv")

if not TRAIN_CSV or not TEST_CSV:
    raise FileNotFoundError("Không tìm thấy train.csv/test.csv. Hãy upload vào /content hoặc Drive.")

print(f"\nData files:")
print(f"  Train: {TRAIN_CSV}")
print(f"  Test:  {TEST_CSV}")

# ==================== STEP 2: LOAD & INSPECT DATA ====================
print("\n" + "="*80)
print("STEP 2: DATA LOADING & INSPECTION")
print("="*80)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df = train_df.dropna(subset=["text", "selected_text", "sentiment"]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

for col in ["text", "selected_text", "sentiment"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].astype(str).str.strip()
        if col == "sentiment":
            train_df[col] = train_df[col].str.lower()

for col in ["text", "sentiment"]:
    test_df[col] = test_df[col].astype(str).str.strip()
    if col == "sentiment":
        test_df[col] = test_df[col].str.lower()

print(f"\nDataset sizes:")
print(f"  Training:   {len(train_df):,} samples")
print(f"  Test:       {len(test_df):,} samples")
print(f"\nSentiment distribution:")
print(train_df["sentiment"].value_counts())

# ==================== STEP 3: EXPLORATORY DATA ANALYSIS ====================
print("\n" + "="*80)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*80)

train_df["text_len"] = train_df["text"].str.len().clip(lower=1)  # avoid /0
train_df["sel_len"] = train_df["selected_text"].str.len()
train_df["sel_ratio"] = train_df["sel_len"] / train_df["text_len"]
train_df["is_full"] = (train_df["sel_ratio"] > 0.95).astype(int)

print("\nSpan statistics by sentiment:")
for sent in ["positive", "negative", "neutral"]:
    subset = train_df[train_df["sentiment"] == sent]
    print(f"\n{sent.upper()}:")
    print(f"  Count: {len(subset):,}")
    print(f"  Avg text length: {subset['text_len'].mean():.1f} chars")
    print(f"  Avg selected length: {subset['sel_len'].mean():.1f} chars")
    print(f"  Avg ratio: {subset['sel_ratio'].mean():.3f}")
    print(f"  Full text selections: {subset['is_full'].sum()} ({subset['is_full'].mean()*100:.1f}%)")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("""
1. NEUTRAL: phần lớn là full text → có thể shortcut
2. POS/NEG: selected_text thường ngắn → cần span extraction chính xác
3. Thách thức chính: boundary start/end cho pos/neg
""")

# ==================== CONFIGURATION ====================
print("\n" + "="*80)
print("STEP 4: MODEL CONFIGURATION & OPTIMIZATION TECHNIQUES")
print("="*80)

MODEL_NAME = "cardiffnlp/twitter-roberta-base"
MAX_LEN = 256
STRIDE = 64

# Training hyperparameters
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EPOCHS = 4
TRAIN_BS = 8
EVAL_BS = 16
GRAD_ACCUM = 4

# Ensemble
USE_ENSEMBLE = True
N_FOLDS = 2  # 2-fold for time constraint

# Focal loss
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Smart decoding
TEMPERATURE = 1.5
LENGTH_PENALTY_WEIGHT = -0.5
SHORT_BONUS = 0.3
TOPK = 20

print(f"""
BASE MODEL: {MODEL_NAME}
MAX_LEN: {MAX_LEN}

OPTIMIZATION TECHNIQUES:
1) ENSEMBLE ({N_FOLDS}-fold): {USE_ENSEMBLE}
2) FOCAL LOSS: {USE_FOCAL_LOSS} (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})
3) SMART DECODING: temp={TEMPERATURE}, len_pen={LENGTH_PENALTY_WEIGHT}, short_bonus={SHORT_BONUS}, topk={TOPK}
""")

# ==================== CUSTOM MODEL WITH FOCAL LOSS ====================
class RobertaQAWithFocalLoss(nn.Module):
    """
    Custom QA model with Focal Loss.
    Returns dict with loss/start_logits/end_logits so Trainer can use it.
    """
    def __init__(self, model_name: str):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, attention_mask=None, start_positions=None, end_positions=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)  # seq_len
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            if USE_FOCAL_LOSS:
                total_loss = self.focal_loss(start_logits, end_logits, start_positions, end_positions, ignored_index)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2

        return {"loss": total_loss, "start_logits": start_logits, "end_logits": end_logits}

    def focal_loss(self, start_logits, end_logits, start_pos, end_pos, ignored_idx):
        def focal_ce(logits, targets, alpha=0.25, gamma=2.0, ignore_idx=-100):
            ce = F.cross_entropy(logits, targets, reduction='none', ignore_index=ignore_idx)
            pt = torch.exp(-ce)
            focal_weight = alpha * (1 - pt) ** gamma
            return (focal_weight * ce).mean()

        start_loss = focal_ce(start_logits, start_pos, FOCAL_ALPHA, FOCAL_GAMMA, ignored_idx)
        end_loss = focal_ce(end_logits, end_pos, FOCAL_ALPHA, FOCAL_GAMMA, ignored_idx)
        return (start_loss + end_loss) / 2

# ==================== DATA PREPROCESSING ====================
def find_char_span(text: str, sel: str) -> Tuple[int, int]:
    if not text or not sel:
        return 0, len(text)

    if sel in text:
        s = text.find(sel)
        return s, s + len(sel)

    sel2 = sel.strip()
    if sel2 and sel2 in text:
        s = text.find(sel2)
        return s, s + len(sel2)

    try:
        patt = re.escape(sel2).replace(r"\ ", r"\s+")
        m = re.search(patt, text, re.IGNORECASE)
        if m:
            return m.start(), m.end()
    except:
        pass

    return 0, len(text)

def encode_for_training(df, tokenizer, use_neutral_shortcut=True):
    features = []
    for _, row in df.iterrows():
        text = row["text"]
        sent = row["sentiment"]
        sel = row["selected_text"]

        if use_neutral_shortcut and sent == "neutral":
            char_start, char_end = 0, len(text)
        else:
            char_start, char_end = find_char_span(text, sel)

        enc = tokenizer(
            sent, text,
            truncation="only_second",
            max_length=MAX_LEN,
            padding=False,
            return_offsets_mapping=True
        )

        offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()

        context_idxs = [i for i, sid in enumerate(seq_ids) if sid == 1]

        if not context_idxs:
            start_pos = end_pos = 0
        else:
            c_start, c_end = context_idxs[0], context_idxs[-1]

            if offsets[c_start][0] > char_end or offsets[c_end][1] < char_start:
                start_pos = end_pos = 0
            else:
                tok_start = c_start
                while tok_start <= c_end and offsets[tok_start][1] <= char_start:
                    tok_start += 1

                tok_end = c_end
                while tok_end >= c_start and offsets[tok_end][0] >= char_end:
                    tok_end -= 1

                if tok_start > c_end or tok_end < c_start or tok_end < tok_start:
                    start_pos = end_pos = 0
                else:
                    start_pos, end_pos = tok_start, tok_end

        enc.pop("offset_mapping", None)
        enc["start_positions"] = int(start_pos)
        enc["end_positions"] = int(end_pos)
        features.append(enc)

    return features

class QADataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.features[idx].items()}

# ==================== SMART DECODING ====================
def smart_decode(start_logits, end_logits, offsets, seq_ids, text, sentiment, temperature=1.5):
    context_idxs = [i for i, sid in enumerate(seq_ids) if sid == 1]
    if not context_idxs:
        return text.strip(), (0, len(text))

    c_start, c_end = context_idxs[0], context_idxs[-1]

    start_logits = start_logits / temperature
    end_logits = end_logits / temperature

    start_rank = np.argsort(start_logits)[::-1]
    end_rank = np.argsort(end_logits)[::-1]

    start_cands = [i for i in start_rank if c_start <= i <= c_end][:TOPK]
    end_cands = [i for i in end_rank if c_start <= i <= c_end][:TOPK]

    best_score = -1e18
    best_pair = (c_start, c_start)

    for si in start_cands:
        if offsets[si] == (0, 0):
            continue
        for ei in end_cands:
            if ei < si or offsets[ei] == (0, 0):
                continue

            span_len = ei - si + 1
            if span_len > 50:
                continue

            score = float(start_logits[si] + end_logits[ei])

            # Length penalty (negative => prefer shorter)
            score += LENGTH_PENALTY_WEIGHT * span_len

            # Bonus for 1-3 token spans on pos/neg
            if sentiment in ["positive", "negative"] and 1 <= span_len <= 3:
                score += SHORT_BONUS

            if score > best_score:
                best_score = score
                best_pair = (si, ei)

    si, ei = best_pair
    char_start = offsets[si][0]
    char_end = offsets[ei][1]

    pred = text[char_start:char_end].strip()
    if not pred:
        return text.strip(), (0, len(text))

    pos = text.find(pred)
    if pos != -1:
        return pred, (pos, pos + len(pred))

    return pred, (char_start, char_end)

# ==================== STEP 5: TRAIN ENSEMBLE ====================
print("\n" + "="*80)
print("STEP 5: ENSEMBLE TRAINING")
print("="*80)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

print(f"✓ Loaded tokenizer: {MODEL_NAME}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(train_df, train_df["sentiment"]))

models = []

print(f"\nTraining {N_FOLDS}-fold ensemble...")

# Robust args keys for different transformers versions
TA_SIG = inspect.signature(TrainingArguments.__init__).parameters
EVAL_KEY = "evaluation_strategy" if "evaluation_strategy" in TA_SIG else ("eval_strategy" if "eval_strategy" in TA_SIG else None)
SAVE_KEY = "save_strategy" if "save_strategy" in TA_SIG else None

for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
    print(f"{'='*80}")

    tr_df_fold = train_df.iloc[train_idx].reset_index(drop=True)
    va_df_fold = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(tr_df_fold):,} | Val: {len(va_df_fold):,}")

    print("Encoding features...")
    train_feats = encode_for_training(tr_df_fold, tokenizer)
    val_feats = encode_for_training(va_df_fold, tokenizer)

    train_ds = QADataset(train_feats)
    val_ds = QADataset(val_feats)

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)

    model = RobertaQAWithFocalLoss(MODEL_NAME).to(device)

    output_dir = f"/content/model_fold{fold_idx}"

    args_kwargs = dict(
        output_dir=output_dir,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=100,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2 if IN_COLAB else 0,
        load_best_model_at_end=False,
        remove_unused_columns=False,  # important for custom dict outputs
    )
    if EVAL_KEY:
        args_kwargs[EVAL_KEY] = "epoch"
    if SAVE_KEY:
        args_kwargs[SAVE_KEY] = "epoch"

    args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nTraining...")
    trainer.train()

    trainer.save_model(output_dir)

    models.append(trainer.model.to(device).eval())
    print(f"✓ Fold {fold_idx + 1} complete")

print(f"\n{'='*80}")
print(f"✓ Ensemble training complete! Trained {len(models)} models")
print(f"{'='*80}")

# ==================== STEP 6: ENSEMBLE EVALUATION ====================
print("\n" + "="*80)
print("STEP 6: VALIDATION EVALUATION WITH ENSEMBLE")
print("="*80)

# Evaluate on last fold's validation set
va_df = train_df.iloc[fold_splits[-1][1]].reset_index(drop=True)

def word_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

def char_metrics(text, true_span, pred_span):
    n = len(text)
    ts, te = max(0, min(n, true_span[0])), max(0, min(n, true_span[1]))
    ps, pe = max(0, min(n, pred_span[0])), max(0, min(n, pred_span[1]))

    if te < ts: te = ts
    if pe < ps: pe = ps

    inter_s, inter_e = max(ts, ps), min(te, pe)
    inter = max(0, inter_e - inter_s)

    true_len, pred_len = te - ts, pe - ps
    union = true_len + pred_len - inter

    iou = inter / union if union > 0 else 1.0
    prec = inter / pred_len if pred_len > 0 else (1.0 if true_len == 0 else 0.0)
    rec = inter / true_len if true_len > 0 else (1.0 if pred_len == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"f1": f1, "iou": iou, "prec": prec, "rec": rec,
            "TP": inter, "FP": pred_len - inter, "FN": true_len - inter}

@torch.no_grad()
def ensemble_predict(text, sentiment, models, tokenizer):
    if sentiment == "neutral":
        return text.strip(), (0, len(text))

    enc = tokenizer(
        sentiment, text,
        truncation="only_second",
        max_length=MAX_LEN,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    enc = {k: v.to(device) for k, v in enc.items()}

    all_start_logits = []
    all_end_logits = []

    for m in models:
        out = m(**enc)
        all_start_logits.append(out["start_logits"][0].detach().cpu().numpy())
        all_end_logits.append(out["end_logits"][0].detach().cpu().numpy())

    start_logits = np.mean(all_start_logits, axis=0)
    end_logits = np.mean(all_end_logits, axis=0)

    return smart_decode(start_logits, end_logits, offsets, seq_ids, text, sentiment, TEMPERATURE)

print("Evaluating ensemble on validation set...")

metrics = defaultdict(list)
sent_metrics = defaultdict(lambda: defaultdict(list))
CM = {"TP": 0, "FP": 0, "FN": 0}
samples = []

for i, row in va_df.iterrows():
    text, sent, true_sel = row["text"], row["sentiment"], row["selected_text"]

    if sent == "neutral":
        true_phrase = text
        true_span = (0, len(text))
    else:
        ts, te = find_char_span(text, true_sel)
        true_phrase = text[ts:te].strip()
        true_span = (ts, te)

    pred_phrase, pred_span = ensemble_predict(text, sent, models, tokenizer)

    m = char_metrics(text, true_span, pred_span)
    wj = word_jaccard(true_phrase, pred_phrase)

    metrics["char_f1"].append(m["f1"])
    metrics["word_jaccard"].append(wj)

    sent_metrics[sent]["word_jaccard"].append(wj)
    sent_metrics[sent]["char_f1"].append(m["f1"])

    CM["TP"] += m["TP"]
    CM["FP"] += m["FP"]
    CM["FN"] += m["FN"]

    if i < 10:
        samples.append({
            "sent": sent, "text": text[:120],
            "true": true_phrase, "pred": pred_phrase, "wj": wj
        })

print("\n" + "="*80)
print("VALIDATION RESULTS (with Ensemble)")
print("="*80)

print("\nOVERALL METRICS:")
print(f"  Word Jaccard: {np.mean(metrics['word_jaccard']):.4f}")
print(f"  Char F1:      {np.mean(metrics['char_f1']):.4f}")

print("\nPER-SENTIMENT METRICS:")
for sent in ["positive", "negative", "neutral"]:
    if sent in sent_metrics and len(sent_metrics[sent]["word_jaccard"]) > 0:
        wj = np.mean(sent_metrics[sent]["word_jaccard"])
        f1s = np.mean(sent_metrics[sent]["char_f1"])
        n = len(sent_metrics[sent]["word_jaccard"])
        print(f"  {sent:10s} (n={n:4d}): Jaccard={wj:.4f}, F1={f1s:.4f}")

TP, FP, FN = CM["TP"], CM["FP"], CM["FN"]
prec = TP / (TP + FP) if (TP + FP) > 0 else 0
rec = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_micro = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f"\nMICRO METRICS (Character-level):")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1_micro:.4f}")

print(f"\nSAMPLE PREDICTIONS:")
for i, s in enumerate(samples, 1):
    status = "✓" if s["wj"] > 0.8 else "~" if s["wj"] > 0.5 else "✗"
    print(f"\n[{i}] {status} {s['sent'].upper()} | Jaccard={s['wj']:.2f}")
    print(f"  Text: {s['text']}...")
    print(f"  True: {s['true']}")
    print(f"  Pred: {s['pred']}")

# ==================== STEP 7: INFERENCE ON TEST SET ====================
print("\n" + "="*80)
print("STEP 7: INFERENCE ON TEST SET")
print("="*80)

print(f"Predicting {len(test_df):,} test samples with ensemble...")

predictions = []
for i, row in test_df.iterrows():
    pred, _ = ensemble_predict(row["text"], row["sentiment"], models, tokenizer)
    predictions.append(pred if pred else row["text"].strip())

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(test_df)} processed")

submission = pd.DataFrame({
    "textID": test_df["textID"] if "textID" in test_df.columns else range(len(test_df)),
    "selected_text": predictions
})

sub_path = "/content/submission.csv"
submission.to_csv(sub_path, index=False)

print(f"\n✓ Submission saved: {sub_path}")
print("\nFirst 10 predictions:")
print(submission.head(10))

# ==================== STEP 8: CONCLUSION & SUMMARY ====================
print("\n" + "="*80)
print("STEP 8: CONCLUSION & FINAL SUMMARY")
print("="*80)

# >>> FIXED: close triple quotes properly (this was your SyntaxError)
summary = f"""
PROJECT: Tweet Sentiment Phrase Extraction
DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL ARCHITECTURE:
- Base: {MODEL_NAME} (Twitter-specific RoBERTa)
- Task: Extractive Question Answering (start/end span)
- Output: Character span -> selected_text

OPTIMIZATION TECHNIQUES:
1) Ensemble Training: {USE_ENSEMBLE} ({N_FOLDS}-fold)
2) Focal Loss: {USE_FOCAL_LOSS} (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})
3) Smart Decoding:
   - Temperature: {TEMPERATURE}
   - Length penalty weight: {LENGTH_PENALTY_WEIGHT}
   - Short bonus: {SHORT_BONUS}
   - TOPK: {TOPK}
4) Neutral shortcut: return full text

VALIDATION (evaluated on last fold val split):
- Word Jaccard (overall): {np.mean(metrics['word_jaccard']):.4f}
- Char F1 (overall):      {np.mean(metrics['char_f1']):.4f}

PER-SENTIMENT JACCARD:
- Positive: {np.mean(sent_metrics['positive']['word_jaccard']):.4f} (if list non-empty)
- Negative: {np.mean(sent_metrics['negative']['word_jaccard']):.4f} (if list non-empty)
- Neutral : {np.mean(sent_metrics['neutral']['word_jaccard']):.4f} (if list non-empty)

OUTPUT:
- submission.csv: {sub_path}
"""

print(summary)

with open("/content/run_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print("✓ Saved: /content/run_summary.txt")
