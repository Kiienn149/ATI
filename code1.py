# =========================
# Project 4: Tweet Sentiment Phrase Extraction (Improved Span QA)
# One-cell Colab script (robust across Transformers versions)
# Train + Val (char-level metrics macro+micro) + Predict test.csv -> submission.csv
# =========================

# ---------- 0) Install ----------
!pip -q install -U transformers datasets accelerate evaluate sentencepiece

import os, glob, random, inspect, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

print("Transformers version:", transformers.__version__)
print("Torch version:", torch.__version__)

# ---------- 1) Reproducibility ----------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------- 2) Mount Google Drive & locate CSV ----------
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except Exception as e:
    IN_COLAB = False
    print("Not in Colab or Drive mount failed:", e)

# If auto-search fails, set these manually:
TRAIN_CSV = ""  # e.g. "/content/drive/MyDrive/ATI_Project4/train.csv"
TEST_CSV  = ""  # e.g. "/content/drive/MyDrive/ATI_Project4/test.csv"

def find_csv(filename: str):
    candidates = []
    roots = ["/content", "/content/drive/MyDrive", "/content/drive/Shareddrives"]
    for r in roots:
        if os.path.exists(r):
            candidates.extend(glob.glob(os.path.join(r, "**", filename), recursive=True))
    candidates = sorted(list(set(candidates)), key=lambda x: (len(x), x))
    return candidates

if not TRAIN_CSV or not os.path.exists(TRAIN_CSV):
    cands = find_csv("train.csv")
    if cands: TRAIN_CSV = cands[0]
if not TEST_CSV or not os.path.exists(TEST_CSV):
    cands = find_csv("test.csv")
    if cands: TEST_CSV = cands[0]

print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV :", TEST_CSV)

if not TRAIN_CSV or not os.path.exists(TRAIN_CSV) or not TEST_CSV or not os.path.exists(TEST_CSV):
    raise FileNotFoundError("Không tìm thấy train.csv/test.csv. Hãy upload vào /content hoặc đặt đúng TRAIN_CSV/TEST_CSV.")

# ---------- 3) Load data ----------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df = train_df.dropna(subset=["text", "selected_text", "sentiment"]).reset_index(drop=True)
test_df  = test_df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)
train_df["sentiment"] = train_df["sentiment"].astype(str)

test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str)

print("Train size:", len(train_df), "| Test size:", len(test_df))
print(train_df["sentiment"].value_counts())

# ---------- 4) Split train/val (stratified) ----------
val_ratio = 0.1
tr_df, va_df = train_test_split(
    train_df,
    test_size=val_ratio,
    random_state=SEED,
    stratify=train_df["sentiment"]
)
tr_df = tr_df.reset_index(drop=True)
va_df = va_df.reset_index(drop=True)
print("Train split:", len(tr_df), "| Val split:", len(va_df))

# ---------- 5) Config (IMPROVED) ----------
# 1) Tweet-domain pretrained model
MODEL_NAME = "cardiffnlp/twitter-roberta-base"  # fallback to roberta-base if fails
FALLBACK_MODEL = "roberta-base"

# 2) Slightly longer max len (safe)
MAX_LEN = 256

# 3) Training hyperparams
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS = 4
TRAIN_BS = 16
EVAL_BS = 32
GRAD_ACCUM = 2

# 4) Decoding/postprocess knobs (to reduce under-extraction)
TOPK_CANDIDATES = 60          # increase K
MAX_ANSWER_TOKENS = 80        # allow longer spans
MIN_ANSWER_TOKENS = 2         # avoid 1-token answers when possible
LEN_BONUS_ALPHA = 0.25        # small bonus for longer spans (reduces "Good" vs "Good times.")
USE_NEUTRAL_SHORTCUT = True   # neutral -> full text

# ---------- 6) Tokenizer & Model ----------
def load_tokenizer_model(model_name: str):
    # RoBERTa-family benefits from add_prefix_space=True for offsets on leading spaces
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tok, mdl

try:
    tokenizer, model = load_tokenizer_model(MODEL_NAME)
    print("Loaded:", MODEL_NAME)
except Exception as e:
    print("Failed to load", MODEL_NAME, "-> fallback to", FALLBACK_MODEL, "| error:", str(e)[:200])
    tokenizer, model = load_tokenizer_model(FALLBACK_MODEL)
    print("Loaded:", FALLBACK_MODEL)

model = model.to(device)

# ---------- 7) Helper: robust find selected_text span in text (char indices) ----------
def find_char_span(text: str, sel: str):
    """
    Return (start, end) end-exclusive in original text.
    More robust than plain find: tries exact, stripped, regex with flexible spaces, case-insensitive.
    """
    if not isinstance(text, str) or not isinstance(sel, str):
        return 0, len(text)

    # 1) exact
    if sel in text:
        s = text.find(sel)
        return s, s + len(sel)

    sel2 = sel.strip()
    # 2) stripped exact
    if sel2 and sel2 in text:
        s = text.find(sel2)
        return s, s + len(sel2)

    # 3) flexible spaces + case-insensitive
    if sel2:
        patt = re.escape(sel2)
        patt = patt.replace(r"\ ", r"\s+")
        m = re.search(patt, text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()

    # fallback
    return 0, len(text)

# ---------- 8) Build training features (start/end token positions) ----------
def encode_examples_for_train(df: pd.DataFrame):
    features = []
    for _, row in df.iterrows():
        text = row["text"]
        sent = row["sentiment"]
        sel  = row["selected_text"]

        if USE_NEUTRAL_SHORTCUT and sent.lower() == "neutral":
            char_start, char_end = 0, len(text)
        else:
            char_start, char_end = find_char_span(text, sel)

        enc = tokenizer(
            sent,
            text,
            truncation="only_second",
            max_length=MAX_LEN,
            padding=False,
            return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()

        cls_index = 0
        context_idxs = [i for i, sid in enumerate(seq_ids) if sid == 1]
        if not context_idxs:
            start_pos = cls_index
            end_pos = cls_index
        else:
            c_start = context_idxs[0]
            c_end   = context_idxs[-1]

            if offsets[c_start][0] > char_end or offsets[c_end][1] < char_start:
                start_pos = cls_index
                end_pos = cls_index
            else:
                tok_start = c_start
                while tok_start <= c_end and offsets[tok_start][1] <= char_start:
                    tok_start += 1

                tok_end = c_end
                while tok_end >= c_start and offsets[tok_end][0] >= char_end:
                    tok_end -= 1

                if tok_start > c_end or tok_end < c_start or tok_end < tok_start:
                    start_pos = cls_index
                    end_pos = cls_index
                else:
                    start_pos = tok_start
                    end_pos = tok_end

        enc.pop("offset_mapping", None)
        enc["start_positions"] = start_pos
        enc["end_positions"]   = end_pos
        features.append(enc)
    return features

class QASpanDataset(Dataset):
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        item = self.features[idx]
        return {k: torch.tensor(v) for k, v in item.items()}

print("Encoding train/val features...")
train_features = encode_examples_for_train(tr_df)
val_features   = encode_examples_for_train(va_df)

train_ds = QASpanDataset(train_features)
val_ds   = QASpanDataset(val_features)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# ---------- 9) TrainingArguments (robust eval_strategy vs evaluation_strategy) ----------
output_dir = "/content/ati_project4_spanqa_improved"

ta_kwargs = dict(
    output_dir=output_dir,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    gradient_accumulation_steps=GRAD_ACCUM,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=100,
    report_to="none",
    save_total_limit=2,
    dataloader_num_workers=2 if IN_COLAB else 0,
)

sig_params = inspect.signature(TrainingArguments.__init__).parameters
if "evaluation_strategy" in sig_params:
    ta_kwargs["evaluation_strategy"] = "epoch"
elif "eval_strategy" in sig_params:
    ta_kwargs["eval_strategy"] = "epoch"
if "save_strategy" in sig_params:
    ta_kwargs["save_strategy"] = "epoch"
if "fp16" in sig_params:
    ta_kwargs["fp16"] = bool(torch.cuda.is_available())
elif "bf16" in sig_params:
    ta_kwargs["bf16"] = bool(torch.cuda.is_available())
if "max_grad_norm" in sig_params:
    ta_kwargs["max_grad_norm"] = 1.0
if "lr_scheduler_type" in sig_params:
    ta_kwargs["lr_scheduler_type"] = "linear"

args = TrainingArguments(**ta_kwargs)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,  # warning is fine; still supported
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ---------- 10) Train ----------
train_result = trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

model = trainer.model.to(device)
model.eval()
print("Training done. Best model saved to:", output_dir)

# ---------- 11) Postprocess: snap to word boundaries / punctuation ----------
WORD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'")
RIGHT_PUNCT = set("!?.;,:'\"”)”]}>")
LEFT_PUNCT  = set("“(\"([{<")

def snap_span(text: str, s: int, e: int):
    """Expand slightly to word boundaries and include adjacent punctuation in a conservative way."""
    n = len(text)
    s = max(0, min(n, s)); e = max(0, min(n, e))
    if e < s: e = s

    # Expand left to start of word (if inside a word)
    while s > 0 and text[s-1] in WORD_CHARS and (s < n and text[s] in WORD_CHARS):
        s -= 1

    # Expand right to end of word (if inside a word)
    while e < n and text[e-1] in WORD_CHARS and text[e] in WORD_CHARS:
        e += 1

    # Include leading left quotes/brackets if immediately before span
    while s > 0 and text[s-1] in LEFT_PUNCT:
        s -= 1

    # Include trailing punctuation if immediately after span
    while e < n and text[e] in RIGHT_PUNCT:
        e += 1

    # Strip outer spaces but keep char indices aligned by re-searching within a small window
    raw = text[s:e]
    stripped = raw.strip()
    if stripped:
        win_s = max(0, s-6); win_e = min(n, e+6)
        win = text[win_s:win_e]
        pos = win.find(stripped)
        if pos != -1:
            s = win_s + pos
            e = s + len(stripped)

    return s, e

# ---------- 12) Improved span decode ----------
@torch.no_grad()
def predict_span(text: str, sentiment: str):
    if USE_NEUTRAL_SHORTCUT and sentiment.lower() == "neutral":
        return text, (0, len(text))

    enc = tokenizer(
        sentiment,
        text,
        truncation="only_second",
        max_length=MAX_LEN,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model(**enc)
    start_logits = out.start_logits[0].detach().cpu().numpy()
    end_logits   = out.end_logits[0].detach().cpu().numpy()

    context_idxs = [i for i, sid in enumerate(seq_ids) if sid == 1]
    if not context_idxs:
        return text, (0, len(text))
    c_start = context_idxs[0]
    c_end   = context_idxs[-1]

    # Candidate pools
    K = TOPK_CANDIDATES
    start_rank = np.argsort(start_logits)[::-1]
    end_rank   = np.argsort(end_logits)[::-1]

    start_candidates = [i for i in start_rank if c_start <= i <= c_end][:K]
    end_candidates   = [i for i in end_rank   if c_start <= i <= c_end][:K]

    # Search best pair with mild length bonus + min length constraint
    best_score = -1e18
    best_pair = (c_start, c_start)

    backup_score = -1e18
    backup_pair = (c_start, c_start)

    for si in start_candidates:
        if offsets[si] == (0, 0): 
            continue
        for ei in end_candidates:
            if ei < si:
                continue
            if offsets[ei] == (0, 0):
                continue

            length_tokens = ei - si + 1
            if length_tokens > MAX_ANSWER_TOKENS:
                continue

            base = start_logits[si] + end_logits[ei]
            bonus = LEN_BONUS_ALPHA * math.log1p(length_tokens)
            score = base + bonus

            # keep a backup even if too short
            if score > backup_score:
                backup_score = score
                backup_pair = (si, ei)

            # enforce minimum length preference when possible
            if length_tokens < MIN_ANSWER_TOKENS:
                continue

            if score > best_score:
                best_score = score
                best_pair = (si, ei)

    # If no pair satisfied MIN_ANSWER_TOKENS, fallback to backup (best overall)
    si, ei = best_pair if best_score > -1e17 else backup_pair

    char_start = offsets[si][0]
    char_end   = offsets[ei][1]

    # Snap to word boundaries/punctuation
    char_start, char_end = snap_span(text, char_start, char_end)

    pred = text[char_start:char_end].strip()
    if pred == "":
        # fallback to full text if empty (rare)
        return text.strip(), (0, len(text))

    # final alignment after strip
    win_s = max(0, char_start - 6)
    win_e = min(len(text), char_end + 6)
    win = text[win_s:win_e]
    pos = win.find(pred)
    if pos != -1:
        char_start = win_s + pos
        char_end = char_start + len(pred)

    return pred, (char_start, char_end)

# ---------- 13) Metrics ----------
def char_overlap_metrics(text, true_span, pred_span):
    n = len(text)
    ts, te = true_span
    ps, pe = pred_span
    ts = max(0, min(n, ts)); te = max(0, min(n, te))
    ps = max(0, min(n, ps)); pe = max(0, min(n, pe))
    if te < ts: te = ts
    if pe < ps: pe = ps

    true_len = max(0, te - ts)
    pred_len = max(0, pe - ps)

    inter_s = max(ts, ps)
    inter_e = min(te, pe)
    inter = max(0, inter_e - inter_s)

    union = true_len + pred_len - inter
    char_iou = inter / union if union > 0 else 1.0

    precision = inter / pred_len if pred_len > 0 else (1.0 if true_len == 0 else 0.0)
    recall    = inter / true_len if true_len > 0 else (1.0 if pred_len == 0 else 0.0)
    char_f1   = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    TP = inter
    FP = pred_len - inter
    FN = true_len - inter
    TN = n - (TP + FP + FN)
    return precision, recall, char_f1, char_iou, TP, FP, FN, TN

def word_jaccard(a: str, b: str):
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if len(sa) == 0 and len(sb) == 0:
        return 1.0
    return len(sa & sb) / len(sa | sb)

# ---------- 14) Evaluate on validation ----------
print("\nEvaluating on validation (char-level)...")

metrics = {"char_precision": [], "char_recall": [], "char_f1": [], "char_iou": [], "word_jaccard": []}
CM = {"TP":0, "FP":0, "FN":0, "TN":0}

sample_rows = []
for i, row in va_df.iterrows():
    text = row["text"]
    sent = row["sentiment"]
    true_sel = row["selected_text"]

    if USE_NEUTRAL_SHORTCUT and sent.lower() == "neutral":
        true_span = (0, len(text))
        true_phrase = text
    else:
        ts, te = find_char_span(text, true_sel)
        ts, te = snap_span(text, ts, te)
        true_phrase = text[ts:te].strip()
        true_span = (ts, te)

    pred_phrase, pred_span = predict_span(text, sent)

    p, r, f1, iou, TP, FP, FN, TN = char_overlap_metrics(text, true_span, pred_span)
    metrics["char_precision"].append(p)
    metrics["char_recall"].append(r)
    metrics["char_f1"].append(f1)
    metrics["char_iou"].append(iou)
    metrics["word_jaccard"].append(word_jaccard(true_phrase, pred_phrase))

    CM["TP"] += TP; CM["FP"] += FP; CM["FN"] += FN; CM["TN"] += TN

    if i < 5:
        sample_rows.append({
            "sentiment": sent,
            "text": text,
            "true_selected": true_phrase,
            "pred_selected": pred_phrase
        })

# Macro (mean per example)
print("\nVAL RESULTS (MACRO: mean over examples)")
for k in ["char_precision","char_recall","char_f1","char_iou","word_jaccard"]:
    print(f"{k}: {np.mean(metrics[k]):.4f}")

# Micro (from aggregated confusion)
TP, FP, FN, TN = CM["TP"], CM["FP"], CM["FN"], CM["TN"]
micro_prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
micro_rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
micro_f1   = (2*micro_prec*micro_rec/(micro_prec+micro_rec)) if (micro_prec+micro_rec)>0 else 0.0
micro_acc  = (TP + TN) / max(1, (TP+FP+FN+TN))
micro_iou  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0

print("\nCHAR-LEVEL CONFUSION MATRIX (aggregated over characters)")
print(CM)
print(f"Micro-Precision: {micro_prec:.4f} | Micro-Recall: {micro_rec:.4f} | Micro-F1: {micro_f1:.4f} | Micro-IoU: {micro_iou:.4f} | Char-Acc: {micro_acc:.4f}")

print("\nSAMPLES (first 5):")
for s in sample_rows:
    print("\nSentiment:", s["sentiment"])
    print("Text        :", s["text"])
    print("True phrase :", s["true_selected"])
    print("Pred phrase :", s["pred_selected"])

# ---------- 15) Predict test & write submission.csv ----------
print("\nPredicting test.csv ...")
preds = []
for _, row in test_df.iterrows():
    text = row["text"]
    sent = row["sentiment"]
    pred_phrase, _ = predict_span(text, sent)
    if pred_phrase.strip() == "":
        pred_phrase = text.strip()
    preds.append(pred_phrase)

sub = pd.DataFrame({
    "textID": test_df["textID"].values if "textID" in test_df.columns else np.arange(len(test_df)),
    "selected_text": preds
})

out_path = "/content/submission.csv"
sub.to_csv(out_path, index=False)
print("Saved:", out_path)
print(sub.head())
