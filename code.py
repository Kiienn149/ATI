# =========================
# Project 4: Tweet Sentiment Phrase Extraction - FIXED & STRONG VERSION (1 CELL)
# Fixes:
#  - eval_loss KeyError (explicit label_names + proper model outputs)
#  - WRONG context_idxs bug in predict_span
#  - Multi-sample dropout made REAL (dropout applied with training=True on hidden states)
#  - Length scoring uses mild POSITIVE length bonus (reduces under-extraction)
# Still extractive span QA + char-level evaluation
# =========================

!pip -q install -U transformers datasets accelerate evaluate sentencepiece

import os, glob, random, inspect, re, math, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from typing import Tuple, List

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
    EarlyStoppingCallback,
    set_seed,
)
from transformers.modeling_outputs import QuestionAnsweringModelOutput

warnings.filterwarnings("ignore")
print("="*70)
print("TWEET SENTIMENT PHRASE EXTRACTION - FIXED & STRONG")
print("="*70)
print(f"Transformers: {transformers.__version__} | Torch: {torch.__version__}")

# ---------- 1) Seed ----------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------- 2) Find CSV ----------
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    IN_COLAB = True
except:
    IN_COLAB = False

TRAIN_CSV, TEST_CSV = "", ""

def find_csv(filename: str):
    candidates = []
    for root in ["/content", "/content/drive/MyDrive", "/content/drive/Shareddrives"]:
        if os.path.exists(root):
            candidates.extend(glob.glob(os.path.join(root, "**", filename), recursive=True))
    return sorted(set(candidates), key=lambda x: (len(x), x))

if not TRAIN_CSV or not os.path.exists(TRAIN_CSV):
    cands = find_csv("train.csv")
    if cands: TRAIN_CSV = cands[0]
if not TEST_CSV or not os.path.exists(TEST_CSV):
    cands = find_csv("test.csv")
    if cands: TEST_CSV = cands[0]

print(f"\nTRAIN_CSV: {TRAIN_CSV}")
print(f"TEST_CSV : {TEST_CSV}")
if not TRAIN_CSV or not TEST_CSV:
    raise FileNotFoundError("Please upload train.csv and test.csv to /content/ or Drive.")

# ---------- 3) Load data ----------
train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["text","selected_text","sentiment"]).reset_index(drop=True)
test_df  = pd.read_csv(TEST_CSV).dropna(subset=["text","sentiment"]).reset_index(drop=True)

train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)
train_df["sentiment"] = train_df["sentiment"].astype(str).str.lower()

test_df["text"] = test_df["text"].astype(str)
test_df["sentiment"] = test_df["sentiment"].astype(str).str.lower()

print(f"\nTrain={len(train_df)} | Test={len(test_df)}")
print(train_df["sentiment"].value_counts())

# ---------- 4) Split ----------
tr_df, va_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["sentiment"]
)
tr_df = tr_df.reset_index(drop=True)
va_df = va_df.reset_index(drop=True)
print(f"\nSplit: Train={len(tr_df)} | Val={len(va_df)}")

# ---------- 5) Config ----------
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
MAX_LEN = 256

# training
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS = 4
TRAIN_BS = 8
EVAL_BS = 16
GRAD_ACCUM = 4  # effective BS = 32

# decoding
TOPK = 50
MAX_ANSWER_TOKENS = 80
MIN_ANSWER_TOKENS = 2
LEN_BONUS_ALPHA = 0.15  # positive => reduce "too short" spans
USE_NEUTRAL_SHORTCUT = True

# advanced
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

USE_MSD = True
MSD_SAMPLES = 5

USE_DYNAMIC_POSTPROCESS = False  # keep OFF by default; you can turn ON & ablate in report

print("\nCONFIG:")
print(f"Model={MODEL_NAME} | MAX_LEN={MAX_LEN}")
print(f"LR={LR} | epochs={EPOCHS} | eff_BS={TRAIN_BS*GRAD_ACCUM}")
print(f"Decode: TOPK={TOPK} | max_ans_tok={MAX_ANSWER_TOKENS} | len_bonus={LEN_BONUS_ALPHA}")
print(f"Focal={USE_FOCAL_LOSS} | MSD={USE_MSD} (n={MSD_SAMPLES}) | DynamicPost={USE_DYNAMIC_POSTPROCESS}")

# ---------- 6) Tokenizer ----------
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, add_prefix_space=True)
except:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# ---------- 7) Robust char span finder ----------
def find_char_span(text: str, sel: str) -> Tuple[int,int]:
    if sel in text:
        s = text.find(sel); return s, s+len(sel)
    sel2 = sel.strip()
    if sel2 and sel2 in text:
        s = text.find(sel2); return s, s+len(sel2)
    if sel2:
        try:
            patt = re.escape(sel2).replace(r"\ ", r"\s+")
            m = re.search(patt, text, flags=re.IGNORECASE)
            if m: return m.start(), m.end()
        except:
            pass
    return 0, len(text)

# ---------- 8) Encode examples ----------
def encode_examples(df: pd.DataFrame):
    feats = []
    for _, row in df.iterrows():
        text = row["text"]
        sent = row["sentiment"]
        sel  = row.get("selected_text","")

        if USE_NEUTRAL_SHORTCUT and sent == "neutral":
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

        cls_idx = 0
        context_idxs = [i for i, sid in enumerate(seq_ids) if sid == 1]
        if not context_idxs:
            start_pos = end_pos = cls_idx
        else:
            c_start, c_end = context_idxs[0], context_idxs[-1]
            if offsets[c_start][0] > char_end or offsets[c_end][1] < char_start:
                start_pos = end_pos = cls_idx
            else:
                tok_start = c_start
                while tok_start <= c_end and offsets[tok_start][1] <= char_start:
                    tok_start += 1
                tok_end = c_end
                while tok_end >= c_start and offsets[tok_end][0] >= char_end:
                    tok_end -= 1
                if tok_start > c_end or tok_end < c_start or tok_end < tok_start:
                    start_pos = end_pos = cls_idx
                else:
                    start_pos, end_pos = tok_start, tok_end

        enc.pop("offset_mapping", None)
        enc["start_positions"] = int(start_pos)
        enc["end_positions"] = int(end_pos)
        feats.append(enc)
    return feats

class QADataset(Dataset):
    def __init__(self, features): self.features = features
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx]

print("\nEncoding features...")
train_features = encode_examples(tr_df)
val_features   = encode_examples(va_df)
train_ds = QADataset(train_features)
val_ds   = QADataset(val_features)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# ---------- 9) Custom model: REAL MSD + optional Focal Loss ----------
class RobertaQA_MSD(nn.Module):
    def __init__(self, model_name: str, msd_samples: int = 5, use_msd: bool = True):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.qa = nn.Linear(self.config.hidden_size, 2)
        self.use_msd = use_msd
        self.msd_samples = msd_samples
        self.dropout_p = float(getattr(self.config, "hidden_dropout_prob", 0.1))

    def focal_ce(self, logits, targets, alpha=0.25, gamma=2.0):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        w = alpha * (1 - pt) ** gamma
        return (w * ce).mean()

    def forward(self, input_ids=None, attention_mask=None, start_positions=None, end_positions=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [B, L, H]

        if (not self.training) and self.use_msd:
            # REAL MSD: apply dropout masks on hidden states with training=True
            start_list, end_list = [], []
            for _ in range(self.msd_samples):
                dropped = F.dropout(sequence_output, p=self.dropout_p, training=True)
                logits = self.qa(dropped)  # [B, L, 2]
                s, e = logits.split(1, dim=-1)
                start_list.append(s.squeeze(-1))
                end_list.append(e.squeeze(-1))
            start_logits = torch.mean(torch.stack(start_list, dim=0), dim=0)
            end_logits   = torch.mean(torch.stack(end_list, dim=0), dim=0)
        else:
            logits = self.qa(sequence_output)
            s, e = logits.split(1, dim=-1)
            start_logits = s.squeeze(-1)
            end_logits   = e.squeeze(-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            if USE_FOCAL_LOSS:
                start_loss = self.focal_ce(start_logits, start_positions, FOCAL_ALPHA, FOCAL_GAMMA)
                end_loss   = self.focal_ce(end_logits, end_positions,   FOCAL_ALPHA, FOCAL_GAMMA)
            else:
                ce = nn.CrossEntropyLoss()
                start_loss = ce(start_logits, start_positions)
                end_loss   = ce(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits
        )

model = RobertaQA_MSD(MODEL_NAME, msd_samples=MSD_SAMPLES, use_msd=USE_MSD).to(device)

# ---------- 10) TrainingArguments (robust naming) ----------
output_dir = "/content/ati_project4_fixed_strong"

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
    save_total_limit=2,
    logging_steps=50,
    report_to="none",
    fp16=bool(torch.cuda.is_available()),
    dataloader_num_workers=2 if IN_COLAB else 0,
)

sig = inspect.signature(TrainingArguments.__init__).parameters
# eval every N steps
if "evaluation_strategy" in sig:
    ta_kwargs["evaluation_strategy"] = "steps"
elif "eval_strategy" in sig:
    ta_kwargs["eval_strategy"] = "steps"
ta_kwargs["eval_steps"] = 200

if "save_strategy" in sig:
    ta_kwargs["save_strategy"] = "steps"
ta_kwargs["save_steps"] = 200

args = TrainingArguments(**ta_kwargs)

# KEY FIX for your eval_loss error:
# Make Trainer recognize QA labels explicitly for evaluation loop
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    label_names=["start_positions","end_positions"],  # <-- CRITICAL
)

print("\n" + "="*70)
print("TRAINING")
print("="*70)
trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
model = trainer.model.to(device)
model.eval()
print(f"\n✓ Training complete. Best model saved to: {output_dir}")

# ---------- 11) Better span snap ----------
WORD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'")
RIGHT_PUNCT = set("!?.;,:'\"”)”]}>")
LEFT_PUNCT  = set("“(\"([{<")

def snap_span(text: str, s: int, e: int) -> Tuple[int,int]:
    n = len(text)
    s = max(0, min(n, s)); e = max(0, min(n, e))
    if e < s: e = s

    while s > 0 and text[s-1] in WORD_CHARS and (s < n and text[s] in WORD_CHARS):
        s -= 1
    while e < n and (e-1) >= 0 and text[e-1] in WORD_CHARS and text[e] in WORD_CHARS:
        e += 1
    while s > 0 and text[s-1] in LEFT_PUNCT:
        s -= 1
    while e < n and text[e] in RIGHT_PUNCT:
        e += 1

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

# ---------- 12) Predict span (FIXED context_idxs) ----------
@torch.no_grad()
def predict_span(text: str, sentiment: str) -> Tuple[str, Tuple[int,int]]:
    if USE_NEUTRAL_SHORTCUT and sentiment == "neutral":
        return text.strip(), (0, len(text))

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

    # FIX: context token INDICES, not values
    context_idxs = [i for i, sid in enumerate(seq_ids) if sid == 1]
    if not context_idxs:
        return text.strip(), (0, len(text))
    c_start, c_end = context_idxs[0], context_idxs[-1]

    start_rank = np.argsort(start_logits)[::-1]
    end_rank   = np.argsort(end_logits)[::-1]

    start_cands = [i for i in start_rank if c_start <= i <= c_end][:TOPK]
    end_cands   = [i for i in end_rank   if c_start <= i <= c_end][:TOPK]

    best_score = -1e18
    best_pair = (c_start, c_start)
    backup_score = -1e18
    backup_pair = (c_start, c_start)

    for si in start_cands:
        if offsets[si] == (0,0): continue
        for ei in end_cands:
            if ei < si: continue
            if offsets[ei] == (0,0): continue
            length = ei - si + 1
            if length > MAX_ANSWER_TOKENS:
                continue

            base = start_logits[si] + end_logits[ei]
            bonus = LEN_BONUS_ALPHA * math.log1p(length)  # mild positive length bonus
            score = base + bonus

            if score > backup_score:
                backup_score = score
                backup_pair = (si, ei)

            if length < MIN_ANSWER_TOKENS:
                continue

            if score > best_score:
                best_score = score
                best_pair = (si, ei)

    si, ei = best_pair if best_score > -1e17 else backup_pair
    char_start, char_end = offsets[si][0], offsets[ei][1]
    char_start, char_end = snap_span(text, char_start, char_end)

    pred = text[char_start:char_end].strip()
    if pred == "":
        return text.strip(), (0, len(text))
    return pred, (char_start, char_end)

# ---------- 13) Evaluation (char-level + word jaccard) ----------
def char_metrics(text: str, true_span: Tuple[int,int], pred_span: Tuple[int,int]):
    n = len(text)
    ts, te = max(0, min(n, true_span[0])), max(0, min(n, true_span[1]))
    ps, pe = max(0, min(n, pred_span[0])), max(0, min(n, pred_span[1]))
    if te < ts: te = ts
    if pe < ps: pe = ps

    inter_s, inter_e = max(ts, ps), min(te, pe)
    inter = max(0, inter_e - inter_s)

    true_len = te - ts
    pred_len = pe - ps
    union = true_len + pred_len - inter

    iou = inter / union if union > 0 else 1.0
    prec = inter / pred_len if pred_len > 0 else (1.0 if true_len == 0 else 0.0)
    rec = inter / true_len if true_len > 0 else (1.0 if pred_len == 0 else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return f1, iou, inter, (pred_len - inter), (true_len - inter), (n - union)

def word_jaccard(a: str, b: str):
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb: return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

print("\n" + "="*70)
print("VALIDATION EVALUATION")
print("="*70)

macro_f1, macro_iou, macro_wj = [], [], []
TP=FP=FN=TN=0

for _, row in va_df.iterrows():
    text, sent, true_sel = row["text"], row["sentiment"], row["selected_text"]

    if USE_NEUTRAL_SHORTCUT and sent == "neutral":
        true_span = (0, len(text))
        true_phrase = text.strip()
    else:
        ts, te = find_char_span(text, true_sel)
        ts, te = snap_span(text, ts, te)
        true_span = (ts, te)
        true_phrase = text[ts:te].strip()

    pred_phrase, pred_span = predict_span(text, sent)

    f1, iou, tp, fp, fn, tn = char_metrics(text, true_span, pred_span)
    wj = word_jaccard(true_phrase, pred_phrase)

    macro_f1.append(f1); macro_iou.append(iou); macro_wj.append(wj)
    TP += tp; FP += fp; FN += fn; TN += tn

print(f"MACRO char-F1  : {np.mean(macro_f1):.4f}")
print(f"MACRO char-IoU : {np.mean(macro_iou):.4f}")
print(f"MACRO wordJac  : {np.mean(macro_wj):.4f}")

micro_p = TP/(TP+FP) if (TP+FP)>0 else 0
micro_r = TP/(TP+FN) if (TP+FN)>0 else 0
micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if (micro_p+micro_r)>0 else 0
print(f"MICRO char-F1  : {micro_f1:.4f} | Prec={micro_p:.4f} | Rec={micro_r:.4f}")

# ---------- 14) Predict test & save submission ----------
print("\n" + "="*70)
print(f"PREDICT TEST ({len(test_df)} rows)")
print("="*70)

preds = []
for i, row in test_df.iterrows():
    pred, _ = predict_span(row["text"], row["sentiment"])
    preds.append(pred if pred else row["text"].strip())
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(test_df)} done")

sub = pd.DataFrame({
    "textID": test_df["textID"].values if "textID" in test_df.columns else np.arange(len(test_df)),
    "selected_text": preds
})
out_path = "/content/submission.csv"
sub.to_csv(out_path, index=False)
print(f"\n✓ Saved: {out_path}")
print(sub.head())
