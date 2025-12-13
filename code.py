# =========================
# ONE-CELL FAST FINAL: arXiv Title Generation (T5) - Time-Optimized but Grade-Friendly
# - Encoder-decoder: T5 seq2seq
# - Fine-tune pre-trained T5
# - Optimization: fp16 (GPU), gradient accumulation, dynamic padding
# - Speed tricks: NO generation during training eval (eval_loss only), smaller max_source_len,
#                 optional disable grad checkpointing, train subset + 1 epoch
# - Final: generate on test -> ROUGE/BLEU + qualitative samples
# =========================

!pip -q install -U transformers datasets evaluate rouge_score sacrebleu accelerate sentencepiece

import os, sys, csv, time, random, inspect
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# -------------------------
# 1) CONFIG (EDIT THESE)
# -------------------------
DATA_PATH  = "/content/drive/MyDrive/ATI/arXiv_scientific dataset.csv"   # <<< EDIT
CLEAN_PATH = "/content/arxiv_clean.csv"

MODEL_NAME = "t5-small"
OUTPUT_DIR = "/content/title-gen-model-fast"

SEED = 42
USE_CATEGORY = True

# Lengths (speed-friendly)
MAX_SOURCE_LEN = 320      # 256-320 are good speed/quality tradeoffs
MAX_TARGET_LEN = 32
GEN_MAX_LEN    = 32

# Training budget (key for time)
MAX_TRAIN_SAMPLES = 50000   # reduce steps strongly; set None for full 108,990
EPOCHS = 1                  # 1 epoch often good; set 2 if you have time
# Alternative budget control (optional): uncomment to cap steps instead of epochs
# MAX_STEPS = 2000

LR = 3e-4
WARMUP_RATIO = 0.06

TRAIN_BATCH = 8            # try 8; if OOM -> 4 or 2
EVAL_BATCH  = 16
GRAD_ACCUM  = 4            # effective batch = TRAIN_BATCH * GRAD_ACCUM

# Decoding for final evaluation
FINAL_EVAL_BEAMS = 4
NO_REPEAT_NGRAM  = 3

# Final evaluation size for report
FINAL_TEST_SAMPLES = 8000   # use 8000; can reduce to 2000 if still too slow

# Optimization toggle (speed vs memory)
USE_GRAD_CHECKPOINTING = False  # set True only if you OOM

# -------------------------
# 2) Setup
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

if DATA_PATH.startswith("/content/drive") and not os.path.exists(DATA_PATH):
    from google.colab import drive
    drive.mount("/content/drive")

assert os.path.exists(DATA_PATH), f"File not found: {DATA_PATH}"
print("GPU available:", torch.cuda.is_available())
use_fp16 = torch.cuda.is_available()

csv.field_size_limit(sys.maxsize)

def scan_inquote_state(line: str, in_quote: bool) -> bool:
    i, n = 0, len(line)
    while i < n:
        if line[i] == '"':
            if in_quote and i + 1 < n and line[i + 1] == '"':
                i += 2
                continue
            in_quote = not in_quote
        i += 1
    return in_quote

def robust_clean_csv(raw_path: str, clean_path: str, max_record_lines=120, max_record_chars=2_000_000):
    """Handles multiline fields; drops corrupted records if needed."""
    t0 = time.time()
    good, bad = 0, 0
    with open(raw_path, "r", encoding="utf-8", errors="replace", newline="") as fin:
        header_buf = ""
        in_q = False
        while True:
            line = fin.readline()
            if not line:
                raise ValueError("Empty file or cannot read header.")
            header_buf += line
            in_q = scan_inquote_state(line, in_q)
            if not in_q:
                break
        header = next(csv.reader([header_buf]))
        ncols = len(header)

        with open(clean_path, "w", encoding="utf-8", newline="") as fout:
            writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(header)

            buf = ""
            in_quote = False
            buf_lines = 0
            for line in fin:
                buf += line
                buf_lines += 1
                in_quote = scan_inquote_state(line, in_quote)

                if not in_quote:
                    try:
                        row = next(csv.reader([buf]))
                        if len(row) == ncols:
                            writer.writerow(row); good += 1
                        else:
                            bad += 1
                    except Exception:
                        bad += 1
                    buf = ""; buf_lines = 0
                else:
                    if buf_lines >= max_record_lines or len(buf) >= max_record_chars:
                        # force-close attempt then drop if still bad
                        try_buf = buf.rstrip("\n") + '"' + "\n"
                        try:
                            row = next(csv.reader([try_buf]))
                            if len(row) == ncols:
                                writer.writerow(row); good += 1
                            else:
                                bad += 1
                        except Exception:
                            bad += 1
                        buf = ""; buf_lines = 0; in_quote = False

            if buf.strip():
                try:
                    if in_quote:
                        buf = buf.rstrip("\n") + '"' + "\n"
                    row = next(csv.reader([buf]))
                    if len(row) == ncols:
                        writer.writerow(row); good += 1
                    else:
                        bad += 1
                except Exception:
                    bad += 1

    print(f"[CLEAN] good rows: {good:,} | bad/dropped: {bad:,} | saved -> {clean_path} | time: {time.time()-t0:.1f}s")

# -------------------------
# 3) Clean -> Load
# -------------------------
print("Cleaning CSV...")
robust_clean_csv(DATA_PATH, CLEAN_PATH)

raw = load_dataset("csv", data_files=CLEAN_PATH)
ds = raw["train"]
print("[LOAD] rows:", ds.num_rows, "| columns:", ds.column_names)

# -------------------------
# 4) Detect columns + clean rows
# -------------------------
cols = set(ds.column_names)
def pick_col(cands):
    for c in cands:
        if c in cols: return c
    return None

TEXT_COL  = pick_col(["summary", "abstract", "paper_abstract", "description"])
TITLE_COL = pick_col(["title", "paper_title"])
CAT_COL   = pick_col(["category_code", "primary_category", "category"])

if TEXT_COL is None or TITLE_COL is None:
    raise ValueError(f"Cannot find summary/title columns. Found: {ds.column_names}")

print("TEXT_COL:", TEXT_COL, "| TITLE_COL:", TITLE_COL, "| CAT_COL:", CAT_COL)

def basic_clean(ex):
    s = ex.get(TEXT_COL, None)
    t = ex.get(TITLE_COL, None)
    if s is None or t is None: return {"_keep": False}
    s = str(s).strip(); t = str(t).strip()
    return {"_keep": bool(s) and bool(t)}

ds2 = ds.map(basic_clean)
ds2 = ds2.filter(lambda x: x["_keep"])
ds2 = ds2.remove_columns(["_keep"])
print("[CLEAN ROWS] rows:", ds2.num_rows)

# primary category for conditioning (quality gain, low overhead)
COND_COL = None
if USE_CATEGORY and (CAT_COL is not None):
    def make_primary_cat(ex):
        rawc = ex.get(CAT_COL, None)
        if rawc is None: return {"cat_primary": "UNKNOWN"}
        rawc = str(rawc).strip()
        if rawc == "": return {"cat_primary": "UNKNOWN"}
        first = rawc.split()[0].split(",")[0].split(";")[0].strip()
        return {"cat_primary": first if first else "UNKNOWN"}
    ds2 = ds2.map(make_primary_cat)
    COND_COL = "cat_primary"
    print("[COND] Using:", COND_COL)

# -------------------------
# 5) Split 80/10/10 + subsample train/test for speed
# -------------------------
split1 = ds2.train_test_split(test_size=0.2, seed=SEED)
tv = split1["test"].train_test_split(test_size=0.5, seed=SEED)
train_ds, valid_ds, test_ds = split1["train"], tv["train"], tv["test"]
print("Train:", train_ds.num_rows, "Valid:", valid_ds.num_rows, "Test:", test_ds.num_rows)

if MAX_TRAIN_SAMPLES is not None and train_ds.num_rows > MAX_TRAIN_SAMPLES:
    train_ds = train_ds.shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))
print("[TRAIN SUBSET] Train:", train_ds.num_rows)

if FINAL_TEST_SAMPLES is not None and test_ds.num_rows > FINAL_TEST_SAMPLES:
    test_ds = test_ds.shuffle(seed=SEED).select(range(FINAL_TEST_SAMPLES))
print("[TEST SUBSET] Test:", test_ds.num_rows)

# -------------------------
# 6) Tokenize
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_input(summary, cat_primary=None):
    summary = str(summary).strip()
    if USE_CATEGORY and (COND_COL is not None) and (cat_primary is not None):
        return f"category: {cat_primary} | generate title: {summary}"
    return f"generate title: {summary}"

def preprocess_batch(batch):
    summaries = batch[TEXT_COL]
    titles = batch[TITLE_COL]
    cats = batch[COND_COL] if (USE_CATEGORY and (COND_COL is not None) and (COND_COL in batch)) else [None]*len(summaries)

    inputs = [build_input(s, c) for s, c in zip(summaries, cats)]
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LEN, truncation=True)

    labels = tokenizer(text_target=titles, max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tok = train_ds.map(preprocess_batch, batched=True, remove_columns=list(train_ds.column_names))
valid_tok = valid_ds.map(preprocess_batch, batched=True, remove_columns=list(valid_ds.column_names))
test_tok  = test_ds.map(preprocess_batch,  batched=True, remove_columns=list(test_ds.column_names))
print("[TOKENIZED] keys:", train_tok[0].keys())

# -------------------------
# 7) Model + speed/optimization
# -------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# generation defaults for final decode (we will override in final loop anyway)
model.generation_config.max_length = GEN_MAX_LEN
model.generation_config.no_repeat_ngram_size = NO_REPEAT_NGRAM

# Grad checkpointing: OFF by default for speed (enable only if OOM)
if USE_GRAD_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("[OPT] Gradient checkpointing: ON (memory saver, slower)")
else:
    print("[OPT] Gradient checkpointing: OFF (faster)")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    pad_to_multiple_of=8
)

# -------------------------
# 8) TrainingArguments: eval per epoch, but NO generation during training
#    -> fastest while still showing training/val loss curves
# -------------------------
sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
supports_eval_strategy = "eval_strategy" in sig.parameters
supports_evaluation_strategy = "evaluation_strategy" in sig.parameters

args_dict = dict(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    fp16=use_fp16,

    per_device_train_batch_size=TRAIN_BATCH,
    per_device_eval_batch_size=EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,

    # IMPORTANT: no generation during training eval
    predict_with_generate=False,

    # pick best by eval_loss (fast, stable)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
)

# eval per epoch (compat)
if supports_eval_strategy:
    args_dict["eval_strategy"] = "epoch"
elif supports_evaluation_strategy:
    args_dict["evaluation_strategy"] = "epoch"

# Optional: cap steps instead of epochs (uncomment if you want)
# if "max_steps" in sig.parameters:
#     args_dict["max_steps"] = MAX_STEPS

args = Seq2SeqTrainingArguments(**args_dict)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=valid_tok,      # eval_loss only, fast
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=None         # no ROUGE/BLEU during training (saves a lot of time)
)

# -------------------------
# 9) Train
# -------------------------
print("\n[TRAIN] starting...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("[SAVE] Model saved to:", OUTPUT_DIR)

# -------------------------
# 10) FINAL metrics (ROUGE/BLEU) on test via batched generation
# -------------------------
rouge = evaluate.load("rouge")
bleu  = evaluate.load("sacrebleu")

# Prepare dataloader for batched generation
test_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dl = DataLoader(test_tok, batch_size=EVAL_BATCH, shuffle=False, collate_fn=data_collator)

model.eval()
device = model.device

pred_texts = []
with torch.no_grad():
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=FINAL_EVAL_BEAMS,
            max_length=GEN_MAX_LEN,
            no_repeat_ngram_size=NO_REPEAT_NGRAM,
        )
        pred_texts.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

# References from original test_ds titles (same order)
ref_texts = [str(t).strip() for t in test_ds[TITLE_COL]]

pred_texts = [p.strip() for p in pred_texts]
ref_texts  = [r.strip() for r in ref_texts]

rouge_res = rouge.compute(predictions=pred_texts, references=ref_texts, use_stemmer=True)
bleu_res  = bleu.compute(predictions=pred_texts, references=[[r] for r in ref_texts])

final_metrics = {
    "rouge1": round(rouge_res["rouge1"], 4),
    "rouge2": round(rouge_res["rouge2"], 4),
    "rougeL": round(rouge_res["rougeL"], 4),
    "bleu": round(bleu_res["score"], 4),
}
print("\n[FINAL TEST METRICS]", final_metrics)

print("\n[QUALITATIVE] 10 samples:")
for i in range(min(10, len(pred_texts))):
    print("="*90)
    print("GT  :", ref_texts[i])
    print("PRED:", pred_texts[i])

print("\nDone.")
