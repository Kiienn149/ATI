# =========================
# REPORT & SLIDE PACK (Run after training)
# - Baseline vs Fine-tuned metrics (ROUGE/BLEU)
# - EDA quick plots
# - Demo inference
# =========================

import os, random, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from collections import Counter
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

# ====== SET THESE to match your run ======
CLEAN_PATH = "/content/arxiv_clean.csv"                # already created
TRAINED_DIR = "/content/title-gen-model-fast"          # your saved model dir
MODEL_NAME_BASELINE = "t5-small"                       # baseline model
SEED = 42

USE_CATEGORY = True
MAX_SOURCE_LEN = 320
MAX_TARGET_LEN = 32
GEN_MAX_LEN = 32
NO_REPEAT_NGRAM = 3
BEAMS = 4

FINAL_TEST_SAMPLES = 8000   # set 2000 if you want faster baseline eval

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ====== Load dataset from cleaned CSV ======
raw = load_dataset("csv", data_files=CLEAN_PATH)
ds = raw["train"]
print("Rows:", ds.num_rows)
print("Columns:", ds.column_names)

# Detect columns
cols = set(ds.column_names)
def pick_col(cands):
    for c in cands:
        if c in cols: return c
    return None

TEXT_COL  = pick_col(["summary", "abstract", "paper_abstract", "description"])
TITLE_COL = pick_col(["title", "paper_title"])
CAT_COL   = pick_col(["category_code", "primary_category", "category"])
print("TEXT_COL:", TEXT_COL, "| TITLE_COL:", TITLE_COL, "| CAT_COL:", CAT_COL)

# Basic clean
def basic_clean(ex):
    s = ex.get(TEXT_COL, None)
    t = ex.get(TITLE_COL, None)
    if s is None or t is None: return {"_keep": False}
    s = str(s).strip(); t = str(t).strip()
    return {"_keep": bool(s) and bool(t)}

ds2 = ds.map(basic_clean).filter(lambda x: x["_keep"]).remove_columns(["_keep"])

# Primary category for conditioning
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

# Split like training (random)
split1 = ds2.train_test_split(test_size=0.2, seed=SEED)
tv = split1["test"].train_test_split(test_size=0.5, seed=SEED)
test_ds = tv["test"]

# Subsample test
if FINAL_TEST_SAMPLES is not None and test_ds.num_rows > FINAL_TEST_SAMPLES:
    test_ds = test_ds.shuffle(seed=SEED).select(range(FINAL_TEST_SAMPLES))

print("Test subset size:", test_ds.num_rows)

# ====== EDA quick ======
print("\n=== EDA ===")
if "summary_word_count" in ds.column_names:
    wc = np.array(ds["summary_word_count"])
    print("summary_word_count quantiles:", np.quantile(wc, [0.5, 0.9, 0.95, 0.99]).round(2))

# Top categories
if CAT_COL is not None:
    cats = [str(x).split()[0] if x is not None else "UNKNOWN" for x in ds[CAT_COL]]
    top = Counter(cats).most_common(10)
    print("Top 10 categories:", top)

# Plot histogram of summary_word_count if available
if "summary_word_count" in ds.column_names:
    plt.figure()
    plt.hist(ds["summary_word_count"], bins=50)
    plt.title("Histogram of summary_word_count")
    plt.xlabel("Word count")
    plt.ylabel("Frequency")
    plt.show()

# ====== Tokenize test set ======
tokenizer = AutoTokenizer.from_pretrained(TRAINED_DIR)

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

test_tok = test_ds.map(preprocess_batch, batched=True, remove_columns=list(test_ds.column_names))

# Data collator + dataloader for fast batched generation
# We'll create collator after loading model (needs model for pad labels)
def generate_all(model, tok_ds, batch_size=16):
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8)
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dl = DataLoader(tok_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=BEAMS,
                max_length=GEN_MAX_LEN,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
            )
            preds.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return [p.strip() for p in preds]

# References
refs = [str(t).strip() for t in test_ds[TITLE_COL]]

rouge = evaluate.load("rouge")
bleu  = evaluate.load("sacrebleu")

def score(preds, refs):
    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    bleu_res  = bleu.compute(predictions=preds, references=[[r] for r in refs])
    return {
        "rouge1": float(rouge_res["rouge1"]),
        "rouge2": float(rouge_res["rouge2"]),
        "rougeL": float(rouge_res["rougeL"]),
        "bleu": float(bleu_res["score"]),
    }

# ====== (A) Fine-tuned model ======
print("\n=== Evaluate Fine-tuned ===")
ft_model = AutoModelForSeq2SeqLM.from_pretrained(TRAINED_DIR).to(device)
ft_preds = generate_all(ft_model, test_tok, batch_size=16)
ft_metrics = score(ft_preds, refs)
print("Fine-tuned metrics:", ft_metrics)

# ====== (B) Baseline model (no fine-tuning) ======
print("\n=== Evaluate Baseline (pretrained only) ===")
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BASELINE)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_BASELINE).to(device)

# Use same input formatting as before for fairness (category + prefix)
def preprocess_batch_base(batch):
    summaries = batch[TEXT_COL]
    cats = batch[COND_COL] if (USE_CATEGORY and (COND_COL is not None) and (COND_COL in batch)) else [None]*len(summaries)
    inputs = [f"category: {c} | generate title: {str(s).strip()}" if (USE_CATEGORY and c is not None)
              else f"generate title: {str(s).strip()}"
              for s, c in zip(summaries, cats)]
    return base_tokenizer(inputs, max_length=MAX_SOURCE_LEN, truncation=True)

test_tok_base = test_ds.map(preprocess_batch_base, batched=True, remove_columns=list(test_ds.column_names))

def generate_all_base(model, tok_ds, batch_size=16):
    collator = DataCollatorForSeq2Seq(tokenizer=base_tokenizer, model=model, pad_to_multiple_of=8)
    tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dl = DataLoader(tok_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=BEAMS,
                max_length=GEN_MAX_LEN,
                no_repeat_ngram_size=NO_REPEAT_NGRAM,
            )
            preds.extend(base_tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return [p.strip() for p in preds]

base_preds = generate_all_base(base_model, test_tok_base, batch_size=16)
base_metrics = score(base_preds, refs)
print("Baseline metrics:", base_metrics)

# ====== Comparison table ======
df = pd.DataFrame([
    {"model": "baseline_pretrained", **base_metrics},
    {"model": "fine_tuned", **ft_metrics},
])
df["delta_rouge1"] = df["rouge1"] - df.loc[0, "rouge1"]
df["delta_rouge2"] = df["rouge2"] - df.loc[0, "rouge2"]
df["delta_rougeL"] = df["rougeL"] - df.loc[0, "rougeL"]
df["delta_bleu"]   = df["bleu"]   - df.loc[0, "bleu"]

print("\n=== METRICS TABLE (copy into report) ===")
print(df.round(4))

out_csv = "/content/metrics_comparison.csv"
df.round(6).to_csv(out_csv, index=False)
print("Saved:", out_csv)

# ====== Qualitative demo (random samples) ======
print("\n=== DEMO (random 5 samples) ===")
idxs = random.sample(range(len(refs)), k=min(5, len(refs)))
for i in idxs:
    print("-"*90)
    print("Abstract (first 300 chars):", str(test_ds[TEXT_COL][i])[:300].replace("\n"," "))
    print("GT  :", refs[i])
    print("PRED:", ft_preds[i])

# ====== Interactive inference ======
print("\n=== INTERACTIVE DEMO ===")
print("Paste an abstract, then press Enter. Type 'q' to quit.\n")
while True:
    user_abs = input("Abstract> ")
    if user_abs.strip().lower() == "q":
        break
    user_cat = "cs.LG" if USE_CATEGORY else None
    inp = f"category: {user_cat} | generate title: {user_abs.strip()}" if (USE_CATEGORY and user_cat) else f"generate title: {user_abs.strip()}"
    inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN).to(device)
    with torch.no_grad():
        gen = ft_model.generate(**inputs, num_beams=BEAMS, max_length=GEN_MAX_LEN, no_repeat_ngram_size=NO_REPEAT_NGRAM)
    print("Generated title:", tokenizer.decode(gen[0], skip_special_tokens=True))
