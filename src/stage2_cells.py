# Stage 2 cells — kept as plain strings so the notebook generator and the local test use identical code.
CELLS = []

CELLS.append(("md", r"""
# Stage 2 — Cleaning, stratified sampling, and per-document measurements
**Paper:** *Evaluating Subword Tokenization Efficiency for the Uzbek Language in Multilingual Large Language Models*

Inputs come from Stage 1: the corpus metadata census and row-group map, saved to Google Drive. Everything is pinned: the dataset snapshot `955412d…`, the three tokenizer commits, and seed 42.

Stage 2 does four things:
1. **Stratified samples:**
   * **A:** script × format origin (OCR/Digital) × source → RQ1, RQ2 (observational), RQ5
   * **B:** domain × script → RQ4
   * **C:** register × script → RQ4. *Religious* (477 docs) and *Technical* (125 docs) are merged into **Other**.
2. **One documented cleaning pipeline,** with an exclusion-flow table for the paper.
3. **Per-document measurements for all three tokenizers:** TPW-all, **TPW-words (primary)**, fertility, PCW, CPT, and SE.
4. **A sanity check:** the 50 most frequent words and how each tokenizer splits them.

No hypothesis tests are run here. Statistical inference is Stage 6.
"""))

CELLS.append(("code", r"""
# 1 — setup, constants, seeds
!pip -q install -U "datasets>=3.0" "transformers>=4.45" tokenizers huggingface_hub pyarrow pandas numpy scipy statsmodels matplotlib

import os, sys, json, re, random, hashlib, platform, unicodedata, time, datetime, zlib, math
import numpy as np, pandas as pd
import importlib.metadata as im

SEED = 42
random.seed(SEED); np.random.seed(SEED)

REPO   = "Digitaluz/uzbek-language-corpus"
DS_SHA = "955412d35fca149ce5374daba0d01565c578a36d"          # pinned in Stage 1
MODELS = {  # name: (repo, pinned commit from Stage 1)
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B",       "8faed761d45a263340a0528343f099c05c9a4323"),
    "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    "Gemma-2-2B":   ("google/gemma-2-2b",       "c5ebcd40d208330abc697524c919956e692655cf"),
}

# sampling and cleaning parameters (all reported in the paper)
N_A, N_B, N_C   = 300, 150, 150      # target documents per stratum
OVERSAMPLE      = 1.25               # draw extra documents to replace exclusions
MIN_WORDS_FIELD = 50                 # pre-filter on the corpus word_count field
MIN_WORDS_CLEAN = 50                 # final filter on regex words after cleaning
MAX_CHARS       = 20_000             # length cap (cut at a word boundary)
MIN_LETTER_SHARE = 0.50              # letters / non-space characters
SCRIPT_THR      = 0.90               # character-based script agreement threshold

OUT = "/content/uz_tok/stage2"; os.makedirs(OUT, exist_ok=True)
PKGS = ["datasets", "transformers", "tokenizers", "huggingface_hub", "pyarrow", "pandas", "numpy", "scipy", "statsmodels", "matplotlib"]
params = dict(SEED=SEED, DS_SHA=DS_SHA, MODELS=MODELS, N_A=N_A, N_B=N_B, N_C=N_C, OVERSAMPLE=OVERSAMPLE,
              MIN_WORDS_FIELD=MIN_WORDS_FIELD, MIN_WORDS_CLEAN=MIN_WORDS_CLEAN, MAX_CHARS=MAX_CHARS,
              MIN_LETTER_SHARE=MIN_LETTER_SHARE, SCRIPT_THR=SCRIPT_THR,
              packages={p: im.version(p) for p in PKGS}, python=sys.version,
              timestamp_utc=datetime.datetime.utcnow().isoformat() + "Z")
json.dump(params, open(f"{OUT}/stage2_params.json", "w"), indent=2)
print(json.dumps(params["packages"], indent=2))
"""))

CELLS.append(("code", r"""
# 2 — Google Drive + Hugging Face login
from google.colab import drive
drive.mount("/content/drive")
S1_DRIVE = "/content/drive/MyDrive/uz_tokenization/stage1"
DRIVE_OUT = "/content/drive/MyDrive/uz_tokenization/stage2"; os.makedirs(DRIVE_OUT, exist_ok=True)

from huggingface_hub import login, HfApi, HfFileSystem
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN); fs = HfFileSystem(token=HF_TOKEN)
BASE = f"datasets/{REPO}@{DS_SHA}"
print("Logged in as:", api.whoami()["name"])
"""))

CELLS.append(("code", r"""
# 3 — load the Stage-1 census and row-group map
import pyarrow.parquet as pq
census = pd.read_parquet(f"{S1_DRIVE}/corpus_metadata.parquet")
rc = pd.read_json(f"{S1_DRIVE}/row_groups.json", orient="records")
census["word_count"] = census["word_count"].astype("int64")
census["gidx"] = np.arange(len(census), dtype=np.int64)          # global row index (file order, then row order)
assert len(census) == int(rc.rows.sum()) == 5_637_052, (len(census), int(rc.rows.sum()))
census["register_m"] = census["register"].astype(str).replace({"Religious": "Other", "Technical": "Other"})
for c in ["source", "domain", "register", "register_m", "script", "format_origin"]:
    census[c] = census[c].astype(str)
starts = np.concatenate([[0], np.cumsum(rc.rows.values)[:-1]]).astype(np.int64)
print(census.shape); census.head(3)
"""))

CELLS.append(("code", r"""
# 4 — stratified sampling plan (deterministic: seed 42 + a CRC32 hash of the stratum name)
elig = census[(census.word_count >= MIN_WORDS_FIELD) & (census.script.isin(["Latin", "Cyrillic"]))]
print(f"Eligible after pre-filter (Latin/Cyrillic, word_count ≥ {MIN_WORDS_FIELD}): {len(elig):,} of {len(census):,}")

def draw(df, keys, n_target, tag):
    out, plan = [], []
    for k, g in df.groupby(keys, observed=True, sort=True):
        k = k if isinstance(k, tuple) else (k,)
        name = " | ".join(map(str, k))
        n_draw = min(len(g), math.ceil(n_target * OVERSAMPLE))
        rng = np.random.default_rng([SEED, zlib.crc32(f"{tag}:{name}".encode())])
        pick = rng.choice(g.gidx.values, size=n_draw, replace=False)           # random order = priority rank
        out.append(pd.DataFrame({"gidx": pick, "sample": tag, "stratum": name, "rank": np.arange(n_draw)}))
        plan.append({"sample": tag, "stratum": name, "available": len(g), "drawn": n_draw, "target": n_target})
    return pd.concat(out, ignore_index=True), pd.DataFrame(plan)

A, planA = draw(elig, ["script", "format_origin", "source"], N_A, "A")
B, planB = draw(elig, ["domain", "script"], N_B, "B")
C, planC = draw(elig, ["register_m", "script"], N_C, "C")
membership = pd.concat([A, B, C], ignore_index=True)
plan = pd.concat([planA, planB, planC], ignore_index=True)
plan["target_reachable"] = plan.available >= plan.target
plan.to_csv(f"{OUT}/sampling_plan.csv", index=False)
membership.to_csv(f"{OUT}/sample_membership_drawn.csv", index=False)
need = np.sort(membership.gidx.unique())
print(f"Strata: A={len(planA)}, B={len(planB)}, C={len(planC)} | documents to fetch (unique): {len(need):,}")
print("Strata below target (all available documents are used):")
print(plan[~plan.target_reachable][["sample", "stratum", "available", "target"]].to_string(index=False))
"""))

CELLS.append(("code", r"""
# 5 — fetch the texts of the sampled documents (reads only the needed row groups; 15–40 min)
FETCH_CAP = 30_000          # raw characters kept per document (cleaning then caps at MAX_CHARS)
fi_of = np.searchsorted(starts, need, side="right") - 1
texts, ids = {}, {}
t0 = time.time()
for k, fi in enumerate(np.unique(fi_of)):
    f = rc.file[fi]
    local = need[fi_of == fi] - starts[fi]
    rg_rows = np.asarray(rc.rg_rows[fi]); rg_starts = np.concatenate([[0], np.cumsum(rg_rows)[:-1]])
    rg_of = np.searchsorted(rg_starts, local, side="right") - 1
    with fs.open(f"{BASE}/{f}") as fh:
        pfile = pq.ParquetFile(fh)
        for rg in np.unique(rg_of):
            tbl = pfile.read_row_group(int(rg), columns=["id", "text"])
            col_text, col_id = tbl.column("text"), tbl.column("id")
            for li in local[rg_of == rg]:
                j = int(li - rg_starts[rg]); g = int(starts[fi] + li)
                t = col_text[j].as_py() or ""
                texts[g] = t[:FETCH_CAP]; ids[g] = col_id[j].as_py()
    if (k + 1) % 10 == 0: print(f"{k+1}/{len(np.unique(fi_of))} files, {len(texts):,} docs, {time.time()-t0:.0f}s")
assert len(texts) == len(need)
raw = census.set_index("gidx").loc[need].reset_index()
raw["id"] = raw.gidx.map(ids); raw["text_raw"] = raw.gidx.map(texts)
raw.to_parquet(f"{OUT}/sample_raw.parquet")
print(f"Fetched {len(raw):,} documents in {time.time()-t0:.0f}s")
"""))

CELLS.append(("code", r"""
# 6 — cleaning pipeline and exclusion flow
APOS_CLASS = "['‘’ʻʼ`´]"
WORD_RE = re.compile(rf"[^\W\d_]+(?:{APOS_CLASS}[^\W\d_]+)*", re.UNICODE)
PLACEHOLDER_RE = re.compile(r"\[(?:PHONE|NAME|EMAIL|ADDRESS|ID|URL|CARD|PASSPORT|[A-Z_]{2,20})\]")

def cap(t, n=MAX_CHARS):
    if len(t) <= n: return t, False
    cut = t.rfind(" ", 0, n); return t[: cut if cut > 0 else n], True

def clean(raw_text):
    t = PLACEHOLDER_RE.sub(" ", raw_text or "")
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return cap(t)

def script_shares(t):
    lat = cyr = oth = 0
    for ch in t:
        if ch.isalpha() and ch not in "ʻʼ":
            n = unicodedata.name(ch, "")
            if n.startswith("LATIN"): lat += 1
            elif n.startswith("CYRILLIC"): cyr += 1
            else: oth += 1
    tot = lat + cyr + oth
    return (lat / tot, cyr / tot) if tot else (np.nan, np.nan)

rows = []
for r in raw.itertuples():
    t, trunc = clean(r.text_raw)
    words = WORD_RE.findall(t)
    nonspace = sum(1 for ch in t if not ch.isspace())
    letters = sum(len(w) for w in words)
    lat, cyr = script_shares(t)
    det = "NoLetters" if np.isnan(lat) else ("Latin" if lat >= SCRIPT_THR else "Cyrillic" if cyr >= SCRIPT_THR else "Mixed")
    if not t: reason = "empty_after_cleaning"
    elif len(words) < MIN_WORDS_CLEAN: reason = "too_short"
    elif nonspace == 0 or letters / nonspace < MIN_LETTER_SHARE: reason = "low_letter_share"
    elif det != r.script: reason = "script_mismatch"
    else: reason = "ok"
    rows.append(dict(gidx=r.gidx, text=t, truncated=trunc, n_words=len(words), letter_share=(letters / nonspace) if nonspace else np.nan,
                     latin_share=lat, cyrillic_share=cyr, script_detected=det, exclusion=reason))
cl = raw.drop(columns=["text_raw"]).merge(pd.DataFrame(rows), on="gidx")

# keep the first n_target eligible documents per stratum, in random-rank order
mem = membership.merge(cl[["gidx", "exclusion"]], on="gidx")
mem = mem.sort_values(["sample", "stratum", "rank"])
mem["eligible"] = mem.exclusion == "ok"
mem["elig_rank"] = mem.groupby(["sample", "stratum"]).eligible.cumsum()
tgt = {"A": N_A, "B": N_B, "C": N_C}
mem["kept"] = mem.eligible & (mem.elig_rank <= mem["sample"].map(tgt))
mem.to_csv(f"{OUT}/sample_membership_final.csv", index=False)

flow = (mem.groupby("sample").agg(drawn=("gidx", "size"),
                                  empty_after_cleaning=("exclusion", lambda s: (s == "empty_after_cleaning").sum()),
                                  too_short=("exclusion", lambda s: (s == "too_short").sum()),
                                  low_letter_share=("exclusion", lambda s: (s == "low_letter_share").sum()),
                                  script_mismatch=("exclusion", lambda s: (s == "script_mismatch").sum()),
                                  eligible=("eligible", "sum"), kept=("kept", "sum")))
flow.to_csv(f"{OUT}/exclusion_flow.csv"); print(flow)
achieved = mem[mem.kept].groupby(["sample", "stratum"]).size().rename("kept").reset_index()
achieved = plan.merge(achieved, on=["sample", "stratum"], how="left").fillna({"kept": 0})
achieved.to_csv(f"{OUT}/strata_achieved.csv", index=False)
docs = cl[cl.gidx.isin(mem[mem.kept].gidx)].copy()
docs.to_parquet(f"{OUT}/sample_clean.parquet")
print(f"\nUnique documents kept for measurement: {len(docs):,}")
"""))

CELLS.append(("code", r"""
# 7 — load the tokenizers at the pinned commits
from transformers import AutoTokenizer
TOK = {}
for name, (repo, sha) in MODELS.items():
    TOK[name] = AutoTokenizer.from_pretrained(repo, revision=sha, token=HF_TOKEN, use_fast=True)
    print(f"{name}: {type(TOK[name]).__name__}, vocab={TOK[name].vocab_size:,}, len={len(TOK[name]):,}")
"""))

CELLS.append(("code", r"""
# 8 — per-document measurements
# TPW-all   = tokens(cleaned text) / words           (includes punctuation, digits, whitespace tokens)
# TPW-words = tokens(words joined by single spaces) / words   ← PRIMARY efficiency metric
# fertility = mean tokens per word, each word tokenized in isolation with a leading space; PCW = share of words with ≥2 tokens
# CPT       = characters / tokens (word-only text);  SE = tokens / min over the three tokenizers (word-only text)
BATCH = 256
docs = docs.reset_index(drop=True)
word_lists = [WORD_RE.findall(t) for t in docs.text]
word_texts = [" ".join(ws) for ws in word_lists]
docs["n_chars_words"] = [len(x) for x in word_texts]
docs["digit_ratio"] = [sum(ch.isdigit() for ch in t) / max(1, len(t)) for t in docs.text]
docs["apos_02BB"] = docs.text.str.count("ʻ"); docs["apos_02BC"] = docs.text.str.count("ʼ"); docs["apos_ascii"] = docs.text.str.count("'")

def count_tokens(tok, seqs):
    out = []
    for i in range(0, len(seqs), BATCH):
        out += [len(x) for x in tok(seqs[i:i + BATCH], add_special_tokens=False)["input_ids"]]
    return np.array(out)

types = sorted({w for ws in word_lists for w in ws})
print(f"Documents: {len(docs):,} | word types: {len(types):,}")
iso = {}
t0 = time.time()
for name, tok in TOK.items():
    docs[f"tokall_{name}"] = count_tokens(tok, list(docs.text))
    docs[f"tokw_{name}"] = count_tokens(tok, word_texts)
    lens = count_tokens(tok, [" " + w for w in types])
    iso[name] = dict(zip(types, lens))
    docs[f"fert_{name}"] = [np.mean([iso[name][w] for w in ws]) for ws in word_lists]
    docs[f"pcw_{name}"] = [np.mean([iso[name][w] >= 2 for w in ws]) for ws in word_lists]
    print(f"{name} done ({time.time()-t0:.0f}s)")

names = list(TOK)
docs["tokw_min"] = docs[[f"tokw_{n}" for n in names]].min(axis=1)
for n in names:
    docs[f"TPWall_{n}"] = docs[f"tokall_{n}"] / docs.n_words
    docs[f"TPWw_{n}"]   = docs[f"tokw_{n}"] / docs.n_words
    docs[f"CPT_{n}"]    = docs.n_chars_words / docs[f"tokw_{n}"]
    docs[f"SE_{n}"]     = docs[f"tokw_{n}"] / docs.tokw_min
metric_cols = [c for c in docs if re.match(r"(tokall|tokw|fert|pcw|TPWall|TPWw|CPT|SE)_", c)]
docs.drop(columns=["text"]).to_csv(f"{OUT}/doc_metrics.csv", index=False)
docs.drop(columns=["text"]).to_parquet(f"{OUT}/doc_metrics.parquet")
wt = pd.DataFrame({"word": types, **{f"ntok_{n}": [iso[n][w] for w in types] for n in names}})
wt.to_parquet(f"{OUT}/word_types_isolated.parquet")
print("Non-finite metric values:", int((~np.isfinite(docs[metric_cols].to_numpy(dtype=float))).sum()))
"""))

CELLS.append(("code", r"""
# 9 — sanity check: the 50 most frequent words and how each tokenizer splits them
from collections import Counter
freq = Counter(w for ws in word_lists for w in ws)
top = pd.DataFrame(freq.most_common(50), columns=["word", "freq"])
for n, tok in TOK.items():
    enc = tok([" " + w for w in top.word], add_special_tokens=False)["input_ids"]
    top[f"ntok_{n}"] = [len(e) for e in enc]
    top[f"pieces_{n}"] = [" | ".join(tok.decode([i]) for i in e) for e in enc]
top.to_csv(f"{OUT}/top50_words.csv", index=False)
pd.set_option("display.max_colwidth", 60)
print("Share of the 50 most frequent words split into ≥2 tokens:")
print({n: round(float((top[f'ntok_{n}'] >= 2).mean()), 2) for n in TOK})
top[["word", "freq"] + [f"ntok_{n}" for n in TOK] + [f"pieces_{n}" for n in TOK]].head(50)
"""))

CELLS.append(("code", r"""
# 10 — descriptive preview (macro = mean of per-document values; NOT final results)
kept = mem[mem.kept][["gidx", "sample", "stratum"]].merge(docs, on="gidx")
def macro(df, by):
    cols = {f"TPWw_{n}": "mean" for n in names}
    return df.groupby(by, observed=True).agg(n=("gidx", "size"), **{f"TPWw_{n}": (f"TPWw_{n}", "mean") for n in names}).round(3)
prevA = macro(kept[kept["sample"] == "A"], ["script", "format_origin"])
print("Sample A — mean TPW-words by script × format origin:\n", prevA)
prevA.to_csv(f"{OUT}/preview_A_script_format.csv")
print("\nAll kept docs — TPW-all vs TPW-words (macro mean):")
print(pd.DataFrame({n: {"TPW_all": kept[f"TPWall_{n}"].mean(), "TPW_words": kept[f"TPWw_{n}"].mean(),
                        "fertility": kept[f"fert_{n}"].mean(), "PCW_%": 100 * kept[f"pcw_{n}"].mean(),
                        "SE": kept[f"SE_{n}"].mean()} for n in names}).T.round(3))
"""))

CELLS.append(("code", r"""
# 11 — Stage-2 gate and copy to Drive
fA = achieved[achieved["sample"] == "A"]
gate = {
  "documents_measured": int(len(docs)),
  "strata_total": int(len(achieved)),
  "strata_at_target_%": round(100 * float((achieved.kept >= achieved.target).mean()), 1),
  "strata_with_<30_docs": achieved[achieved.kept < 30][["sample", "stratum", "kept"]].values.tolist(),
  "exclusion_rate_%": round(100 * float(1 - mem.eligible.mean()), 2),
  "no_nonfinite_metrics": bool(np.isfinite(docs[metric_cols].to_numpy(dtype=float)).all()),
  "pinned_dataset_sha": DS_SHA,
}
json.dump(gate, open(f"{OUT}/stage2_gate.json", "w"), indent=2, default=str)
print(json.dumps(gate, indent=2, default=str))
import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True); print("Copied to", DRIVE_OUT)
"""))
