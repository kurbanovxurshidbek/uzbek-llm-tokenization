#!/usr/bin/env python3
"""Generates Stage1_Dataset_Inspection_and_Pilot.ipynb (Google Colab)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Stage 1 — Dataset inspection, validation, and 1,000-document pilot
**Paper:** *Evaluating Subword Tokenization Efficiency for the Uzbek Language in Multilingual Large Language Models*

This notebook does three things and nothing more:

1. **Verifies the corpus characteristics** (documents, words, sources, domains, registers, scripts, OCR share) from the actual Parquet files, so that no number in the paper is copied from the dataset card without verification.
2. **Draws a reproducible, uniformly random 1,000-document pilot sample** and runs data-quality checks on it (script labels, apostrophe variants, OCR noise, placeholders, duplicates, word-count agreement).
3. **Runs a descriptive pilot of the three tokenizers** (Qwen2.5-1.5B, Llama-3.2-1B, Gemma-2-2B) to validate the measurement pipeline.

> ⚠️ Pilot numbers are for **pipeline validation only**. They must not be reported as findings: the sample is small and is not stratified by script, domain, or source.

**Runtime:** CPU is enough. Outputs go to `/content/uz_tok/stage1/` (optionally mirrored to Google Drive).
""")

md("## 0. Environment, seeds, and version tracking")
code(r"""
!pip -q install -U "datasets>=3.0" "transformers>=4.45" tokenizers huggingface_hub pyarrow pandas numpy scipy statsmodels matplotlib

import os, sys, json, re, random, hashlib, platform, unicodedata, time, datetime
import numpy as np, pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)

OUT = "/content/uz_tok/stage1"
os.makedirs(OUT, exist_ok=True)

import importlib.metadata as im
PKGS = ["datasets", "transformers", "tokenizers", "huggingface_hub", "pyarrow", "pandas", "numpy", "scipy", "statsmodels", "matplotlib"]
env = {"python": sys.version, "platform": platform.platform(), "seed": SEED,
       "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "packages": {p: im.version(p) for p in PKGS}}
json.dump(env, open(f"{OUT}/environment.json", "w"), indent=2)
env
""")

md(r"""
### Optional: mirror outputs to Google Drive
Colab deletes `/content` when the session ends. Run this cell to keep the results.
""")
code(r"""
USE_DRIVE = True   # set False to skip
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_OUT = "/content/drive/MyDrive/uz_tokenization/stage1"
    os.makedirs(DRIVE_OUT, exist_ok=True)
""")

md(r"""
## 1. Hugging Face authentication
`meta-llama/Llama-3.2-1B` and `google/gemma-2-2b` are **gated**. Before running:
1. Open both model pages on huggingface.co while logged in and accept the licence terms.
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. In Colab: 🔑 *Secrets* → add `HF_TOKEN` and enable notebook access. Alternatively, paste the token when prompted.
""")
code(r"""
from huggingface_hub import login, HfApi, HfFileSystem
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    import getpass
    HF_TOKEN = getpass.getpass("HF token: ")
login(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)
fs = HfFileSystem(token=HF_TOKEN)
print("Logged in as:", api.whoami()["name"])
""")

md(r"""
## 2. Pin the dataset revision and list its files
The commit SHA is recorded so that every later stage reads **exactly the same snapshot**.
""")
code(r"""
REPO = "Digitaluz/uzbek-language-corpus"
info = api.dataset_info(REPO, files_metadata=True)
DS_SHA = info.sha
parquet = sorted([s for s in info.siblings if s.rfilename.endswith(".parquet")], key=lambda s: s.rfilename)
files_df = pd.DataFrame([{"file": s.rfilename, "bytes": s.size} for s in parquet])
ds_meta = {"repo": REPO, "revision_sha": DS_SHA, "last_modified": str(info.last_modified),
           "gated": getattr(info, "gated", None), "n_parquet_files": len(files_df),
           "total_parquet_bytes": int(files_df.bytes.sum()),
           "total_parquet_GB_decimal": round(files_df.bytes.sum() / 1e9, 3),
           "total_parquet_GiB": round(files_df.bytes.sum() / 2**30, 3),
           "card_license": (info.card_data or {}).get("license") if info.card_data else None}
json.dump(ds_meta, open(f"{OUT}/dataset_revision.json", "w"), indent=2)
files_df.to_csv(f"{OUT}/parquet_files.csv", index=False)
print(json.dumps(ds_meta, indent=2)); files_df.head()
""")

md("## 3. Schema check (read from the first Parquet footer only)")
code(r"""
import pyarrow.parquet as pq
BASE = f"datasets/{REPO}@{DS_SHA}"
pf0 = pq.ParquetFile(fs.open(f"{BASE}/{files_df.file[0]}"))
schema = pf0.schema_arrow
print(schema)
EXPECTED = ["id", "text", "source", "domain", "topics", "register", "script", "format_origin", "publication_year", "word_count", "processed_at"]
missing = [c for c in EXPECTED if c not in schema.names]
extra = [c for c in schema.names if c not in EXPECTED]
print("Missing vs card:", missing, "| Extra vs card:", extra)
""")

md(r"""
## 4. Full-corpus metadata census (no text is downloaded)
Only the small metadata columns are read from **all** files. This verifies the corpus figures (documents, words, and the distributions of source, domain, register, script, and OCR vs digital origin) directly from the data. It takes about 5–20 minutes, depending on Colab's network speed.
""")
code(r"""
META_COLS = [c for c in ["source", "domain", "register", "script", "format_origin", "publication_year", "word_count"] if c in schema.names]
parts, row_counts = [], []
t0 = time.time()
for i, f in enumerate(files_df.file):
    with fs.open(f"{BASE}/{f}") as fh:
        pfile = pq.ParquetFile(fh)
        row_counts.append({"file": f, "rows": pfile.metadata.num_rows, "row_groups": pfile.metadata.num_row_groups,
                           "rg_rows": [pfile.metadata.row_group(k).num_rows for k in range(pfile.metadata.num_row_groups)]})
        t = pfile.read(columns=META_COLS).to_pandas()
    t["file"] = f
    for c in ["source", "domain", "register", "script", "format_origin"]:
        if c in t: t[c] = t[c].astype("category")
    parts.append(t)
    if (i + 1) % 10 == 0: print(f"{i+1}/{len(files_df)} files, {time.time()-t0:.0f}s")
census = pd.concat(parts, ignore_index=True)
rc = pd.DataFrame(row_counts)
rc.to_json(f"{OUT}/row_groups.json", orient="records")
census.drop(columns=["file"]).to_parquet(f"{OUT}/corpus_metadata.parquet")
print("Done in", round(time.time() - t0), "s")
""")

code(r"""
def dist(col, weight=None):
    g = census.groupby(col, observed=True)
    d = pd.DataFrame({"documents": g.size()})
    d["doc_share_%"] = 100 * d.documents / d.documents.sum()
    if "word_count" in census:
        d["words"] = g["word_count"].sum()
        d["word_share_%"] = 100 * d.words / d.words.sum()
    return d.sort_values("documents", ascending=False).round(2)

summary = {
    "documents": int(len(census)),
    "words_sum_word_count_field": int(census.word_count.sum()) if "word_count" in census else None,
    "n_sources": int(census.source.nunique()), "n_domains": int(census.domain.nunique()),
    "n_registers": int(census.register.nunique()), "n_scripts": int(census.script.nunique()),
    "null_counts": {c: int(census[c].isna().sum()) for c in META_COLS},
    "word_count_quantiles": census.word_count.quantile([.01, .1, .25, .5, .75, .9, .99, .999]).round(0).to_dict(),
}
tables = {c: dist(c) for c in ["source", "domain", "register", "script", "format_origin"] if c in census}
for c, d in tables.items():
    d.to_csv(f"{OUT}/census_{c}.csv")
    print(f"\n=== {c} ===\n", d)
json.dump(summary, open(f"{OUT}/census_summary.json", "w"), indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
""")

md(r"""
### 4.1 Card vs data verification
The values below are **reported by the dataset card** (5,637,052 documents; 6,062,245,161 words; 16.2 GB; 22 domains; 9 registers; 3 script labels). The table compares them with the census. Only verified values may appear in the paper.
""")
code(r"""
card = {"documents": 5_637_052, "words": 6_062_245_161, "n_domains": 22, "n_registers": 9, "n_scripts": 3}
obs = {"documents": summary["documents"], "words": summary["words_sum_word_count_field"],
       "n_domains": summary["n_domains"], "n_registers": summary["n_registers"], "n_scripts": summary["n_scripts"]}
ver = pd.DataFrame({"card": card, "observed": obs})
ver["match"] = ver.card == ver.observed
ver.to_csv(f"{OUT}/card_vs_data.csv"); ver
""")

md(r"""
## 5. Reproducible uniform 1,000-document pilot sample
Document indices are drawn uniformly at random (seed 42) over the whole corpus, and only the Parquet row groups that contain them are downloaded. A uniform sample is dominated by short web and Telegram documents. That is acceptable for a pilot; the main study will use stratified sampling.
""")
code(r"""
N_PILOT = 1000
rng = np.random.default_rng(SEED)
total = int(rc.rows.sum())
gidx = np.sort(rng.choice(total, size=N_PILOT, replace=False))
starts = np.concatenate([[0], np.cumsum(rc.rows.values)[:-1]])
rows = []
for fi, f in enumerate(rc.file):
    lo, hi = starts[fi], starts[fi] + rc.rows[fi]
    local = gidx[(gidx >= lo) & (gidx < hi)] - lo
    if len(local) == 0: continue
    rg_starts = np.concatenate([[0], np.cumsum(rc.rg_rows[fi])[:-1]])
    with fs.open(f"{BASE}/{f}") as fh:
        pfile = pq.ParquetFile(fh)
        for rg in np.unique(np.searchsorted(rg_starts, local, side="right") - 1):
            tbl = pfile.read_row_group(int(rg)).to_pandas()
            sel = local[(local >= rg_starts[rg]) & (local < rg_starts[rg] + rc.rg_rows[fi][rg])] - rg_starts[rg]
            sub = tbl.iloc[sel].copy(); sub["file"] = f; sub["row_in_file"] = sel + rg_starts[rg]
            rows.append(sub)
pilot = pd.concat(rows, ignore_index=True)
assert len(pilot) == N_PILOT, len(pilot)
pilot.to_parquet(f"{OUT}/pilot_1000.parquet")
pilot[["id", "file", "row_in_file"]].to_csv(f"{OUT}/pilot_1000_ids.csv", index=False)
print(pilot.shape); pilot[["source", "domain", "register", "script", "format_origin", "word_count"]].head()
""")

md(r"""
## 6. Data cleaning and validation checks (pilot)
| Check | Why it matters for tokenization |
|---|---|
| Recomputed word count vs `word_count` | Tokens-per-word needs a fixed, documented definition of a *word* |
| Character-based script detection vs `script` label | Script labels were assigned automatically and may be wrong; this is needed for RQ2 |
| Apostrophe variants in *o‘, g‘* and the tutuq belgisi (ʼ) | Uzbek Latin uses several Unicode characters for the same letter; each tokenizes differently |
| OCR-noise proxies | OCR artefacts inflate fragmentation |
| PII placeholders (`[PHONE]`, `[NAME]` …) | Not natural Uzbek text; must be counted and handled consistently |
| Duplicates (id and text hash) | The card reports exact deduplication; verify it |
""")
code(r"""
APOS = {"‘": "LEFT SINGLE QUOTE ‘", "’": "RIGHT SINGLE QUOTE ’", "ʻ": "MODIFIER TURNED COMMA ʻ",
        "ʼ": "MODIFIER APOSTROPHE ʼ", "'": "ASCII APOSTROPHE '", "`": "GRAVE `", "´": "ACUTE ´"}
APOS_CLASS = "['‘’ʻʼ`´]"
WORD_RE = re.compile(rf"[^\W\d_]+(?:{APOS_CLASS}[^\W\d_]+)*", re.UNICODE)   # letters, with word-internal apostrophes
PLACEHOLDER_RE = re.compile(r"\[(?:PHONE|NAME|EMAIL|ADDRESS|ID|URL|CARD|PASSPORT|[A-Z_]{2,20})\]")

def char_script_shares(text):
    lat = cyr = other_letters = 0
    for ch in text:
        if ch.isalpha() and ch not in "\u02BB\u02BC":   # ʻ ʼ are script-neutral modifier letters
            n = unicodedata.name(ch, "")
            if n.startswith("LATIN"): lat += 1
            elif n.startswith("CYRILLIC"): cyr += 1
            else: other_letters += 1
    tot = lat + cyr + other_letters
    return (lat / tot if tot else np.nan, cyr / tot if tot else np.nan, other_letters / tot if tot else np.nan)

def detect_script(lat, cyr, thr=0.90):
    if np.isnan(lat): return "NoLetters"
    if lat >= thr: return "Latin"
    if cyr >= thr: return "Cyrillic"
    return "Mixed"

def ocr_noise(text):
    n = len(text)
    if n == 0: return dict(nonalnum_ratio=np.nan, digit_ratio=np.nan, short_line_ratio=np.nan, repeated_char_runs=0, replacement_chars=0)
    nonalnum = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?-–—()\"'«»“”‘’ʻʼ"))
    lines = [l for l in text.split("\n") if l.strip()]
    return dict(nonalnum_ratio=nonalnum / n, digit_ratio=sum(ch.isdigit() for ch in text) / n,
                short_line_ratio=(sum(len(l.strip()) < 20 for l in lines) / len(lines)) if lines else np.nan,
                repeated_char_runs=len(re.findall(r"(.)\1{5,}", text)), replacement_chars=text.count("�"))

recs = []
for r in pilot.itertuples():
    raw = r.text or ""
    n_ph = len(PLACEHOLDER_RE.findall(raw))
    t = PLACEHOLDER_RE.sub(" ", raw)          # PII placeholders are not natural Uzbek text
    lat, cyr, oth = char_script_shares(t)
    words = WORD_RE.findall(t)
    rec = dict(id=r.id, chars=len(t), words_regex=len(words), words_whitespace=len(t.split()),
               word_count_field=r.word_count, latin_share=lat, cyrillic_share=cyr, other_letter_share=oth,
               script_label=r.script, script_detected=detect_script(lat, cyr),
               placeholders=n_ph, sha256_text=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
               nfc_changed=unicodedata.normalize("NFC", t) != t)
    for ch, name in APOS.items(): rec[f"apos_{name.split()[0].lower()}_{ord(ch):04X}"] = t.count(ch)
    rec.update(ocr_noise(t))
    recs.append(rec)
qc = pd.DataFrame(recs)
qc = qc.merge(pilot[["id", "source", "domain", "register", "format_origin", "publication_year"]], on="id")
qc.to_csv(f"{OUT}/pilot_quality_checks.csv", index=False)

checks = {
  "empty_texts": int((qc.chars == 0).sum()),
  "duplicate_ids": int(qc.id.duplicated().sum()),
  "duplicate_text_hashes": int(qc.sha256_text.duplicated().sum()),
  "docs_with_placeholders": int((qc.placeholders > 0).sum()),
  "docs_not_NFC": int(qc.nfc_changed.sum()),
  "word_count_field_vs_whitespace_exact_match_%": round(100 * (qc.word_count_field == qc.words_whitespace).mean(), 1),
  "word_count_field_vs_regex_median_ratio": round(float((qc.words_regex / qc.word_count_field.replace(0, np.nan)).median()), 3),
}
print(json.dumps(checks, indent=2))
print("\nScript label (rows) vs character-based detection (cols):")
ct = pd.crosstab(qc.script_label, qc.script_detected, margins=True); print(ct)
ct.to_csv(f"{OUT}/pilot_script_label_vs_detected.csv")
apos_cols = [c for c in qc if c.startswith("apos_")]
print("\nApostrophe-like characters, by script label (total counts):")
print(qc.groupby("script_label")[apos_cols].sum())
print("\nOCR vs digital — noise proxies (median):")
print(qc.groupby("format_origin")[["nonalnum_ratio", "digit_ratio", "short_line_ratio"]].median().round(4))
json.dump(checks, open(f"{OUT}/pilot_checks.json", "w"), indent=2)
""")

md(r"""
## 7. Tokenizer loading with version pinning
For each checkpoint, the notebook records the commit SHA, tokenizer class, vocabulary size, and normalizer and pre-tokenizer configuration. Tokenizers are then loaded **at that SHA** (`revision=`), so later stages use exactly the same files.
""")
code(r"""
from transformers import AutoTokenizer
MODELS = {"Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B", "Llama-3.2-1B": "meta-llama/Llama-3.2-1B", "Gemma-2-2B": "google/gemma-2-2b"}
TOK, tok_info = {}, {}
for name, repo in MODELS.items():
    mi = api.model_info(repo)
    tok = AutoTokenizer.from_pretrained(repo, revision=mi.sha, token=HF_TOKEN, use_fast=True)
    TOK[name] = tok
    bk = json.loads(tok.backend_tokenizer.to_str()) if getattr(tok, "is_fast", False) else {}
    tok_info[name] = {"repo": repo, "revision_sha": mi.sha, "gated": getattr(mi, "gated", None),
                      "license": (mi.card_data or {}).get("license") if mi.card_data else None,
                      "class": type(tok).__name__, "is_fast": tok.is_fast, "vocab_size": tok.vocab_size, "len_tokenizer": len(tok),
                      "model_type": (bk.get("model") or {}).get("type"),
                      "normalizer": json.dumps(bk.get("normalizer"))[:300], "pre_tokenizer": json.dumps(bk.get("pre_tokenizer"))[:300],
                      "bos": tok.bos_token, "eos": tok.eos_token, "add_bos_by_default": len(tok("a")["input_ids"]) > len(tok("a", add_special_tokens=False)["input_ids"])}
json.dump(tok_info, open(f"{OUT}/tokenizers.json", "w"), indent=2)
pd.DataFrame(tok_info).T[["revision_sha", "class", "model_type", "vocab_size", "len_tokenizer", "gated", "license"]]
""")

md(r"""
### 7.1 Sanity check: lossless round-trip and Uzbek-specific characters
Every tokenizer must decode back to the original text. If it does not, its counts are not comparable. The test strings cover the apostrophe variants in *oʻ / gʻ* and Cyrillic *ў, қ, ғ, ҳ*.
""")
code(r"""
PROBES = ["Oʻzbekiston Respublikasi", "O‘zbekiston Respublikasi", "O'zbekiston Respublikasi", "Ўзбекистон Республикаси",
          "maʼlumot", "ma’lumot", "ma'lumot", "маълумот", "gʻalaba gʻoya", "ғалаба ғоя", "qishloq xoʻjaligi", "қишлоқ хўжалиги"]
rt = []
for name, tok in TOK.items():
    for s in PROBES:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        dec = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rt.append({"tokenizer": name, "text": s, "n_tokens": len(ids), "roundtrip_ok": dec == s,
                   "pieces": " | ".join(tok.convert_ids_to_tokens(ids))})
rt = pd.DataFrame(rt); rt.to_csv(f"{OUT}/probe_tokenizations.csv", index=False)
print("Round-trip failures:", int((~rt.roundtrip_ok).sum()))
rt.pivot(index="text", columns="tokenizer", values="n_tokens")
""")

md(r"""
## 8. Pilot measurements (descriptive only)
**Definitions** (fixed now, so all later stages use the same ones):
* **Word**: maximal run of Unicode letters, allowing word-internal apostrophes (`WORD_RE`). Digits and punctuation are not words.
* **Tokens per word (TPW)** = tokens(text) / words(text). Text is tokenized in context, without special tokens.
* **Token fertility** = mean number of tokens per word when each word is tokenized *in isolation* (with a leading space). **Proportion of continued words (PCW)** = share of words split into ≥2 tokens.
* **Characters per token (CPT)** = characters(text) / tokens(text).
* **Sequence expansion (SE)** = tokens(text, tokenizer) / min over tokenizers of tokens(text). It equals 1 for the most compact tokenizer on each document.

TPW, CPT, and SE are mathematical transforms of the same token count and are therefore **highly correlated**. The paper will report TPW as the primary efficiency metric, and treat CPT and SE as complementary views, **not** as independent evidence. Fertility and PCW (word level) and morpheme-boundary alignment (Stage 3) measure *how* words are split.

**Placeholders:** PII placeholders inserted by the corpus builders (e.g. `[PHONE]`, `[NAME]`) are removed before measurement, because they are not natural Uzbek text.

**Length control:** each document is truncated to its first `MAX_CHARS` characters at a word boundary. Very long books therefore cannot dominate, and the truncation is recorded.
""")
code(r"""
MAX_CHARS = 20_000
def cap(t, n=MAX_CHARS):
    if len(t) <= n: return t, False
    cut = t.rfind(" ", 0, n); return t[: cut if cut > 0 else n], True

rows, word_freq = [], {}
for r in pilot.itertuples():
    t, trunc = cap(re.sub(r"[ \t]{2,}", " ", PLACEHOLDER_RE.sub(" ", r.text or "")))
    words = WORD_RE.findall(t)
    if not words: continue
    for w in words: word_freq[w] = word_freq.get(w, 0) + 1
    rec = dict(id=r.id, chars=len(t), words=len(words), truncated=trunc, script=r.script, source=r.source,
               domain=r.domain, register=r.register, format_origin=r.format_origin)
    for name, tok in TOK.items():
        rec[f"tok_{name}"] = len(tok(t, add_special_tokens=False)["input_ids"])
    rows.append(rec)
m = pd.DataFrame(rows)
tcols = [f"tok_{n}" for n in TOK]
m["tok_min"] = m[tcols].min(axis=1)
for n in TOK:
    m[f"TPW_{n}"] = m[f"tok_{n}"] / m.words
    m[f"CPT_{n}"] = m.chars / m[f"tok_{n}"]
    m[f"SE_{n}"] = m[f"tok_{n}"] / m.tok_min
m.to_csv(f"{OUT}/pilot_doc_metrics.csv", index=False)
print(f"Documents measured: {len(m)} | truncated: {int(m.truncated.sum())}")

# word-level fertility on word TYPES weighted by frequency (each type tokenized once, in isolation)
wf = pd.Series(word_freq, name="freq").rename_axis("word").reset_index()
for name, tok in TOK.items():
    enc = tok([" " + w for w in wf.word], add_special_tokens=False)["input_ids"]
    wf[f"ntok_{name}"] = [len(x) for x in enc]
wf.to_csv(f"{OUT}/pilot_word_types.csv", index=False)
fert = {n: {"fertility_token_weighted": float(np.average(wf[f"ntok_{n}"], weights=wf.freq)),
            "PCW_%": float(100 * np.average(wf[f"ntok_{n}"] >= 2, weights=wf.freq)),
            "fertility_type_level": float(wf[f"ntok_{n}"].mean())} for n in TOK}
print(pd.DataFrame(fert).T.round(3))
json.dump(fert, open(f"{OUT}/pilot_fertility.json", "w"), indent=2)
""")

code(r"""
def corpus_level(df):
    out = {}
    for n in TOK:
        out[n] = {"TPW_micro": df[f"tok_{n}"].sum() / df.words.sum(),
                  "TPW_doc_median": df[f"TPW_{n}"].median(),
                  "CPT_micro": df.chars.sum() / df[f"tok_{n}"].sum(),
                  "SE_doc_median": df[f"SE_{n}"].median()}
    return pd.DataFrame(out).T.round(3)
print("=== All pilot documents ===");  print(corpus_level(m))
for s in ["Latin", "Cyrillic", "Mixed"]:
    sub = m[m.script == s]
    if len(sub): print(f"\n=== script = {s} (n={len(sub)}) ==="); print(corpus_level(sub))
corpus_level(m).to_csv(f"{OUT}/pilot_summary_all.csv")
print("\nSpearman correlations between TPW, CPT, SE (Qwen) — illustrates metric redundancy:")
q = list(TOK)[0]; print(m[[f"TPW_{q}", f"CPT_{q}", f"SE_{q}"]].corr(method="spearman").round(3))
""")

code(r"""
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
data = [m[f"TPW_{n}"] for n in TOK]
ax[0].boxplot(data, showfliers=False); ax[0].set_xticks(range(1, len(TOK) + 1), list(TOK)); ax[0].set_ylabel("Tokens per word"); ax[0].set_title("(a) All pilot documents", loc="left")
for i, s in enumerate(["Latin", "Cyrillic"]):
    sub = m[m.script == s]
    ax[1].bar(np.arange(len(TOK)) + (i - 0.5) * 0.38, [sub[f"tok_{n}"].sum() / sub.words.sum() for n in TOK], 0.36, label=f"{s} (n={len(sub)})")
ax[1].set_xticks(range(len(TOK))); ax[1].set_xticklabels(list(TOK)); ax[1].set_ylabel("TPW (micro-average)")
ax[1].set_title("(b) By script label — pilot only", loc="left"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/pilot_tpw.png", dpi=200); plt.show()
""")

md(r"""
## 9. Stage-1 validation checklist
Send the output of this cell (and the files in `stage1/`) back before moving to Stage 2.
""")
code(r"""
gate = {
  "card_vs_data_all_match": bool(ver.match.all()),
  "schema_matches_card": (missing == [] and extra == []),
  "pilot_size_1000": len(pilot) == 1000,
  "no_duplicate_ids_or_texts": checks["duplicate_ids"] == 0 and checks["duplicate_text_hashes"] == 0,
  "all_tokenizers_loaded_at_pinned_sha": len(TOK) == 3,
  "all_probe_roundtrips_ok": bool(rt.roundtrip_ok.all()),
  "script_label_agreement_%": round(100 * (qc.script_label == qc.script_detected).mean(), 1),
}
json.dump(gate, open(f"{OUT}/stage1_gate.json", "w"), indent=2)
print(json.dumps(gate, indent=2))
if "DRIVE_OUT" in globals():
    import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True); print("Copied to", DRIVE_OUT)
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                  "colab": {"provenance": []}}
nbf.write(nb, "Stage1_Dataset_Inspection_and_Pilot.ipynb")
print("ok", len(C), "cells")
