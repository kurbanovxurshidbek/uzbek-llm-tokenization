CELL = r'''
# Stage 5a — candidate word types for the morphological gold standard (Latin script, official spelling)
import os, re, json, unicodedata, datetime
import numpy as np, pandas as pd
from collections import Counter
from google.colab import drive
drive.mount("/content/drive")
SEED = 42; MIN_STRATUM_N = 30; N_PER_BAND = 125
S2 = "/content/drive/MyDrive/uz_tokenization/stage2"
OUT = "/content/drive/MyDrive/uz_tokenization/stage5"; os.makedirs(OUT, exist_ok=True)
OKINA, TUTUQ = "ʻ", "ʼ"
APOS_ANY = "'‘’ʻʼ`´"
WORD_RE = re.compile(rf"[^\W\d_]+(?:[{APOS_ANY}][^\W\d_]+)*", re.UNICODE)
def norm_lat_apos(w):
    return re.sub(rf"([^{APOS_ANY}])[{APOS_ANY}]", lambda m: m.group(1) + (OKINA if m.group(1) in "oOgG" else TUTUQ), w)
def is_uz_latin(w):
    letters = [ch for ch in w if ch.isalpha() and ch not in OKINA + TUTUQ]
    lw = w.lower()
    return (all(unicodedata.name(ch, "").startswith("LATIN") for ch in letters)
            and "w" not in lw and "c" not in lw.replace("ch", "") and not re.search(r"[^a-zʻʼ]", lw))

mem = pd.read_csv(f"{S2}/sample_membership_final.csv")
docs = pd.read_parquet(f"{S2}/doc_metrics.parquet", columns=["gidx", "script"])
texts = pd.read_parquet(f"{S2}/sample_clean.parquet", columns=["gidx", "text"]).set_index("gidx").text
A = mem[mem.kept & (mem["sample"] == "A")].merge(docs, on="gidx")
sizes = A.groupby("stratum").size(); A = A[A.stratum.isin(sizes[sizes >= MIN_STRATUM_N].index)]
freq = Counter(norm_lat_apos(w).lower() for g in A.loc[A.script == "Latin", "gidx"] for w in WORD_RE.findall(texts[g]))
cand = pd.DataFrame([(w, c) for w, c in freq.items() if len(w) >= 3 and c >= 2 and is_uz_latin(w)], columns=["word", "freq"])
cand = cand.sort_values(["freq", "word"], ascending=[False, True]).reset_index(drop=True); cand["rank"] = np.arange(1, len(cand) + 1)
bands = [("B1 rank 1–1k", 1, 1_000), ("B2 rank 1k–10k", 1_001, 10_000), ("B3 rank 10k–50k", 10_001, 50_000), ("B4 rank >50k", 50_001, 10**9)]
rng = np.random.default_rng(SEED); parts = []
for name, lo, hi in bands:
    pool = cand[(cand["rank"] >= lo) & (cand["rank"] <= hi)]
    take = pool.iloc[np.sort(rng.choice(len(pool), size=min(N_PER_BAND, len(pool)), replace=False))].copy()
    take["band"] = name; parts.append(take)
    print(f"{name}: pool {len(pool):,} types → sampled {len(take)}")
gold = pd.concat(parts, ignore_index=True)
gold.insert(0, "word_id", [f"w{i:03d}" for i in range(1, len(gold) + 1)])
gold["segmentation"] = ""; gold["word_class"] = ""; gold["note"] = ""
gold.to_csv(f"{OUT}/morph_candidates.csv", index=False, encoding="utf-8-sig")
json.dump({"seed": SEED, "n_per_band": N_PER_BAND, "bands": bands, "candidate_types": len(cand),
           "latin_word_tokens": int(sum(freq.values())), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open(f"{OUT}/stage5a_params.json", "w"), indent=2, ensure_ascii=False)
print(f"\nCandidate types (Latin, ≥3 chars, freq ≥2, Uzbek letters only): {len(cand):,}")
print(f"Saved {len(gold)} words → {OUT}/morph_candidates.csv"); print(gold.head(10).to_string(index=False))
'''
