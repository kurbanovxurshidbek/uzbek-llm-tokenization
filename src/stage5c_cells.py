# Stage 5c — Experiment 3: alignment of tokenizer boundaries with gold morpheme boundaries (RQ3)
import stage4_cells as _s4
TRANSLIT_SRC = _s4.CELLS[2][1].strip().replace("# 2 — transliterator (both directions) and unit tests", "# 3 — transliterator v4 (identical to Stage 4A) + unit tests")
TRANSLIT_SRC += '\nassert not fails, "transliterator unit tests failed — do not continue"'

CELLS = []
CELLS.append(("md", r"""
# Stage 5c — Experiment 3: do tokenizer boundaries follow morpheme boundaries? (RQ3)
**Gold standard:** 500 Latin-script word types from the corpus, in 4 frequency bands of 125 words each. The segmentation is a surface morpheme segmentation (`morph_gold_*.csv`). The Cyrillic version is derived by transliterating each morpheme separately; a word is used in Cyrillic only if the joined morphemes equal the transliteration of the whole word.

**Tokenizer boundaries** come from the character offsets of the tokens of `" " + word`. Two tokens that are byte pieces of the same character share one character span, so they never create a boundary inside a character.

**Metrics:**
* boundary precision, recall and F1, micro-averaged over words;
* exact-match rate;
* *morph-consistent* rate: every token boundary is a morpheme boundary;
* tokens per morpheme, and Spearman ρ between token count and morpheme count.

**Chance baseline:** the expected precision and recall if the same number of boundaries were placed uniformly at random inside each word.

CIs come from a bootstrap over words (1,000 replicates).
"""))

CELLS.append(("code", r"""
# 1 — setup and gold standard
import os, re, json, time, datetime, unicodedata
import numpy as np, pandas as pd
import importlib.metadata as im
from scipy.stats import spearmanr
SEED = 42; N_BOOT = 1000
MODELS = {
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B",       "8faed761d45a263340a0528343f099c05c9a4323"),
    "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    "Gemma-2-2B":   ("google/gemma-2-2b",       "c5ebcd40d208330abc697524c919956e692655cf"),
}
NAMES = list(MODELS)
from google.colab import drive
drive.mount("/content/drive")
S5 = "/content/drive/MyDrive/uz_tokenization/stage5"
OUT = "/content/uz_tok/stage5c"; os.makedirs(f"{OUT}/figures", exist_ok=True)
DRIVE_OUT = "/content/drive/MyDrive/uz_tokenization/stage5c"; os.makedirs(DRIVE_OUT, exist_ok=True)
print("Files in stage5/:", sorted(os.listdir(S5)))
GOLD_FILE = next(f for f in [f"{S5}/morph_gold_reviewed.csv", f"{S5}/morph_gold_v2.csv"] if os.path.exists(f))
assert "v1" not in GOLD_FILE, "v1 gold file must not be used"
gold = pd.read_csv(GOLD_FILE, dtype=str, keep_default_na=False)
gold["freq"] = gold.freq.astype(int)
assert all(s.replace("-", "") == w for w, s in zip(gold.word, gold.segmentation)), "segmentation does not reproduce the word"
PKG = {p: im.version(p) for p in ["transformers", "tokenizers", "huggingface_hub", "pandas", "numpy", "scipy", "matplotlib"]}
print("Gold file:", GOLD_FILE, "| words:", len(gold)); print(gold.word_class.value_counts().to_dict()); print(PKG)
"""))

CELLS.append(("code", r"""
# 2 — tokenizers (pinned)
from huggingface_hub import login
try:
    from google.colab import userdata; HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    import getpass; HF_TOKEN = getpass.getpass("Hugging Face token (hf_...): ")
login(token=HF_TOKEN)
from transformers import AutoTokenizer
TOK = {n: AutoTokenizer.from_pretrained(r, revision=s, token=HF_TOKEN, use_fast=True) for n, (r, s) in MODELS.items()}
print({n: type(t).__name__ for n, t in TOK.items()})
"""))

CELLS.append(("code", TRANSLIT_SRC))

CELLS.append(("code", r"""
# 4 — Cyrillic version of the gold standard (morpheme-wise transliteration) and gold boundary sets
def bounds_from_morphs(morphs):
    pos, b = 0, set()
    for m in morphs[:-1]:
        pos += len(m); b.add(pos)
    return b

rows = []
for r in gold.itertuples():
    morphs = r.segmentation.split("-")
    cyr_morphs = [lat2cyr(m) for m in morphs]; cyr_word = lat2cyr(r.word)
    rows.append({"word_id": r.word_id, "script": "Latin", "word": r.word, "morphs": morphs, "ok": True})
    rows.append({"word_id": r.word_id, "script": "Cyrillic", "word": cyr_word, "morphs": cyr_morphs, "ok": "".join(cyr_morphs) == cyr_word})
G = pd.DataFrame(rows).merge(gold[["word_id", "freq", "band", "word_class"]], on="word_id")
print("Cyrillic words dropped (morpheme-wise transliteration ≠ whole-word transliteration):",
      G[(G.script == "Cyrillic") & ~G.ok][["word"]].assign(latin=lambda d: gold.set_index("word_id").loc[G[(G.script == "Cyrillic") & ~G.ok].word_id, "segmentation"].values).to_string(index=False))
G = G[G.ok].reset_index(drop=True)
G["gold_b"] = G.morphs.apply(bounds_from_morphs); G["n_morphs"] = G.morphs.str.len(); G["L"] = G.word.str.len()
print(G.groupby("script").size().to_dict())
print(G[G.script == "Cyrillic"].head(8)[["word", "morphs"]].to_string(index=False))
"""))

CELLS.append(("code", r"""
# 5 — tokenizer boundaries from character offsets (with an integrity check)
def tok_info(n, w):
    enc = TOK[n](" " + w, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ends = [e for s, e in offs]
    b = {e - 1 for e in ends[:-1] if 0 < e - 1 < len(w)}
    ok = all(e1 <= e2 for e1, e2 in zip(ends, ends[1:])) and ends[-1] == len(w) + 1
    return len(ids), b, ok

bad = {}
for n in NAMES:
    info = [tok_info(n, w) for w in G.word]
    G[f"ntok_{n}"] = [i[0] for i in info]; G[f"tok_b_{n}"] = [i[1] for i in info]
    bad[n] = int(sum(not i[2] for i in info))
print("Words whose token offsets fail the integrity check (should be 0):", bad)
ex = G[G.word.isin(["oʻzbekistonda", "koʻrsatilgan", "xususiyatlarini", "ўзбекистонда", "кўрсатилган", "хусусиятларини"])]
for r in ex.itertuples():
    segs = []
    for n in NAMES:
        tb = G.loc[r.Index, f"tok_b_{n}"]
        segs.append(f"{n}: " + "".join(ch + ("|" if (i + 1) in tb else "") for i, ch in enumerate(r.word)))
    print(f"{r.script:8s} gold {'-'.join(r.morphs):26s} " + "   ".join(segs))
"""))

CELLS.append(("code", r"""
# 6 — Experiment 3 metrics with bootstrap CIs (primary set: native + loan words; secondary: all 500)
def metric_arrays(df, n):
    tp = np.array([len(t & g) for t, g in zip(df[f"tok_b_{n}"], df.gold_b)], float)
    nt = np.array([len(t) for t in df[f"tok_b_{n}"]], float); ng = np.array([len(g) for g in df.gold_b], float)
    exact = np.array([t == g for t, g in zip(df[f"tok_b_{n}"], df.gold_b)], float)
    consistent = np.array([t <= g for t, g in zip(df[f"tok_b_{n}"], df.gold_b)], float)
    slots = np.maximum(df.L.values - 1, 1).astype(float)
    exp_tp = nt * ng / slots                                       # expected hits if nt boundaries were placed at random
    return tp, nt, ng, exact, consistent, exp_tp

def summarize(tp, nt, ng, exact, cons, exp_tp, ntok, nm):
    P = tp.sum() / nt.sum(); R = tp.sum() / ng.sum(); F = 2 * P * R / (P + R) if P + R else 0
    Pc = exp_tp.sum() / nt.sum(); Rc = exp_tp.sum() / ng.sum(); Fc = 2 * Pc * Rc / (Pc + Rc) if Pc + Rc else 0
    return np.array([100 * P, 100 * R, 100 * F, 100 * exact.mean(), 100 * cons.mean(), ntok.sum() / nm.sum(), 100 * Fc])
KEYS = ["precision", "recall", "F1", "exact_match", "morph_consistent", "tokens_per_morph", "F1_chance"]

rows = []; rng = np.random.default_rng(SEED)
for subset, mask in [("native+loan (primary)", G.word_class.isin(["native", "loan"])), ("all 500", G.word_class.notna())]:
    for s in ["Latin", "Cyrillic"]:
        df = G[mask & (G.script == s)].reset_index(drop=True)
        idx_boot = [rng.integers(0, len(df), len(df)) for _ in range(N_BOOT)]
        for n in NAMES:
            arrs = metric_arrays(df, n); ntok = df[f"ntok_{n}"].values.astype(float); nm = df.n_morphs.values.astype(float)
            point = summarize(*arrs, ntok, nm)
            reps = np.array([summarize(*(a[ix] for a in arrs), ntok[ix], nm[ix]) for ix in idx_boot])
            lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
            rho = spearmanr(df[f"ntok_{n}"], df.n_morphs).correlation
            rows.append({"subset": subset, "script": s, "tokenizer": n, "n_words": len(df), "gold_boundaries": int(arrs[2].sum()),
                         **{k: point[i] for i, k in enumerate(KEYS)}, **{k + "_lo": lo[i] for i, k in enumerate(KEYS)},
                         **{k + "_hi": hi[i] for i, k in enumerate(KEYS)}, "spearman_ntok_vs_nmorph": rho})
exp3 = pd.DataFrame(rows); exp3.to_csv(f"{OUT}/exp3_alignment.csv", index=False)
show = exp3.copy()
for k in KEYS:
    d = 3 if k == "tokens_per_morph" else 1
    show[k] = [f"{a:.{d}f} [{b:.{d}f}, {c:.{d}f}]" for a, b, c in zip(show[k], show[k + "_lo"], show[k + "_hi"])]
pd.set_option("display.width", 260); pd.set_option("display.max_columns", None)
for subset in show.subset.unique():
    print(f"\n=== {subset} ===")
    print(show[show.subset == subset][["script", "tokenizer", "n_words", "precision", "recall", "F1", "F1_chance"]].to_string(index=False))
    print(show[show.subset == subset][["script", "tokenizer", "exact_match", "morph_consistent", "tokens_per_morph", "spearman_ntok_vs_nmorph"]].round(3).to_string(index=False))
"""))

CELLS.append(("code", r"""
# 7 — paired tokenizer differences in boundary F1 (same words) and breakdowns by frequency band and word class
def f1_of(df, n):
    tp, nt, ng, *_ = metric_arrays(df, n); P = tp.sum() / nt.sum(); R = tp.sum() / ng.sum(); return 200 * P * R / (P + R)
rows = []; rng = np.random.default_rng(SEED + 1)
prim = G[G.word_class.isin(["native", "loan"])]
for s in ["Latin", "Cyrillic"]:
    df = prim[prim.script == s].reset_index(drop=True)
    for a, b in [("Gemma-2-2B", "Qwen2.5-1.5B"), ("Gemma-2-2B", "Llama-3.2-1B"), ("Llama-3.2-1B", "Qwen2.5-1.5B")]:
        d0 = f1_of(df, a) - f1_of(df, b)
        reps = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, len(df), len(df)); sub = df.iloc[ix]
            reps.append(f1_of(sub, a) - f1_of(sub, b))
        lo, hi = np.percentile(reps, [2.5, 97.5])
        rows.append({"script": s, "comparison": f"{a} − {b}", "delta_F1_pp": round(d0, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
pairs = pd.DataFrame(rows); pairs.to_csv(f"{OUT}/exp3_paired_F1.csv", index=False); print(pairs.to_string(index=False))

br = []
for key in ["band", "word_class"]:
    for (s, g), df in G[G.word_class.isin(["native", "loan"]) | (key == "word_class")].groupby(["script", key]):
        rec = {"breakdown": key, "script": s, "group": g, "n_words": len(df), "mean_morphs": round(df.n_morphs.mean(), 2)}
        for n in NAMES:
            rec[f"F1_{n}"] = round(f1_of(df, n), 1) if any(len(x) for x in df.gold_b) else np.nan
            rec[f"tok_per_morph_{n}"] = round(df[f"ntok_{n}"].sum() / df.n_morphs.sum(), 3)
        br.append(rec)
br = pd.DataFrame(br); br.to_csv(f"{OUT}/exp3_breakdowns.csv", index=False); print(); print(br.to_string(index=False))

# fragmentation vs morphological complexity: mean tokens by number of morphemes (primary set)
cx = prim.groupby(["script", "n_morphs"]).agg(n=("word", "size"), **{f"tok_{n}": (f"ntok_{n}", "mean") for n in NAMES}).round(2)
cx.to_csv(f"{OUT}/exp3_tokens_by_nmorphs.csv"); print(); print(cx.to_string())
"""))

CELLS.append(("code", r"""
# 8 — figures (PDF + PNG, 300 dpi)
import matplotlib; import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8985", "axes.labelcolor": "#33322e", "xtick.color": "#55544f", "ytick.color": "#55544f",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
COL = {"Qwen2.5-1.5B": "#2a78d6", "Llama-3.2-1B": "#eb6834", "Gemma-2-2B": "#1baf7a"}
MRK = {"Qwen2.5-1.5B": "o", "Llama-3.2-1B": "s", "Gemma-2-2B": "^"}
SAVE_ERRORS = []
def save(fig, name):
    for ext, kw in [("png", {"dpi": 300}), ("pdf", {})]:
        try: fig.savefig(f"{OUT}/figures/{name}.{ext}", bbox_inches="tight", **kw)
        except Exception as ex: SAVE_ERRORS.append(f"{name}.{ext}: {ex}")

e = exp3[exp3.subset.str.startswith("native")]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]; x = np.arange(2); w = 0.26
for i, n in enumerate(NAMES):
    d = e[e.tokenizer == n].set_index("script").loc[["Latin", "Cyrillic"]]; xp = x + (i - 1) * (w + 0.02)
    ax.bar(xp, d.F1, w, color=COL[n], label=n, edgecolor="white", linewidth=0.6)
    ax.errorbar(xp, d.F1, yerr=np.clip([d.F1 - d.F1_lo, d.F1_hi - d.F1], 0, None), fmt="none", ecolor="#33322e", elinewidth=0.8, capsize=2)
    ax.scatter(xp, d.F1_chance, marker="_", s=120, color="#33322e", zorder=3, label="chance" if i == 0 else None)
    for xi, v in zip(xp, d.F1): ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5, color="#33322e")
ax.set_xticks(x); ax.set_xticklabels(["Latin", "Cyrillic"]); ax.set_ylabel("Boundary F1 (%)"); ax.set_ylim(0, 100)
ax.set_title("(a) Morpheme-boundary F1", loc="left"); ax.grid(axis="x", visible=False)
ax = axes[1]
lat = prim[prim.script == "Latin"]; mx = int(min(lat.n_morphs.max(), 6))
for n in NAMES:
    m = lat[lat.n_morphs <= mx].groupby("n_morphs")[f"ntok_{n}"].mean()
    ax.plot(m.index, m.values, marker=MRK[n], color=COL[n], ms=4, lw=1.2, label=n)
ax.plot(range(1, mx + 1), range(1, mx + 1), ls="--", color="#8a8985", lw=0.8, label="1 token per morpheme")
ax.set_xlabel("Morphemes per word (gold)"); ax.set_ylabel("Mean tokens per word"); ax.set_title("(b) Fragmentation vs morphology (Latin)", loc="left")
h, l = axes[0].get_legend_handles_labels(); h2, l2 = ax.get_legend_handles_labels()
fig.legend(h[:3] + [h2[-1], h[-1]] if len(h) > 3 else h + h2[-1:], (l[:3] + [l2[-1], l[-1]]) if len(l) > 3 else l + l2[-1:],
           loc="upper center", ncol=5, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(); save(fig, "fig_exp3_morph_alignment"); plt.show()
print("Save errors:", SAVE_ERRORS or "none")
"""))

CELLS.append(("code", r"""
# 9 — gate and copy to Drive
gate = {
  "gold_file": GOLD_FILE, "gold_words": len(gold),
  "words_used": G.groupby("script").size().to_dict(),
  "offset_integrity_failures": bad,
  "transliterator_unit_test_failures": len(fails),
  "F1_primary": {f"{r.script} | {r.tokenizer}": round(r.F1, 2) for r in exp3[exp3.subset.str.startswith("native")].itertuples()},
  "figures": sorted(os.listdir(f"{OUT}/figures")), "packages": PKG, "seed": SEED, "n_boot": N_BOOT,
  "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
json.dump(gate, open(f"{OUT}/stage5c_gate.json", "w"), indent=2, ensure_ascii=False); print(json.dumps(gate, indent=2, ensure_ascii=False))
G.assign(**{c: G[c].apply(lambda s: sorted(s)) for c in G if c.endswith("_b") or c.startswith("tok_b_")}).to_json(f"{OUT}/exp3_word_level.json", orient="records", force_ascii=False)
import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True); print("Copied to", DRIVE_OUT)
"""))
