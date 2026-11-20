# Stage 3 cells — Experiments 1 (cross-tokenizer, RQ1/RQ5 descriptive) and 4 (domain & register, RQ4)
CELLS = []

CELLS.append(("md", r"""
# Stage 3 — Experiment 1 (cross-tokenizer comparison) and Experiment 4 (domains and registers)
**Inputs:** Stage-2 outputs on Google Drive (`uz_tokenization/stage2/`). No data is downloaded again.

**What this stage adds:**
1. **Population-weighted estimates with 95 % stratified-bootstrap confidence intervals.** Sample A gives each stratum the same number of documents, so corpus-level values must weight each stratum by its real size.
2. **Paired tokenizer differences:** the same documents are measured by all three tokenizers.
3. **A character-coverage diagnostic,** explaining *why* the tokenizers differ:
   * the **alphabet coverage table**: is each Uzbek letter a single token or is it split into raw bytes?
   * the **byte-fragment rate**: the share of tokens that do not decode to a complete character (computed over all word types).
4. **Experiment 4:** fertility by domain (22) and register (8), separately for each script, plus a ranking-stability check.
5. **Publication-quality figures** (PDF + PNG, 300 dpi).

Primary metrics, as decided in Stage 2: **fertility** (tokens per word) and **PCW** (share of words split into ≥ 2 tokens). **SE** is reported as a relative measure. TPW-all is in the appendix.
Formal hypothesis tests, effect sizes, and multiple-comparison control come in **Stage 6**.
"""))

CELLS.append(("code", r"""
# 1 — setup and inputs
import os, sys, json, re, time, unicodedata, datetime, zlib
import numpy as np, pandas as pd
import importlib.metadata as im
from collections import Counter

SEED = 42; rng_global = np.random.default_rng(SEED)
N_BOOT = 1000
MIN_STRATUM_N = 30
MODELS = {
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B",       "8faed761d45a263340a0528343f099c05c9a4323"),
    "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    "Gemma-2-2B":   ("google/gemma-2-2b",       "c5ebcd40d208330abc697524c919956e692655cf"),
}
NAMES = list(MODELS)
VOCAB = {"Qwen2.5-1.5B": 151_643, "Llama-3.2-1B": 128_000, "Gemma-2-2B": 256_000}

from google.colab import drive
drive.mount("/content/drive")
S2 = "/content/drive/MyDrive/uz_tokenization/stage2"
OUT = "/content/uz_tok/stage3"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
DRIVE_OUT = "/content/drive/MyDrive/uz_tokenization/stage3"; os.makedirs(DRIVE_OUT, exist_ok=True)

docs = pd.read_parquet(f"{S2}/doc_metrics.parquet")
mem  = pd.read_csv(f"{S2}/sample_membership_final.csv")
plan = pd.read_csv(f"{S2}/sampling_plan.csv")
achieved = pd.read_csv(f"{S2}/strata_achieved.csv")
texts = pd.read_parquet(f"{S2}/sample_clean.parquet", columns=["gidx", "text"])
kept = mem[mem.kept].merge(docs, on="gidx", how="left")
assert kept[[f"fert_{n}" for n in NAMES]].notna().all().all()
print(f"docs: {len(docs):,} | kept memberships: {len(kept):,} | samples: {kept['sample'].value_counts().to_dict()}")
PKG = {p: im.version(p) for p in ["transformers", "tokenizers", "huggingface_hub", "pyarrow", "pandas", "numpy", "scipy", "matplotlib"]}
print(PKG)
json.dump({"packages": PKG,
           "seed": SEED, "n_boot": N_BOOT, "min_stratum_n": MIN_STRATUM_N,
           "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"}, open(f"{OUT}/stage3_params.json", "w"), indent=2)
"""))

CELLS.append(("code", r"""
# 2 — tokenizers (pinned) and per-vocabulary byte-fragment flags
from huggingface_hub import login
try:
    from google.colab import userdata; HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    import getpass; HF_TOKEN = getpass.getpass("Hugging Face token (hf_...): ")
login(token=HF_TOKEN)
from transformers import AutoTokenizer
TOK = {n: AutoTokenizer.from_pretrained(r, revision=s, token=HF_TOKEN, use_fast=True) for n, (r, s) in MODELS.items()}

BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")          # SentencePiece byte-fallback tokens
FRAG = {}
for n, tok in TOK.items():
    V = len(tok)
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    dec = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(V)]
    FRAG[n] = np.array([("�" in d) or bool(BYTE_TOKEN_RE.match(p or "")) for d, p in zip(dec, pieces)])
    print(f"{n}: {FRAG[n].sum():,} of {V:,} vocabulary entries are byte fragments")
"""))

CELLS.append(("code", r"""
# 3 — alphabet coverage: is each Uzbek letter a single token, or split into raw bytes?
LAT = list("ABCDEFGHIJKLMNOPQRSTUVXYZ") + ["ʻ", "ʼ", "'", "‘", "’"]
CYR = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЪЬЭЮЯЎҚҒҲ")
rows = []
for script, letters in [("Latin", LAT), ("Cyrillic", CYR)]:
    for ch in letters:
        variants = [ch] if not ch.isalpha() or ch in "ʻʼ" else [ch, ch.lower()]
        for v in variants:
            rec = {"script": script, "char": v, "codepoint": f"U+{ord(v):04X}", "utf8_bytes": len(v.encode("utf-8"))}
            for n, tok in TOK.items():
                ids = tok(v, add_special_tokens=False)["input_ids"]
                rec[f"ntok_{n}"] = len(ids)
                rec[f"bytesplit_{n}"] = bool(FRAG[n][ids].any())
            rows.append(rec)
cov = pd.DataFrame(rows)
cov.to_csv(f"{OUT}/alphabet_coverage.csv", index=False)
split_tbl = cov.groupby("script")[[f"bytesplit_{n}" for n in NAMES]].sum()
print("Letters (upper+lower) that are split into raw bytes when written alone:\n", split_tbl)
print("\nWhich letters:")
for n in NAMES:
    for s in ["Latin", "Cyrillic"]:
        bad = cov[(cov.script == s) & cov[f"bytesplit_{n}"]]["char"].tolist()
        print(f"  {n:13s} {s:8s}: {' '.join(bad) if bad else '—'}")
"""))

CELLS.append(("code", r"""
# 4 — byte-fragment rate over all word types, then per document
APOS_CLASS = "['‘’ʻʼ`´]"
WORD_RE = re.compile(rf"[^\W\d_]+(?:{APOS_CLASS}[^\W\d_]+)*", re.UNICODE)
texts = texts.set_index("gidx").loc[docs.gidx].reset_index()
word_lists = [WORD_RE.findall(t) for t in texts.text]
types = sorted({w for ws in word_lists for w in ws})
print(f"word types: {len(types):,}")
frag_type, ntok_type = {}, {}
t0 = time.time()
for n, tok in TOK.items():
    fr, nt = {}, {}
    for i in range(0, len(types), 2048):
        chunk = types[i:i + 2048]
        enc = tok([" " + w for w in chunk], add_special_tokens=False)["input_ids"]
        for w, ids in zip(chunk, enc):
            fr[w] = int(FRAG[n][ids].sum()); nt[w] = len(ids)
    frag_type[n], ntok_type[n] = fr, nt
    print(f"{n} done ({time.time()-t0:.0f}s)")

for n in NAMES:
    f_tok = np.array([sum(frag_type[n][w] for w in ws) for ws in word_lists], dtype=float)
    a_tok = np.array([sum(ntok_type[n][w] for w in ws) for ws in word_lists], dtype=float)
    docs[f"frag_{n}"] = f_tok / a_tok
    assert np.allclose(a_tok / docs.n_words.values, docs[f"fert_{n}"].values, rtol=1e-6), "fertility mismatch with Stage 2"

# character frequencies in the measured text: share of letter occurrences each tokenizer cannot represent as one token
char_freq = {s: Counter() for s in ["Latin", "Cyrillic"]}
for s, ws in zip(docs.script, word_lists):
    if s in char_freq: char_freq[s].update("".join(ws))
cs = []
for s in ["Latin", "Cyrillic"]:
    tot = sum(char_freq[s].values())
    for n in NAMES:
        bad = set(cov[cov[f"bytesplit_{n}"]]["char"])
        cs.append({"script": s, "tokenizer": n, "byte_split_letter_share_%": 100 * sum(c for ch, c in char_freq[s].items() if ch in bad) / tot})
cs = pd.DataFrame(cs).pivot(index="script", columns="tokenizer", values="byte_split_letter_share_%")[NAMES].round(2)
cs.to_csv(f"{OUT}/byte_split_letter_share.csv"); print("\n% of letter occurrences whose letter is not a single token:\n", cs)
docs.drop(columns=[c for c in docs if c == "text"], errors="ignore").to_parquet(f"{OUT}/doc_metrics_with_frag.parquet")
kept = mem[mem.kept].merge(docs, on="gidx", how="left")
"""))

CELLS.append(("code", r"""
# 5 — Experiment 1: population-weighted estimates with stratified-bootstrap 95 % CIs (sample A)
METRICS = {"fertility": "fert", "PCW": "pcw", "SE": "SE", "byte_frag_rate": "frag", "TPW_all": "TPWall"}
A = kept[kept["sample"] == "A"].copy()
pop = plan[plan["sample"] == "A"].set_index("stratum")["available"]
sizes = A.groupby("stratum").size()
use = sizes[sizes >= MIN_STRATUM_N].index
A = A[A.stratum.isin(use)]
print(f"Sample A strata used: {len(use)} (excluded <{MIN_STRATUM_N} docs: {sorted(set(sizes.index) - set(use))})")

def boot_weighted(df, cols, n_boot=N_BOOT, seed=SEED):
    # population-weighted mean of each column; stratified bootstrap over documents within strata
    groups = [(pop[s], g[cols].to_numpy(float)) for s, g in df.groupby("stratum")]
    W = np.array([w for w, _ in groups], float); W /= W.sum()
    point = sum(w * x.mean(0) for w, (_, x) in zip(W, groups))
    rng = np.random.default_rng(seed); reps = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        reps[b] = sum(w * x[rng.integers(0, len(x), len(x))].mean(0) for w, (_, x) in zip(W, groups))
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return point, lo, hi

rows = []
for scope, sub in [("All (population-weighted)", A), ("Latin", A[A.script == "Latin"]), ("Cyrillic", A[A.script == "Cyrillic"])]:
    for mname, pre in METRICS.items():
        cols = [f"{pre}_{n}" for n in NAMES]
        p, lo, hi = boot_weighted(sub, cols)
        scale = 100 if mname in ("PCW", "byte_frag_rate") else 1
        for n, a, l, h in zip(NAMES, p, lo, hi):
            rows.append({"scope": scope, "metric": mname, "tokenizer": n, "estimate": a * scale, "ci_low": l * scale, "ci_high": h * scale})
exp1 = pd.DataFrame(rows)
exp1.to_csv(f"{OUT}/exp1_estimates.csv", index=False)
tbl = exp1.assign(val=lambda d: d.apply(lambda r: f"{r.estimate:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)) \
          .pivot_table(index=["scope", "metric"], columns="tokenizer", values="val", aggfunc="first")[NAMES]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None); print(tbl.to_string())
"""))

CELLS.append(("code", r"""
# 6 — paired tokenizer differences in fertility (same documents), population-weighted, with 95 % CIs
pairs = [("Gemma-2-2B", "Qwen2.5-1.5B"), ("Gemma-2-2B", "Llama-3.2-1B"), ("Llama-3.2-1B", "Qwen2.5-1.5B")]
rows = []
for scope, sub in [("All (population-weighted)", A), ("Latin", A[A.script == "Latin"]), ("Cyrillic", A[A.script == "Cyrillic"])]:
    sub = sub.copy()
    for a, b in pairs:
        sub[f"d_{a}_vs_{b}"] = sub[f"fert_{a}"] - sub[f"fert_{b}"]
        sub[f"r_{a}_vs_{b}"] = sub[f"fert_{a}"] / sub[f"fert_{b}"] - 1
    cols = [f"d_{a}_vs_{b}" for a, b in pairs] + [f"r_{a}_vs_{b}" for a, b in pairs]
    p, lo, hi = boot_weighted(sub, cols)
    for c, x, l, h in zip(cols, p, lo, hi):
        kind, rest = c.split("_", 1)
        rows.append({"scope": scope, "comparison": rest.replace("_vs_", " − " if kind == "d" else " / "),
                     "type": "difference (tokens/word)" if kind == "d" else "relative (%)",
                     "estimate": x * (100 if kind == "r" else 1), "ci_low": l * (100 if kind == "r" else 1), "ci_high": h * (100 if kind == "r" else 1)})
paired = pd.DataFrame(rows); paired.to_csv(f"{OUT}/exp1_paired_differences.csv", index=False)
print(paired.round(3).to_string(index=False))
print("\nShare of documents where each tokenizer is the most compact (fertility):")
win = A[[f"fert_{n}" for n in NAMES]].idxmin(axis=1).str.replace("fert_", "")
print((100 * win.value_counts(normalize=True)).round(1).to_dict())
"""))

CELLS.append(("code", r"""
# 7 — RQ2 (observational): script ratio within each source × format (sample A, Cyrillic / Latin fertility)
rows = []
for (src, fmt), g in A.groupby(["source", "format_origin"]):
    lat, cyr = g[g.script == "Latin"], g[g.script == "Cyrillic"]
    if len(lat) < MIN_STRATUM_N or len(cyr) < MIN_STRATUM_N: continue
    rec = {"source": src, "format_origin": fmt, "n_Latin": len(lat), "n_Cyrillic": len(cyr)}
    for n in NAMES: rec[f"ratio_{n}"] = cyr[f"fert_{n}"].mean() / lat[f"fert_{n}"].mean()
    rows.append(rec)
within = pd.DataFrame(rows).round(3); within.to_csv(f"{OUT}/exp1_within_source_script_ratio.csv", index=False)
print("Cyrillic / Latin mean fertility within the same source and format (1.00 = no difference):"); print(within.to_string(index=False))
"""))

CELLS.append(("code", r"""
# 8 — Experiment 4: domains (sample B) and registers (sample C), per script, with bootstrap CIs
def simple_boot(x, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x, float); r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n_boot, len(x)))].mean(1); return x.mean(), *np.percentile(m, [2.5, 97.5])

def by_group(sample, key):
    S = kept[kept["sample"] == sample]
    rows = []
    for (gval, scr), g in S.groupby([key, "script"]):
        if len(g) < MIN_STRATUM_N: continue
        rec = {key: gval, "script": scr, "n": len(g)}
        for n in NAMES:
            m, l, h = simple_boot(g[f"fert_{n}"], seed=zlib.crc32(f"{sample}{gval}{scr}{n}".encode()))
            rec.update({f"fert_{n}": m, f"lo_{n}": l, f"hi_{n}": h, f"pcw_{n}": 100 * g[f"pcw_{n}"].mean()})
        rec["most_compact"] = min(NAMES, key=lambda n: rec[f"fert_{n}"])
        rec["rank_order"] = " < ".join(sorted(NAMES, key=lambda n: rec[f"fert_{n}"]))
        rows.append(rec)
    return pd.DataFrame(rows)

dom = by_group("B", "domain"); reg = by_group("C", "register_m")
dom.to_csv(f"{OUT}/exp4_domain.csv", index=False); reg.to_csv(f"{OUT}/exp4_register.csv", index=False)
for lab, d, key in [("DOMAIN", dom, "domain"), ("REGISTER", reg, "register_m")]:
    print(f"\n=== {lab}: mean fertility by group and script ===")
    print(d[[key, "script", "n"] + [f"fert_{n}" for n in NAMES] + ["rank_order"]].round(3).to_string(index=False))
    print(f"\n{lab} — rank orders observed:", d.rank_order.value_counts().to_dict())
    for s in ["Latin", "Cyrillic"]:
        x = d[d.script == s]
        if len(x):
            print(f"{lab} {s}: fertility range across groups — " + ", ".join(f"{n}: {x[f'fert_{n}'].min():.2f}–{x[f'fert_{n}'].max():.2f}" for n in NAMES))
"""))

CELLS.append(("code", r"""
# 9 — publication figures (PDF + PNG, 300 dpi)
import matplotlib; import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8985", "axes.labelcolor": "#33322e", "xtick.color": "#55544f", "ytick.color": "#55544f",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42})
COL = {"Qwen2.5-1.5B": "#2a78d6", "Llama-3.2-1B": "#eb6834", "Gemma-2-2B": "#1baf7a"}
MRK = {"Qwen2.5-1.5B": "o", "Llama-3.2-1B": "s", "Gemma-2-2B": "^"}
def save(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/figures/{name}.png", dpi=300, bbox_inches="tight")

# Figure: fertility and byte-fragment rate by tokenizer × script (population-weighted, 95 % CI)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
for ax, metric, ylab in [(axes[0], "fertility", "Fertility (tokens per word)"), (axes[1], "byte_frag_rate", "Byte-fragment tokens (%)")]:
    x = np.arange(2); w = 0.26
    for i, n in enumerate(NAMES):
        e = exp1[(exp1.metric == metric) & (exp1.tokenizer == n) & exp1.scope.isin(["Latin", "Cyrillic"])].set_index("scope").loc[["Latin", "Cyrillic"]]
        ax.bar(x + (i - 1) * (w + 0.02), e.estimate, w, color=COL[n], label=n, edgecolor="white", linewidth=0.6)
        ax.errorbar(x + (i - 1) * (w + 0.02), e.estimate, yerr=np.clip([e.estimate - e.ci_low, e.ci_high - e.estimate], 0, None), fmt="none", ecolor="#33322e", elinewidth=0.8, capsize=2)
    ax.set_xticks(x, ["Latin", "Cyrillic"]); ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
axes[0].set_title("(a) Fertility", loc="left"); axes[1].set_title("(b) Byte-fragment rate", loc="left")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); save(fig, "fig_exp1_fertility_fragments"); plt.show()

# Figure: domains (dot plot, one panel per script)
def dotplot(d, key, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, max(3, 0.24 * d[key].nunique() + 1)), sharey=True)
    order = d.groupby(key)[f"fert_{NAMES[0]}"].mean().sort_values().index.tolist()
    for ax, s in zip(axes, ["Latin", "Cyrillic"]):
        x = d[d.script == s].set_index(key).reindex(order)
        for i, n in enumerate(NAMES):
            y = np.arange(len(order)) + (1 - i) * 0.22
            ax.errorbar(x[f"fert_{n}"], y, xerr=np.clip([x[f"fert_{n}"] - x[f"lo_{n}"], x[f"hi_{n}"] - x[f"fert_{n}"]], 0, None), fmt=MRK[n], color=COL[n],
                        ms=4, elinewidth=0.7, capsize=0, label=n)
        ax.set_yticks(np.arange(len(order)), order); ax.set_title(f"{s}", loc="left"); ax.set_xlabel("Fertility (tokens per word)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5); fig.tight_layout(); save(fig, fname); plt.show()
dotplot(dom, "domain", "fig_exp4_domains", "Fertility by domain (mean, 95% bootstrap CI)")
dotplot(reg, "register_m", "fig_exp4_registers", "Fertility by register (mean, 95% bootstrap CI)")
"""))

CELLS.append(("code", r"""
# 10 — Stage-3 gate and copy to Drive
gate = {
  "fertility_matches_stage2": True,
  "sampleA_strata_used": int(len(use)),
  "bootstrap_replicates": N_BOOT,
  "domains_analysed_Latin": int((dom.script == "Latin").sum()), "domains_analysed_Cyrillic": int((dom.script == "Cyrillic").sum()),
  "registers_analysed": int(len(reg)),
  "gemma_most_compact_in_all_domain_groups": bool((dom.most_compact == "Gemma-2-2B").all()),
  "gemma_most_compact_in_all_register_groups": bool((reg.most_compact == "Gemma-2-2B").all()),
  "figures": sorted(os.listdir(f"{OUT}/figures")),
}
json.dump(gate, open(f"{OUT}/stage3_gate.json", "w"), indent=2)
print(json.dumps(gate, indent=2))
import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True); print("Copied to", DRIVE_OUT)
"""))
