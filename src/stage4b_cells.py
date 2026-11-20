# Stage 4B — Experiment 2a (controlled script comparison via transliteration) and 2b (apostrophe variants)
import stage4_cells as _s4
TRANSLIT_SRC = _s4.CELLS[2][1].strip().replace("# 2 — transliterator (both directions) and unit tests", "# 3 — transliterator v4 (identical to Stage 4A) + unit tests")
TRANSLIT_SRC += '\nassert not fails, "transliterator unit tests failed — do not continue"'

CELLS = []
CELLS.append(("md", r"""
# Stage 4B — Experiment 2a (same text, two scripts) and Experiment 2b (apostrophe variants)
**2a.** Every sample-A document is measured twice with *the same words*: once in its original script and once transliterated (validated transliterator v4, Stage 4A).
* Cyrillic documents → Latin is the primary direction; Latin documents → Cyrillic is the robustness direction.
* Only the script changes: word count is identical by construction.
* Words the transliterator cannot handle (щ/ы, c/w, mixed-script) are dropped from **both** conditions.
* Words already in the other script are kept unchanged in both conditions.

**2b.** The Latin text is rendered with five spellings of the oʻ/gʻ mark and the tutuq belgisi: ʻ ʼ (official), ', ‘ ’, ’, and `. All other characters stay fixed.

Measurement unit, as in Stages 2–3: each word tokenized in isolation with a leading space.
* fertility = tokens / words;
* CPT = characters / token;
* byte-fragment rate = fragment tokens / tokens.

Estimates are population-weighted over the sample-A strata (≥ 30 docs) with stratified-bootstrap 95 % CIs. Formal tests come in Stage 6.
"""))

CELLS.append(("code", r"""
# 1 — setup and inputs
import os, re, json, time, datetime, unicodedata, zlib
import numpy as np, pandas as pd
import importlib.metadata as im
from collections import Counter
SEED = 42; N_BOOT = 1000; MIN_STRATUM_N = 30
MODELS = {
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B",       "8faed761d45a263340a0528343f099c05c9a4323"),
    "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    "Gemma-2-2B":   ("google/gemma-2-2b",       "c5ebcd40d208330abc697524c919956e692655cf"),
}
NAMES = list(MODELS)
from google.colab import drive
drive.mount("/content/drive")
S2 = "/content/drive/MyDrive/uz_tokenization/stage2"; S3 = "/content/drive/MyDrive/uz_tokenization/stage3"
OUT = "/content/uz_tok/stage4b"; os.makedirs(f"{OUT}/figures", exist_ok=True)
DRIVE_OUT = "/content/drive/MyDrive/uz_tokenization/stage4b"; os.makedirs(DRIVE_OUT, exist_ok=True)

mem = pd.read_csv(f"{S2}/sample_membership_final.csv")
plan = pd.read_csv(f"{S2}/sampling_plan.csv")
docs = pd.read_parquet(f"{S2}/doc_metrics.parquet", columns=["gidx", "script", "source", "format_origin"])
texts = pd.read_parquet(f"{S2}/sample_clean.parquet", columns=["gidx", "text"]).set_index("gidx").text
A = mem[mem.kept & (mem["sample"] == "A")].merge(docs, on="gidx")
sizes = A.groupby("stratum").size(); use = sizes[sizes >= MIN_STRATUM_N].index
A = A[A.stratum.isin(use)].reset_index(drop=True)
pop = plan[plan["sample"] == "A"].set_index("stratum")["available"]
PKG = {p: im.version(p) for p in ["transformers", "tokenizers", "huggingface_hub", "pandas", "numpy", "matplotlib"]}
print(f"Sample A: {len(A):,} docs in {len(use)} strata | Latin {int((A.script=='Latin').sum()):,} | Cyrillic {int((A.script=='Cyrillic').sum()):,}")
print(PKG)
"""))

CELLS.append(("code", r"""
# 2 — tokenizers (pinned) and byte-fragment flags (same definition as Stage 3)
from huggingface_hub import login
try:
    from google.colab import userdata; HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    import getpass; HF_TOKEN = getpass.getpass("Hugging Face token (hf_...): ")
login(token=HF_TOKEN)
from transformers import AutoTokenizer
TOK = {n: AutoTokenizer.from_pretrained(r, revision=s, token=HF_TOKEN, use_fast=True) for n, (r, s) in MODELS.items()}
BYTE_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
FRAG = {}
for n, tok in TOK.items():
    V = len(tok)
    pieces = tok.convert_ids_to_tokens(list(range(V)))
    dec = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(V)]
    FRAG[n] = np.array([("�" in d) or bool(BYTE_TOKEN_RE.match(p or "")) for d, p in zip(dec, pieces)])
    print(f"{n}: {FRAG[n].sum():,} of {V:,} vocabulary entries are byte fragments")
"""))

CELLS.append(("code", TRANSLIT_SRC))

CELLS.append(("code", r"""
# 4 — build the paired conditions for every document (word lists), and the observed apostrophe usage
WORD_RE = re.compile(rf"[^\W\d_]+(?:[{APOS_ANY}][^\W\d_]+)*", re.UNICODE)
VARIANTS = {"ʻ ʼ (official U+02BB/U+02BC)": (OKINA, TUTUQ), "' (ASCII U+0027)": ("'", "'"),
            "‘ ’ (U+2018/U+2019)": ("‘", "’"), "’ (U+2019 for both)": ("’", "’"), "` (backtick U+0060)": ("`", "`")}
def apos_variant(w, okina, tutuq): return w.replace(OKINA, okina).replace(TUTUQ, tutuq)

COND = {}            # gidx -> {condition: [words]}
excl = Counter(); apos_seen = Counter()
for r in A.itertuples():
    words = WORD_RE.findall(texts[r.gidx])
    if r.script == "Cyrillic":
        keep = [w for w in words if cyr_transliterable(w) or script_of(w) == "Latin"]
        lat = [cyr2lat(w) if cyr_transliterable(w) else w for w in keep]
        COND[r.gidx] = {"Cyrillic": keep, "Latin": lat}
    else:
        for w in words:                                   # observed spelling of the marks (raw text, before normalisation)
            for m in re.finditer(rf"([oOgG]?)([{APOS_ANY}])", w):
                apos_seen[("oʻ/gʻ mark" if m.group(1) else "tutuq") + " = U+%04X" % ord(m.group(2))] += 1
        keep = [norm_lat_apos(w) for w in words if lat_transliterable(w) or script_of(w) == "Cyrillic"]
        cyr = [lat2cyr(w) if script_of(w) == "Latin" else w for w in keep]
        COND[r.gidx] = {"Cyrillic": cyr, "Latin": keep}
    excl[r.script, "kept"] += len(keep); excl[r.script, "all"] += len(words)
    lat_words = COND[r.gidx]["Latin"]
    for v, (o, t) in VARIANTS.items():
        if v.startswith("ʻ"): continue
        COND[r.gidx]["Latin " + v] = [apos_variant(w, o, t) for w in lat_words]
    assert len({len(x) for x in COND[r.gidx].values()}) == 1          # same number of words in every condition

for s in ["Cyrillic", "Latin"]:
    print(f"{s} documents: {excl[s,'kept']:,} of {excl[s,'all']:,} word tokens kept ({100*excl[s,'kept']/excl[s,'all']:.2f}%)")
tot = sum(c for k, c in apos_seen.items() if k.startswith("oʻ")); tot_t = sum(c for k, c in apos_seen.items() if k.startswith("tutuq"))
obs = pd.DataFrame([{"position": k.split(" = ")[0], "codepoint": k.split(" = ")[1], "char": chr(int(k.split("U+")[1], 16)), "count": c,
                     "share_%": round(100 * c / (tot if k.startswith("oʻ") else tot_t), 2)} for k, c in apos_seen.most_common()])
obs.to_csv(f"{OUT}/exp2b_observed_apostrophes.csv", index=False)
print("\nObserved spelling of the marks in the Latin sample-A documents (raw text):"); print(obs.to_string(index=False))
"""))

CELLS.append(("code", r"""
# 5 — tokenize every distinct string once per tokenizer, then per-document metrics for every condition
all_strings = sorted({w for d in COND.values() for ws in d.values() for w in ws})
print(f"distinct strings to tokenize: {len(all_strings):,}")
STAT = {}
t0 = time.time()
for n, tok in TOK.items():
    F = FRAG[n]; st = {}
    for i in range(0, len(all_strings), 2048):
        chunk = all_strings[i:i + 2048]
        for w, ids in zip(chunk, tok([" " + w for w in chunk], add_special_tokens=False)["input_ids"]):
            st[w] = (len(ids), int(F[ids].sum()))
    STAT[n] = st; print(f"{n} done ({time.time()-t0:.0f}s)")

rows = []
meta = A.set_index("gidx")[["stratum", "script", "source", "format_origin"]]
for g, d in COND.items():
    for cond, ws in d.items():
        N = len(ws); C = sum(len(w) for w in ws)
        rec = {"gidx": g, "condition": cond, "n_words": N, "n_chars": C}
        for n in NAMES:
            T = sum(STAT[n][w][0] for w in ws); Fr = sum(STAT[n][w][1] for w in ws)
            rec.update({f"fert_{n}": T / N, f"cpt_{n}": C / T, f"frag_{n}": Fr / T, f"tok_{n}": T})
        rows.append(rec)
M = pd.DataFrame(rows).merge(meta, left_on="gidx", right_index=True)
assert np.isfinite(M[[c for c in M if c.startswith(("fert_", "cpt_", "frag_"))]].to_numpy(float)).all()
M.to_parquet(f"{OUT}/exp2_doc_condition_metrics.parquet")
print(M.groupby(["script", "condition"]).size().unstack(0))
"""))

CELLS.append(("code", r"""
# 6 — Experiment 2a: same documents, original vs transliterated script (population-weighted, 95 % CI)
def boot_weighted(df, cols, n_boot=N_BOOT, seed=SEED):
    groups = [(pop[s], g[cols].to_numpy(float)) for s, g in df.groupby("stratum")]
    W = np.array([w for w, _ in groups], float); W /= W.sum()
    point = sum(w * x.mean(0) for w, (_, x) in zip(W, groups))
    rng = np.random.default_rng(seed); reps = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        reps[b] = sum(w * x[rng.integers(0, len(x), len(x))].mean(0) for w, (_, x) in zip(W, groups))
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0); return point, lo, hi

def fmt(p, l, h, d=3): return f"{p:.{d}f} [{l:.{d}f}, {h:.{d}f}]"
rows = []
for direction, s in [("Cyrillic docs → Latin (primary)", "Cyrillic"), ("Latin docs → Cyrillic (robustness)", "Latin")]:
    sub = M[M.script == s]
    W = sub.pivot(index="gidx", columns="condition")
    P = pd.DataFrame({"stratum": sub.drop_duplicates("gidx").set_index("gidx").stratum})
    P["cpw_cyr"] = W["n_chars"]["Cyrillic"] / W["n_words"]["Cyrillic"]; P["cpw_lat"] = W["n_chars"]["Latin"] / W["n_words"]["Latin"]
    for n in NAMES:
        P[f"fc_{n}"] = W[f"fert_{n}"]["Cyrillic"]; P[f"fl_{n}"] = W[f"fert_{n}"]["Latin"]
        P[f"r_{n}"] = P[f"fc_{n}"] / P[f"fl_{n}"]
        P[f"cc_{n}"] = W[f"cpt_{n}"]["Cyrillic"]; P[f"cl_{n}"] = W[f"cpt_{n}"]["Latin"]
        P[f"gc_{n}"] = 100 * W[f"frag_{n}"]["Cyrillic"]; P[f"gl_{n}"] = 100 * W[f"frag_{n}"]["Latin"]
    pc, lc, hc = boot_weighted(P, ["cpw_cyr", "cpw_lat"])
    print(f"\n{direction}: {len(P):,} docs | characters per word — Cyrillic {fmt(pc[0], lc[0], hc[0])}, Latin {fmt(pc[1], lc[1], hc[1])}")
    for n in NAMES:
        cols = [f"fc_{n}", f"fl_{n}", f"r_{n}", f"cc_{n}", f"cl_{n}", f"gc_{n}", f"gl_{n}"]
        p, l, h = boot_weighted(P, cols)
        rows.append({"direction": direction, "tokenizer": n, "n_docs": len(P),
                     **{k: v for c, k in zip(range(7), ["fert_cyr", "fert_lat", "ratio_cyr_over_lat", "cpt_cyr", "cpt_lat", "frag_cyr_%", "frag_lat_%"])
                        for v in [p[c]]},
                     **{k + "_lo": l[c] for c, k in zip(range(7), ["fert_cyr", "fert_lat", "ratio_cyr_over_lat", "cpt_cyr", "cpt_lat", "frag_cyr_%", "frag_lat_%"])},
                     **{k + "_hi": h[c] for c, k in zip(range(7), ["fert_cyr", "fert_lat", "ratio_cyr_over_lat", "cpt_cyr", "cpt_lat", "frag_cyr_%", "frag_lat_%"])}})
exp2a = pd.DataFrame(rows); exp2a.to_csv(f"{OUT}/exp2a_controlled_script.csv", index=False)
show = exp2a.copy()
for k in ["fert_cyr", "fert_lat", "ratio_cyr_over_lat", "cpt_cyr", "cpt_lat", "frag_cyr_%", "frag_lat_%"]:
    show[k] = [fmt(a, b, c, 3 if not k.startswith("frag") else 2) for a, b, c in zip(show[k], show[k + "_lo"], show[k + "_hi"])]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print(show[["direction", "tokenizer", "fert_cyr", "fert_lat", "ratio_cyr_over_lat"]].to_string(index=False))
print(show[["direction", "tokenizer", "cpt_cyr", "cpt_lat", "frag_cyr_%", "frag_lat_%"]].to_string(index=False))

# observational ratio from Stage 3 (different documents per script) for comparison
try:
    e3 = pd.read_csv(f"{S3}/exp1_estimates.csv"); f3 = e3[e3.metric == "fertility"].set_index(["scope", "tokenizer"]).estimate
    print("\nObservational Cyrillic/Latin ratio (Stage 3, different documents): " +
          ", ".join(f"{n}: {f3['Cyrillic', n] / f3['Latin', n]:.3f}" for n in NAMES))
except Exception as ex:
    print("Stage-3 estimates not found:", ex)
"""))

CELLS.append(("code", r"""
# 7 — Experiment 2b: apostrophe variants in Latin text (both document groups: original Latin, and Latin transliterated from Cyrillic)
VNAMES = list(VARIANTS); OFF = VNAMES[0]
cond_of = {v: ("Latin" if v == OFF else "Latin " + v) for v in VNAMES}
rows = []
for grp, s in [("Latin docs (original text, marks normalised)", "Latin"), ("Cyrillic docs transliterated to Latin", "Cyrillic")]:
    sub = M[M.script == s]
    W = sub.pivot(index="gidx", columns="condition")
    P = pd.DataFrame({"stratum": sub.drop_duplicates("gidx").set_index("gidx").stratum})
    for n in NAMES:
        for j, v in enumerate(VNAMES):
            P[f"f{j}_{n}"] = W[f"fert_{n}"][cond_of[v]]
            P[f"d{j}_{n}"] = 100 * (W[f"fert_{n}"][cond_of[v]] / W[f"fert_{n}"][cond_of[OFF]] - 1)
    for n in NAMES:
        cols = [f"f{j}_{n}" for j in range(len(VNAMES))] + [f"d{j}_{n}" for j in range(len(VNAMES))]
        p, l, h = boot_weighted(P, cols)
        for j, v in enumerate(VNAMES):
            k = len(VNAMES) + j
            rows.append({"documents": grp, "tokenizer": n, "variant": v, "fertility": p[j], "fert_lo": l[j], "fert_hi": h[j],
                         "change_vs_official_%": p[k], "chg_lo": l[k], "chg_hi": h[k]})
exp2b = pd.DataFrame(rows); exp2b.to_csv(f"{OUT}/exp2b_apostrophe_variants.csv", index=False)
for grp in exp2b.documents.unique():
    t = exp2b[exp2b.documents == grp].assign(val=lambda d: [f"{a:+.2f} [{b:+.2f}, {c:+.2f}]" for a, b, c in zip(d["change_vs_official_%"], d.chg_lo, d.chg_hi)])
    print(f"\n=== {grp}: % change in fertility vs official ʻ ʼ ===")
    print(t.pivot(index="variant", columns="tokenizer", values="val").reindex(VNAMES)[NAMES].to_string())

# tokens per marked word (pooled over all sample-A Latin-side words that contain a mark)
marked = Counter(w for d in COND.values() for w in d["Latin"] if OKINA in w or TUTUQ in w)
allw = sum(len(d["Latin"]) for d in COND.values())
print(f"\nWords containing oʻ/gʻ or tutuq: {sum(marked.values()):,} of {allw:,} Latin-side word tokens ({100*sum(marked.values())/allw:.1f}%)")
mt = []
for v, (o, t) in VARIANTS.items():
    rec = {"variant": v}
    for n in NAMES:
        rec[n] = sum(STAT[n][apos_variant(w, o, t)][0] * c for w, c in marked.items()) / sum(marked.values())
    mt.append(rec)
mt = pd.DataFrame(mt).set_index("variant").round(3); mt.to_csv(f"{OUT}/exp2b_tokens_per_marked_word.csv")
print("Mean tokens per marked word, by spelling of the mark:"); print(mt.to_string())
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

# Fig: Cyrillic/Latin fertility ratio — observational (Stage 3) vs controlled (2a, both directions)
cats = ["Observational\n(Stage 3, different docs)", "Controlled\nCyrillic docs → Latin", "Controlled\nLatin docs → Cyrillic"]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for i, n in enumerate(NAMES):
    x = np.arange(3) + (i - 1) * 0.18
    try: obs_r = f3["Cyrillic", n] / f3["Latin", n]
    except Exception: obs_r = np.nan
    e = exp2a[exp2a.tokenizer == n].set_index("direction")
    y = [obs_r] + [e.loc[d, "ratio_cyr_over_lat"] for d in e.index]
    lo = [0] + [e.loc[d, "ratio_cyr_over_lat"] - e.loc[d, "ratio_cyr_over_lat_lo"] for d in e.index]
    hi = [0] + [e.loc[d, "ratio_cyr_over_lat_hi"] - e.loc[d, "ratio_cyr_over_lat"] for d in e.index]
    ax.errorbar(x, y, yerr=np.clip([lo, hi], 0, None), fmt=MRK[n], color=COL[n], ms=5, capsize=2, elinewidth=0.8, label=n)
    for xi, yi in zip(x, y):
        if np.isfinite(yi): ax.text(xi + 0.04, yi, f"{yi:.2f}", fontsize=6.5, va="center", color="#33322e")
ax.axhline(1, color="#8a8985", lw=0.8, ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels(cats, fontsize=8)
ax.set_ylabel("Cyrillic / Latin fertility"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout(); save(fig, "fig_exp2a_script_ratio"); plt.show()

# Fig: apostrophe variants — % change in fertility vs official, Latin documents
t = exp2b[exp2b.documents.str.startswith("Latin docs")]
fig, ax = plt.subplots(figsize=(6.2, 2.9))
for i, n in enumerate(NAMES):
    e = t[t.tokenizer == n].set_index("variant").reindex(VNAMES); x = np.arange(len(VNAMES)) + (i - 1) * 0.18
    ax.errorbar(x, e["change_vs_official_%"], yerr=np.clip([e["change_vs_official_%"] - e.chg_lo, e.chg_hi - e["change_vs_official_%"]], 0, None),
                fmt=MRK[n], color=COL[n], ms=5, capsize=2, elinewidth=0.8, label=n)
ax.axhline(0, color="#8a8985", lw=0.8, ls="--"); ax.set_xticks(range(len(VNAMES)))
ax.set_xticklabels([v.split(" (")[0] + "\n(" + v.split(" (")[1] for v in VNAMES], fontsize=7.5)
ax.set_ylabel("Change in fertility vs ʻ ʼ (%)"); ax.grid(axis="x", visible=False)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
fig.tight_layout(); save(fig, "fig_exp2b_apostrophes"); plt.show()
print("Save errors:", SAVE_ERRORS or "none")
"""))

CELLS.append(("code", r"""
# 9 — gate and copy to Drive
gate = {
  "transliterator_unit_test_failures": len(fails),
  "docs_cyrillic": int((A.script == "Cyrillic").sum()), "docs_latin": int((A.script == "Latin").sum()),
  "conditions_per_doc": int(M.groupby("gidx").condition.nunique().min()),
  "equal_word_counts_across_conditions": bool((M.groupby("gidx").n_words.nunique() == 1).all()),
  "all_metrics_finite": True,
  "ratio_cyr_over_lat_controlled": {f"{r.direction} | {r.tokenizer}": round(r.ratio_cyr_over_lat, 4) for r in exp2a.itertuples()},
  "figures": sorted(os.listdir(f"{OUT}/figures")),
  "packages": PKG, "seed": SEED, "n_boot": N_BOOT,
  "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
json.dump(gate, open(f"{OUT}/stage4b_gate.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(gate, indent=2, ensure_ascii=False))
import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True); print("Copied to", DRIVE_OUT, "→", sorted(os.listdir(DRIVE_OUT)))
"""))
