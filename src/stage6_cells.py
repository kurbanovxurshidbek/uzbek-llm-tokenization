# Stage 6 — inferential statistics, effect sizes, multiple-comparison control (RQ1–RQ5)
CELLS = []
CELLS.append(("md", r"""
# Stage 6 — Statistics: tests, effect sizes, Holm correction, confound-controlled models
Inputs are Drive outputs of Stages 2–5; nothing is re-tokenized.

| RQ | Analysis |
|---|---|
| RQ1 / RQ5 | Paired per-document tokenizer comparisons (Wilcoxon signed-rank, d_z, rank-biserial r, share of documents favouring each tokenizer); Friedman test with Kendall's W |
| RQ2 | (a) Experiment 2a: controlled log Cyrillic/Latin ratio per document, its tests, and between-tokenizer differences in the script penalty. (b) Observational script effect after controlling for source, format and characters per word (OLS, HC3). (c) Experiment 2b: effect of the apostrophe spelling, and the Qwen/Llama order flip |
| RQ3 | Word-level OLS: tokens ~ morphemes + characters (HC3); partial Spearman; F1 minus chance with a bootstrap test |
| RQ4 | Kruskal–Wallis with ε² across domains and registers; partial η² of domain/register before and after controlling characters per word and source |

**Testing conventions:**
* Documents (or words) are the units. Tests are unweighted; the population-weighted point estimates come from Stages 3–4.
* With thousands of documents almost every p-value is tiny, so **effect sizes are the primary result**.
* Holm correction is applied within each RQ family.
"""))

CELLS.append(("code", r"""
# 1 — setup and inputs
import os, json, datetime, warnings
import numpy as np, pandas as pd
import importlib.metadata as im
from scipy import stats
try:
    import statsmodels.formula.api as smf; import statsmodels.api as sm
except ImportError:
    import subprocess, sys; subprocess.run([sys.executable, "-m", "pip", "-q", "install", "statsmodels"], check=True)
    import statsmodels.formula.api as smf; import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore", category=RuntimeWarning)
SEED = 42; N_BOOT = 2000; MIN_STRATUM_N = 30
NAMES = ["Qwen2.5-1.5B", "Llama-3.2-1B", "Gemma-2-2B"]
from google.colab import drive
drive.mount("/content/drive")
R = "/content/drive/MyDrive/uz_tokenization"
S2, S4B, S5C = f"{R}/stage2", f"{R}/stage4b", f"{R}/stage5c"
OUT = "/content/uz_tok/stage6"; os.makedirs(OUT, exist_ok=True)
DRIVE_OUT = f"{R}/stage6"; os.makedirs(DRIVE_OUT, exist_ok=True)

mem = pd.read_csv(f"{S2}/sample_membership_final.csv")
docs = pd.read_parquet(f"{S2}/doc_metrics.parquet")
docs["cpw"] = (docs.n_chars_words - (docs.n_words - 1)) / docs.n_words          # letters per word (spaces removed)
kept = mem[mem.kept].merge(docs, on="gidx")
sizes = kept[kept["sample"] == "A"].groupby("stratum").size(); useA = sizes[sizes >= MIN_STRATUM_N].index
A = kept[(kept["sample"] == "A") & kept.stratum.isin(useA)].copy()
SB = kept[kept["sample"] == "B"].copy(); SC = kept[kept["sample"] == "C"].copy()
M2 = pd.read_parquet(f"{S4B}/exp2_doc_condition_metrics.parquet")
W3 = pd.read_json(f"{S5C}/exp3_word_level.json", orient="records")
PKG = {p: im.version(p) for p in ["pandas", "numpy", "scipy", "statsmodels"]}
print(f"A {len(A):,} docs | B {len(SB):,} | C {len(SC):,} | exp2 rows {len(M2):,} | exp3 words {len(W3):,}"); print(PKG)
"""))

CELLS.append(("code", r"""
# 2 — helpers: paired effect sizes, Holm, formatting
def paired(x, y):
    # x, y: paired arrays (same units). Returns effect sizes for x − y and a two-sided Wilcoxon signed-rank test.
    x, y = np.asarray(x, float), np.asarray(y, float); d = x - y; d = d[np.isfinite(d)]; n = len(d)
    md = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    nz = d[d != 0]
    if len(nz) == 0: p, r_rb = 1.0, 0.0
    else:
        ranks = stats.rankdata(np.abs(nz)); wpos, wneg = ranks[nz > 0].sum(), ranks[nz < 0].sum()
        r_rb = (wpos - wneg) / (wpos + wneg)                                  # matched-pairs rank-biserial correlation
        p = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided", method="approx").pvalue
    return {"n": n, "mean_diff": md, "ci_lo": md - 1.96 * se, "ci_hi": md + 1.96 * se,
            "d_z": md / sd if sd > 0 else np.nan, "rank_biserial": r_rb, "share_x_lower_%": 100 * (d < 0).mean(), "p": p}

def one_sample(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    return paired(d, np.zeros_like(d))

def holm(df, col="p", fam="family"):
    df = df.copy(); df["p_holm"] = np.nan
    for f, g in df.groupby(fam):
        ok = g[col].notna()
        if ok.any(): df.loc[g.index[ok], "p_holm"] = multipletests(g.loc[ok, col], method="holm")[1]
    return df

def fp(p): return "<1e-300" if p == 0 else f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"
RESULTS = []      # rows: family, analysis, comparison, effect, effect_lo, effect_hi, effect_unit, d_z, r_rb, extra, p
def add(fam, analysis, comp, eff, lo, hi, unit, dz=np.nan, rrb=np.nan, extra="", p=np.nan):
    RESULTS.append(dict(family=fam, analysis=analysis, comparison=comp, effect=eff, effect_lo=lo, effect_hi=hi, unit=unit,
                        d_z=dz, rank_biserial=rrb, extra=extra, p=p))
print("helpers ready")
"""))

CELLS.append(("code", r"""
# 3 — RQ1 / RQ5: paired tokenizer comparisons on sample A (fertility and PCW), Friedman + Kendall's W
PAIRS = [("Gemma-2-2B", "Qwen2.5-1.5B"), ("Gemma-2-2B", "Llama-3.2-1B"), ("Llama-3.2-1B", "Qwen2.5-1.5B")]
for s in ["Latin", "Cyrillic"]:
    sub = A[A.script == s]
    for metric, pre, unit in [("fertility", "fert", "tokens/word"), ("PCW", "pcw", "pp")]:
        scale = 100 if pre == "pcw" else 1
        for a, b in PAIRS:
            r = paired(scale * sub[f"{pre}_{a}"], scale * sub[f"{pre}_{b}"])
            rel = 100 * (sub[f"fert_{a}"] / sub[f"fert_{b}"] - 1).mean() if pre == "fert" else np.nan
            add("RQ1/RQ5", f"{metric} | {s}", f"{a} − {b}", r["mean_diff"], r["ci_lo"], r["ci_hi"], unit, r["d_z"], r["rank_biserial"],
                f"n={r['n']}; {a} lower in {r['share_x_lower_%']:.1f}% of docs" + (f"; mean relative diff {rel:+.2f}%" if pre == "fert" else ""), r["p"])
    X = sub[[f"fert_{n}" for n in NAMES]].to_numpy()
    fr = stats.friedmanchisquare(*X.T); Wk = fr.statistic / (X.shape[0] * (X.shape[1] - 1))
    mean_ranks = dict(zip(NAMES, stats.rankdata(X, axis=1).mean(0).round(3)))
    add("RQ1/RQ5", f"fertility | {s}", "Friedman (3 tokenizers)", Wk, np.nan, np.nan, "Kendall's W", extra=f"chi2={fr.statistic:.1f}; mean ranks {mean_ranks}", p=fr.pvalue)
t = pd.DataFrame([r for r in RESULTS if r["family"] == "RQ1/RQ5"])
print(t[["analysis", "comparison", "effect", "effect_lo", "effect_hi", "unit", "d_z", "rank_biserial", "p", "extra"]].round(4).to_string(index=False))
"""))

CELLS.append(("code", r"""
# 4 — RQ2: (a) controlled script effect (Exp 2a), (b) observational effect after controls, (c) apostrophe spelling (Exp 2b)
W2 = M2.pivot_table(index=["gidx", "script", "stratum", "source", "format_origin"], columns="condition",
                    values=[f"fert_{n}" for n in NAMES]).reset_index()
OFF, ASC = "Latin", "Latin ' (ASCII U+0027)"
for s, label in [("Cyrillic", "Cyrillic docs → Latin"), ("Latin", "Latin docs → Cyrillic")]:
    sub = W2[W2.script == s]
    logr = {n: np.log(sub[(f"fert_{n}", "Cyrillic")].values / sub[(f"fert_{n}", OFF)].values) for n in NAMES}
    for n in NAMES:
        r = one_sample(logr[n])
        add("RQ2", f"2a controlled | {label}", f"{n}: log(Cyr/Lat)", np.exp(r["mean_diff"]), np.exp(r["ci_lo"]), np.exp(r["ci_hi"]), "geometric-mean ratio",
            r["d_z"], r["rank_biserial"], f"n={r['n']}; Cyrillic higher in {100 - r['share_x_lower_%']:.1f}% of docs", r["p"])
    for a, b in PAIRS:
        r = paired(logr[a], logr[b])
        add("RQ2", f"2a script penalty difference | {label}", f"{a} − {b}", 100 * (np.exp(r["mean_diff"]) - 1), 100 * (np.exp(r["ci_lo"]) - 1),
            100 * (np.exp(r["ci_hi"]) - 1), "% difference in Cyr/Lat ratio", r["d_z"], r["rank_biserial"], f"n={r['n']}", r["p"])

# (b) observational script effect on sample A, controlling source, format and letters per word (OLS on log fertility, HC3)
obs_rows = []
for n in NAMES:
    for formula, label in [(f"np.log(Q('fert_{n}')) ~ C(script, Treatment('Latin'))", "script only"),
                           (f"np.log(Q('fert_{n}')) ~ C(script, Treatment('Latin')) + C(source) + C(format_origin)", "+ source + format"),
                           (f"np.log(Q('fert_{n}')) ~ C(script, Treatment('Latin')) + C(source) + C(format_origin) + np.log(cpw)", "+ source + format + log letters/word")]:
        m = smf.ols(formula, data=A).fit(cov_type="HC3")
        k = [c for c in m.params.index if c.startswith("C(script")][0]           # = log(Cyrillic / Latin)
        ratio = np.exp(m.params[k]); lo, hi = np.exp(m.conf_int().loc[k])
        obs_rows.append({"tokenizer": n, "model": label, "Cyr_over_Lat": ratio, "lo": lo, "hi": hi, "adj_R2": m.rsquared_adj, "p": m.pvalues[k]})
        if label.startswith("+ source + format + log"):
            add("RQ2", "2 observational (sample A), controls: source, format, log letters/word", f"{n}: Cyrillic vs Latin", ratio, lo, hi,
                "ratio of fertility", extra=f"OLS HC3; adj R2={m.rsquared_adj:.3f}", p=m.pvalues[k])
obs = pd.DataFrame(obs_rows); obs.to_csv(f"{OUT}/rq2_observational_models.csv", index=False)
print("Observational Cyrillic/Latin fertility ratio, sample A (OLS on log fertility, HC3):"); print(obs.round(4).to_string(index=False))

# (c) apostrophe spelling (Exp 2b): official vs ASCII, per tokenizer, Latin documents (original text)
lat = W2[W2.script == "Latin"]
for n in NAMES:
    r = paired(100 * np.log(lat[(f"fert_{n}", ASC)] / lat[(f"fert_{n}", OFF)]), np.zeros(len(lat)))
    add("RQ2", "2b apostrophe | Latin docs", f"{n}: ASCII ' vs official ʻ ʼ", 100 * (np.exp(r["mean_diff"] / 100) - 1), 100 * (np.exp(r["ci_lo"] / 100) - 1),
        100 * (np.exp(r["ci_hi"] / 100) - 1), "% change in fertility", r["d_z"], r["rank_biserial"], f"n={r['n']}", r["p"])
for cond, lab in [(OFF, "official ʻ ʼ"), (ASC, "ASCII '")]:
    r = paired(lat[("fert_Llama-3.2-1B", cond)], lat[("fert_Qwen2.5-1.5B", cond)])
    add("RQ2", f"2b Qwen/Llama order | Latin docs, {lab}", "Llama − Qwen", r["mean_diff"], r["ci_lo"], r["ci_hi"], "tokens/word", r["d_z"], r["rank_biserial"],
        f"n={r['n']}; Llama lower in {r['share_x_lower_%']:.1f}% of docs", r["p"])
t = pd.DataFrame([r for r in RESULTS if r["family"] == "RQ2"])
print(); print(t[["analysis", "comparison", "effect", "effect_lo", "effect_hi", "unit", "d_z", "rank_biserial", "extra"]].round(4).to_string(index=False))
"""))

CELLS.append(("code", r"""
# 5 — RQ3: word-level models (primary set: native + loan) and F1 above chance (bootstrap)
P3 = W3[W3.word_class.isin(["native", "loan"])].copy()
rows = []
for s in ["Latin", "Cyrillic"]:
    sub = P3[P3.script == s]
    for n in NAMES:
        m0 = smf.ols(f"Q('ntok_{n}') ~ n_morphs", data=sub).fit(cov_type="HC3")
        m1 = smf.ols(f"Q('ntok_{n}') ~ n_morphs + L", data=sub).fit(cov_type="HC3")
        # partial Spearman: correlation of rank residuals after removing word length
        rk = sub[[f"ntok_{n}", "n_morphs", "L"]].rank()
        res = lambda y: sm.OLS(rk[y], sm.add_constant(rk["L"])).fit().resid
        prs = stats.pearsonr(res(f"ntok_{n}"), res("n_morphs"))
        rows.append({"script": s, "tokenizer": n, "b_morph_alone": m0.params["n_morphs"], "b_morph_ctrl_len": m1.params["n_morphs"],
                     "b_morph_lo": m1.conf_int().loc["n_morphs", 0], "b_morph_hi": m1.conf_int().loc["n_morphs", 1], "p_morph": m1.pvalues["n_morphs"],
                     "b_len_per_char": m1.params["L"], "adj_R2_len_only": smf.ols(f"Q('ntok_{n}') ~ L", data=sub).fit().rsquared_adj,
                     "adj_R2_len_morph": m1.rsquared_adj, "partial_spearman": prs.statistic, "partial_spearman_p": prs.pvalue})
        add("RQ3", f"tokens ~ morphemes + letters | {s}", f"{n}: extra tokens per morpheme (length held fixed)", m1.params["n_morphs"],
            m1.conf_int().loc["n_morphs", 0], m1.conf_int().loc["n_morphs", 1], "tokens per morpheme", extra=f"partial Spearman={prs.statistic:.3f}; n={len(sub)}",
            p=m1.pvalues["n_morphs"])
rq3m = pd.DataFrame(rows); rq3m.to_csv(f"{OUT}/rq3_word_models.csv", index=False)
print(rq3m.round(4).to_string(index=False))

def f1_and_chance(df, n):
    tb = df[f"tok_b_{n}"].apply(set); gb = df.gold_b.apply(set)
    tp = np.array([len(a & b) for a, b in zip(tb, gb)], float); nt = tb.apply(len).to_numpy(float); ng = gb.apply(len).to_numpy(float)
    slots = np.maximum(df.L.to_numpy() - 1, 1).astype(float); ex = nt * ng / slots
    def f(tp_, nt_, ng_):
        P, Rr = tp_.sum() / nt_.sum(), tp_.sum() / ng_.sum(); return 200 * P * Rr / (P + Rr)
    return tp, nt, ng, ex, f
rng = np.random.default_rng(SEED)
for s in ["Latin", "Cyrillic"]:
    sub = P3[P3.script == s].reset_index(drop=True)
    for n in NAMES:
        tp, nt, ng, ex, f = f1_and_chance(sub, n)
        d0 = f(tp, nt, ng) - f(ex, nt, ng)
        reps = np.array([f(tp[ix], nt[ix], ng[ix]) - f(ex[ix], nt[ix], ng[ix]) for ix in (rng.integers(0, len(sub), len(sub)) for _ in range(N_BOOT))])
        p_boot = min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())) if (reps <= 0).any() else 1 / N_BOOT
        add("RQ3", f"boundary F1 above chance | {s}", f"{n}: F1 − chance F1", d0, *np.percentile(reps, [2.5, 97.5]), "pp",
            extra=f"bootstrap over words (B={N_BOOT}); p reported as ≤1/B when no replicate ≤0", p=p_boot)
t = pd.DataFrame([r for r in RESULTS if r["family"] == "RQ3"])
print(); print(t[["analysis", "comparison", "effect", "effect_lo", "effect_hi", "unit", "extra"]].round(4).to_string(index=False))
"""))

CELLS.append(("code", r"""
# 6 — RQ4: domain (sample B) and register (sample C): Kruskal–Wallis ε², and partial η² before/after controls
def partial_eta2(model, term):
    an = sm.stats.anova_lm(model, typ=2); return an.loc[term, "sum_sq"] / (an.loc[term, "sum_sq"] + an.loc["Residual", "sum_sq"])
rows = []
for D, key, lab in [(SB, "domain", "domain"), (SC, "register_m", "register")]:
    for s in ["Latin", "Cyrillic"]:
        sub = D[D.script == s]
        sub = sub[sub.groupby(key)[key].transform("size") >= MIN_STRATUM_N]
        for n in NAMES:
            groups = [g[f"fert_{n}"].values for _, g in sub.groupby(key)]
            H = stats.kruskal(*groups); k, N = len(groups), len(sub); eps2 = (H.statistic - k + 1) / (N - k)
            term = f"C({key})"
            m1 = smf.ols(f"np.log(Q('fert_{n}')) ~ C({key})", data=sub).fit()
            m2 = smf.ols(f"np.log(Q('fert_{n}')) ~ C({key}) + np.log(cpw)", data=sub).fit()
            m3 = smf.ols(f"np.log(Q('fert_{n}')) ~ C({key}) + np.log(cpw) + C(source)", data=sub).fit()
            e1, e2, e3 = partial_eta2(m1, term), partial_eta2(m2, term), partial_eta2(m3, term)
            an3 = sm.stats.anova_lm(m3, typ=2)
            rows.append({"factor": lab, "script": s, "tokenizer": n, "groups": k, "n": N, "KW_H": H.statistic, "epsilon2": eps2, "KW_p": H.pvalue,
                         "partial_eta2_group_only": e1, "partial_eta2_ctrl_letters": e2, "partial_eta2_ctrl_letters_source": e3,
                         "p_group_ctrl_letters_source": an3.loc[term, "PR(>F)"], "partial_eta2_letters": partial_eta2(m2, "np.log(cpw)")})
            add("RQ4", f"{lab} effect | {s}", f"{n}: Kruskal–Wallis", eps2, np.nan, np.nan, "epsilon²", extra=f"H={H.statistic:.1f}; k={k}; n={N}", p=H.pvalue)
            add("RQ4", f"{lab} effect controlled | {s}", f"{n}: OLS log fert ~ {lab} + log letters/word + source", e3, np.nan, np.nan,
                "partial η² of " + lab, extra=f"partial η² uncontrolled={e1:.3f}; with letters/word={e2:.3f}", p=an3.loc[term, "PR(>F)"])
rq4 = pd.DataFrame(rows); rq4.to_csv(f"{OUT}/rq4_domain_register_models.csv", index=False)
print(rq4[["factor", "script", "tokenizer", "groups", "n", "epsilon2", "partial_eta2_group_only", "partial_eta2_ctrl_letters",
           "partial_eta2_ctrl_letters_source", "partial_eta2_letters"]].round(4).to_string(index=False))
"""))

CELLS.append(("code", r"""
# 7 — Holm correction within each RQ family, final results table
res = holm(pd.DataFrame(RESULTS))
res["p_fmt"] = res.p.apply(lambda v: "" if pd.isna(v) else fp(v)); res["p_holm_fmt"] = res.p_holm.apply(lambda v: "" if pd.isna(v) else fp(v))
res.to_csv(f"{OUT}/stage6_all_tests.csv", index=False)
pd.set_option("display.width", 280); pd.set_option("display.max_colwidth", 70)
for fam in ["RQ1/RQ5", "RQ2", "RQ3", "RQ4"]:
    t = res[res.family == fam]
    print(f"\n=================== {fam}: {len(t)} tests ===================")
    print(t[["analysis", "comparison", "effect", "effect_lo", "effect_hi", "unit", "d_z", "rank_biserial", "p_holm_fmt"]].round(4).to_string(index=False))
print("\nTests with Holm-adjusted p ≥ 0.05:")
print(res[res.p_holm >= 0.05][["family", "analysis", "comparison", "effect", "p_holm_fmt"]].round(4).to_string(index=False) or "none")
"""))

CELLS.append(("code", r"""
# 8 — gate and copy to Drive
gate = {"n_tests": int(res.p.notna().sum()), "families": res.groupby("family").size().to_dict(),
        "nonsignificant_after_holm": int((res.p_holm >= 0.05).sum()), "docs_A": len(A), "docs_B": len(SB), "docs_C": len(SC),
        "exp2_docs": int(M2.gidx.nunique()), "exp3_words": int(len(P3)), "packages": PKG, "seed": SEED, "n_boot": N_BOOT,
        "files": sorted(os.listdir(OUT)), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
json.dump(gate, open(f"{OUT}/stage6_gate.json", "w"), indent=2); print(json.dumps(gate, indent=2))
import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True); print("Copied to", DRIVE_OUT)
"""))
