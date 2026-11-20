# Stage 4, part A — transliterator construction and validation (Experiment 2a prerequisites). Measurement cells come after validation.
CELLS = []

CELLS.append(("md", r"""
# Stage 4 (part A) — Building and validating the Uzbek transliterator
Experiment 2a measures the *pure* script effect by tokenizing **the same text in both scripts**. That only works if the transliteration is correct, so this notebook builds and validates the transliterator first. **No tokenization measurements are made here.**

* **Cyrillic → Latin:** official 1995 Uzbek Latin alphabet; ʻ (U+02BB) for oʻ/gʻ and ʼ (U+02BC) for the tutuq belgisi. Context rules: е → *ye* at word start / after a vowel / after ъ,ь, otherwise *e*; ц → *ts* after a vowel, otherwise *s*; ь is dropped.
* **Latin → Cyrillic** (the reverse direction, used as a robustness check): sh→ш, ch→ч, oʻ→ў, gʻ→ғ, yo/yu/ya/ye→ё/ю/я/е, word-initial e→э, ts after a vowel→ц.
* **Words excluded from the controlled comparison:** Russian-only letters (щ, ы), Latin letters with no Uzbek counterpart (c outside *ch*, w), and mixed-script words. Their share is reported.
* **Validation:** (1) unit tests, (2) corpus round-trip consistency, (3) **manual check of 400 frequency-weighted word types by a native speaker** (required before part B).
"""))

CELLS.append(("code", r"""
# 1 — setup and inputs (Stage-2 outputs on Drive)
import os, re, json, datetime, unicodedata, zlib
import numpy as np, pandas as pd
from collections import Counter
SEED = 42; MIN_STRATUM_N = 30
from google.colab import drive
drive.mount("/content/drive")
S2 = "/content/drive/MyDrive/uz_tokenization/stage2"
OUT = "/content/uz_tok/stage4"; os.makedirs(OUT, exist_ok=True)
DRIVE_OUT = "/content/drive/MyDrive/uz_tokenization/stage4"; os.makedirs(DRIVE_OUT, exist_ok=True)

mem = pd.read_csv(f"{S2}/sample_membership_final.csv")
docs = pd.read_parquet(f"{S2}/doc_metrics.parquet", columns=["gidx", "script", "source", "format_origin"])
texts = pd.read_parquet(f"{S2}/sample_clean.parquet", columns=["gidx", "text"])
A = mem[mem.kept & (mem["sample"] == "A")].merge(docs, on="gidx")
sizes = A.groupby("stratum").size(); use = sizes[sizes >= MIN_STRATUM_N].index
A = A[A.stratum.isin(use)]
print(f"Sample A documents: {len(A):,} in {len(use)} strata | Latin {int((A.script=='Latin').sum()):,} | Cyrillic {int((A.script=='Cyrillic').sum()):,}")
"""))

CELLS.append(("code", r"""
# 2 — transliterator (both directions) and unit tests
OKINA, TUTUQ = "ʻ", "ʼ"
APOS_ANY = "'‘’ʻʼ`´"
CYR_VOW = "аеёиоуўэюя"
C2L = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
       "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ч": "ch", "ш": "sh", "ъ": TUTUQ,
       "ь": "", "э": "e", "ю": "yu", "я": "ya", "ё": "yo", "ў": "o" + OKINA, "қ": "q", "ғ": "g" + OKINA, "ҳ": "h"}
L2C_MULTI = [("o" + OKINA, "ў"), ("g" + OKINA, "ғ"), ("sh", "ш"), ("ch", "ч"), ("yo", "ё"), ("yu", "ю"), ("ya", "я"), ("ye", "е")]
L2C = {"a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ", "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н",
       "o": "о", "p": "п", "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "x": "х", "y": "й", "z": "з", TUTUQ: "ъ"}

def script_of(w):
    s = {unicodedata.name(ch, "").split(" ")[0] for ch in w if ch.isalpha() and ch not in OKINA + TUTUQ}
    return "Latin" if s == {"LATIN"} else "Cyrillic" if s == {"CYRILLIC"} else "Mixed"

def cyr_transliterable(w):
    return script_of(w) == "Cyrillic" and not any(ch in "щыЩЫ" for ch in w)

def lat_transliterable(w):
    lw = w.lower()
    return script_of(w) == "Latin" and "w" not in lw and "c" not in lw.replace("ch", "")

def _case(t, src_upper, allcaps):
    if not src_upper or not t: return t
    return t.upper() if allcaps else t[0].upper() + t[1:]

def cyr2lat(w):
    allcaps = w.isupper() and len(w) > 1
    out = []
    for i, ch in enumerate(w):
        lo = ch.lower(); prev = w[i - 1].lower() if i else ""; nxt = w[i + 1].lower() if i + 1 < len(w) else ""
        if lo == "е":   t = "ye" if (i == 0 or prev in CYR_VOW + "ъь") else "e"
        elif lo == "ц": t = "ts" if (prev and prev in CYR_VOW) else "s"
        elif lo == "с" and nxt == "ҳ": t = "s" + TUTUQ          # Исҳоқ → Isʼhoq (separator keeps s+h distinct from sh)
        elif lo == "ъ" and nxt == "е": t = ""             # объект → obyekt, съезд → syezd (ъ before е not written); меъёр → meʼyor keeps ʼ
        elif lo in C2L: t = C2L[lo]
        else:           out.append(ch); continue
        out.append(_case(t, ch != lo, allcaps))
    return "".join(out)

def norm_lat_apos(w):
    # apostrophe after o/g = letter mark of oʻ/gʻ (U+02BB); any other apostrophe = tutuq belgisi (U+02BC)
    return re.sub(rf"([^{APOS_ANY}])[{APOS_ANY}]", lambda m: m.group(1) + (OKINA if m.group(1) in "oOgG" else TUTUQ), w)

def lat2cyr(w):
    w = norm_lat_apos(w); lw = w.lower(); allcaps = w.isupper() and len(w) > 1
    out, i = [], 0
    while i < len(w):
        prev = lw[i - 1] if i else ""
        after_vowel = bool(prev) and (prev in "aeiou" or (prev == OKINA and i >= 2 and lw[i - 2] == "o"))
        two = lw[i:i + 2]; t, n = None, 1
        for a, b in L2C_MULTI:
            if two == a and not (a[0] == "y" and lw[i + 2:i + 3] == OKINA):   # "yoʻl" = y + oʻ, not yo + ʻ
                t, n = b, 2; break
        if t == "е" and prev and prev.isalpha() and prev not in "aeiou": t = "ъе"   # obyekt → объект (ye after a consonant = ъе)
        if t is None and two == "ts" and after_vowel and re.match(r"ts(iy|ion|ey)", lw[i:i + 5]): t, n = "ц", 2   # militsiya, revolyutsion, litsey; not ketsa/ketsin
        if t is None:
            c = lw[i]
            if c == "e": t = "э" if (i == 0 or after_vowel) else "е"
            elif c == TUTUQ and prev == "s" and lw[i + 1:i + 2] == "h": i += 1; continue   # sʼh = с + ҳ (separator only)
            elif c in L2C: t = L2C[c]
            else: out.append(w[i]); i += 1; continue
        out.append(_case(t, w[i].isupper(), allcaps)); i += n
    return "".join(out)

UNIT = [("Ўзбекистон", "Oʻzbekiston"), ("ЎЗБЕКИСТОН", "OʻZBEKISTON"), ("Республикаси", "Respublikasi"), ("қишлоқ", "qishloq"),
        ("хўжалиги", "xoʻjaligi"), ("ғалаба", "gʻalaba"), ("ҳаёт", "hayot"), ("Шаҳар", "Shahar"), ("чой", "choy"), ("ёшлар", "yoshlar"),
        ("Ёшлар", "Yoshlar"), ("юз", "yuz"), ("якшанба", "yakshanba"), ("ер", "yer"), ("поезд", "poyezd"), ("муаммо", "muammo"),
        ("эълон", "eʼlon"), ("маълумот", "maʼlumot"), ("Тошкент", "Toshkent"), ("жамият", "jamiyat"), ("мактаб", "maktab"),
        ("милиция", "militsiya"), ("цирк", "sirk"), ("концерт", "konsert"), ("йўл", "yoʻl"), ("Йўл", "Yoʻl"), ("объект", "obyekt"),
        ("субъектлари", "subyektlari"), ("съезд", "syezd"), ("АҚШ", "AQSH"), ("УЧУН", "UCHUN"), ("ЯНГИ", "YANGI"),
        ("дунё", "dunyo"), ("меъёр", "meʼyor"), ("меъёрий", "meʼyoriy"), ("адабиётшунослари", "adabiyotshunoslari"),
        ("кетса", "ketsa"), ("кетсин", "ketsin"), ("ўтса", "oʻtsa"), ("лицей", "litsey"), ("Исҳоқ", "Isʼhoq"), ("диссертацияси", "dissertatsiyasi")]
LOSSY_C2L = {"цирк", "концерт"}          # ц → s is not reversible; tested in one direction only
fails = []
for c, l in UNIT:
    if cyr2lat(c) != l: fails.append(("C→L", c, cyr2lat(c), l))
    if c not in LOSSY_C2L and lat2cyr(l) != c: fails.append(("L→C", l, lat2cyr(l), c))
for raw, exp in [("o'zbek", "oʻzbek"), ("g‘alaba", "gʻalaba"), ("ma'lumot", "maʼlumot"), ("O’zbekiston", "Oʻzbekiston")]:
    if norm_lat_apos(raw) != exp: fails.append(("apostrophe", raw, norm_lat_apos(raw), exp))
print(f"Unit tests: {2 * len(UNIT) - len(LOSSY_C2L) + 4} checks, {len(fails)} failures")
for f in fails: print("  FAIL", f)
"""))

CELLS.append(("code", r"""
# 3 — word inventory per script, exclusions, and corpus round-trip consistency
WORD_RE = re.compile(rf"[^\W\d_]+(?:[{APOS_ANY}][^\W\d_]+)*", re.UNICODE)
tx = texts.set_index("gidx").text
freq = {s: Counter(w for g in A.loc[A.script == s, "gidx"] for w in WORD_RE.findall(tx[g])) for s in ["Cyrillic", "Latin"]}

def inventory(s):
    f = freq[s]; tot = sum(f.values())
    ok = cyr_transliterable if s == "Cyrillic" else lat_transliterable
    cat = Counter()
    for w, c in f.items():
        sc = script_of(w)
        if ok(w): cat["transliterable"] += c
        elif sc == s: cat["excluded: non-Uzbek letter (щ/ы or c/w)"] += c
        elif sc == "Mixed": cat["excluded: mixed-script word"] += c
        else: cat[f"other-script word (kept unchanged in both conditions)"] += c
    return pd.Series({k: 100 * v / tot for k, v in cat.items()}).round(3), tot

RT, SUMMARY = {}, {}
for s in ["Cyrillic", "Latin"]:
    inv, tot = inventory(s)
    print(f"\n=== {s} documents: {tot:,} word tokens, {len(freq[s]):,} types — % of tokens ===\n{inv.to_string()}")
    fwd, back = (cyr2lat, lat2cyr) if s == "Cyrillic" else (lat2cyr, cyr2lat)
    ok = cyr_transliterable if s == "Cyrillic" else lat_transliterable
    rows = []
    for w, c in freq[s].items():
        if not ok(w): continue
        t = fwd(w); b = back(t); ref = w if s == "Cyrillic" else norm_lat_apos(w)
        rows.append((w, t, b, c, b == ref))
    rt = pd.DataFrame(rows, columns=["word", "translit", "back", "freq", "roundtrip_ok"]); RT[s] = rt
    rate = 100 * rt.loc[rt.roundtrip_ok, "freq"].sum() / rt.freq.sum()
    print(f"Round-trip identical ({s} → other → {s}), token-weighted: {rate:.2f}%  | type-level: {100 * rt.roundtrip_ok.mean():.2f}%")
    bad = rt[~rt.roundtrip_ok].sort_values("freq", ascending=False)
    print("Most frequent round-trip mismatches (a mismatch is not necessarily an error, e.g. ь is dropped by the official rules):")
    print(bad.head(15).to_string(index=False))
    rt.to_csv(f"{OUT}/roundtrip_{s}.csv", index=False)
    SUMMARY[s] = {"word_tokens": int(tot), "word_types": len(freq[s]), **{k: float(v) for k, v in inv.items()},
                  "roundtrip_token_%": round(float(rate), 3), "roundtrip_type_%": round(float(100 * rt.roundtrip_ok.mean()), 3)}

print("\n================ SUMMARY (send this block) ================")
print(json.dumps(SUMMARY, indent=2, ensure_ascii=False))
json.dump(SUMMARY, open(f"{OUT}/stage4a_inventory_summary.json", "w"), indent=2, ensure_ascii=False)
"""))

CELLS.append(("code", r"""
# 4 — manual-validation sample: 200 word types per direction, drawn with probability ∝ corpus frequency
N_CHECK = 200
rng = np.random.default_rng(SEED)
parts = []
for s, direction in [("Cyrillic", "Cyrillic→Latin"), ("Latin", "Latin→Cyrillic")]:
    rt = RT[s]
    idx = rng.choice(len(rt), size=min(N_CHECK, len(rt)), replace=False, p=rt.freq / rt.freq.sum())
    p = rt.iloc[idx][["word", "translit", "freq"]].rename(columns={"word": "source_word", "translit": "transliteration", "freq": "corpus_freq"})
    p.insert(0, "direction", direction); parts.append(p)
check = pd.concat(parts, ignore_index=True)
check["correct_1_0"] = ""; check["corrected_form"] = ""; check["note"] = ""
check = check.sample(frac=1, random_state=SEED).reset_index(drop=True)      # shuffle so directions are interleaved
check.to_csv(f"{OUT}/translit_manual_check.csv", index=False, encoding="utf-8-sig")
json.dump({"seed": SEED, "n_check_per_direction": N_CHECK, "unit_test_failures": len(fails),
           "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"}, open(f"{OUT}/stage4a_params.json", "w"), indent=2)
import shutil; shutil.copytree(OUT, DRIVE_OUT, dirs_exist_ok=True)
print(f"Saved {len(check)} rows → {DRIVE_OUT}/translit_manual_check.csv")
print(check.head(15).to_string(index=False))
"""))
