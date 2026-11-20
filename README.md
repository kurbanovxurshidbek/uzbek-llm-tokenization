# Evaluating Subword Tokenization Efficiency for the Uzbek Language in Multilingual LLMs

Code, annotations and run logs for the paper:

> K. Kurbanov and R. M. A. Kayumbek, "Evaluating Subword Tokenization Efficiency for the Uzbek Language in Multilingual Large Language Models," 2026.

The study measures how efficiently the tokenizers of **Qwen2.5-1.5B**, **Llama-3.2-1B** and **Gemma-2-2B** encode Uzbek text in Latin and Cyrillic script. It covers fertility, the share of words split into two or more tokens (PCW), byte fragments, controlled transliteration, the spelling of the oʻ/gʻ mark, and alignment with morpheme boundaries. Only the tokenizers are loaded; no model is run.

## Pinned inputs

| Item | Value |
|---|---|
| Corpus | [`Digitaluz/uzbek-language-corpus`](https://huggingface.co/datasets/Digitaluz/uzbek-language-corpus), revision `955412d35fca149ce5374daba0d01565c578a36d` (ODC-By) |
| Qwen2.5-1.5B tokenizer | commit `8faed761d45a…` |
| Llama-3.2-1B tokenizer | commit `4e20de362430…` |
| Gemma-2-2B tokenizer | commit `c5ebcd40d208…` |
| Random seed | 42 everywhere; each stratum uses `default_rng([42, crc32(stratum)])` |
| Bootstrap | 1,000 stratified replicates, 95% percentile intervals |
| Packages | Stages 1–3: transformers 5.23.0, tokenizers 0.23.3. Later stages: transformers 5.22.1, which reproduces the Stage 3 fertility values to the third decimal. Every notebook prints the exact versions in its first cell. |

The full commit hashes are printed and asserted in the notebooks.

## How to run

The notebooks are written for **Google Colab** and store intermediate files in `MyDrive/uz_tokenization/`. Run them in this order:

| Notebook | Paper section |
|---|---|
| `01_Stage1_Dataset_Inspection_and_Pilot` | §3: census, pilot audit |
| `02_Stage2_Cleaning_Sampling_Measurement` | §3–4: samples A/B/C, cleaning, per-word token counts |
| `03_Stage3_Exp1_Exp4` | §5.1, §5.4: efficiency, ranking, domains and registers |
| `04a_Stage4A_Transliterator_Validation` | §4: transliterator (86 unit tests, round trip, 400-word check) |
| `04b_Stage4B_Exp2a_Exp2b` | §5.2: controlled script experiment, oʻ/gʻ spelling |
| `src/stage5a_cell.py` | §4: draws the 500 morphology candidates |
| `05_Stage5C_Exp3_Morphology` | §5.3: morpheme-boundary alignment |
| `06_Stage6_Statistics_NOT_USED_IN_PAPER` | exploratory; **not** used in the paper |

Llama 3.2 and Gemma 2 are gated on Hugging Face. Accept their licences, then add your token as the Colab secret `HF_TOKEN`. **Never write the token into a notebook.**

`src/` holds the Python sources that generate the notebooks. `paper/make_figs.py` redraws the paper figures from the verified numbers and the logs in `results/`.

## Annotations (`annotations/`)

- `morph_candidates.csv`: 500 Latin word types, 125 from each of four frequency bands (seed 42).
- `morph_gold_v2.csv`: the surface morph segmentation used in the paper. `morph_gold_v1.csv` is kept for transparency; between v1 and v2, 4 entries changed and no metric moved by more than 0.2 points.
- `translit_manual_check_annotated.csv`: the 400-type transliteration check.

**Important:** the segmentation and the transliteration check were produced by an LLM (Claude, Anthropic) following written rules. A native-speaker annotator has **not** verified them. Corrections are welcome as issues or pull requests.

## Run logs (`results/`)

These are outputs copied from the Colab runs:

- the corpus census;
- domain and register means;
- the oʻzbek token-piece diagnostic;
- the Stage 5C summary computed on the v2 annotations.

## Licence

- **Code:** MIT (see `LICENSE`).
- **Annotations:** CC BY 4.0.
- **Corpus:** remains under its own ODC-By licence and is not redistributed here.

## Citation

If you use this code or the annotations, please cite the paper (the reference will be added on publication) and this repository:

> K. Kurbanov, "uzbek-llm-tokenization," GitHub repository, 2026. https://github.com/kurbanovxurshidbek/uzbek-llm-tokenization
