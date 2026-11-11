# Datasets

Every source used by Tameez, where it lives on disk, and what it is for.
Download with `python scripts/download_data.py`. Raw files are git-ignored.

| Set | Name (`data/raw/<name>`) | Hugging Face / Git source | Files fetched | Rows used | License | Role |
|---|---|---|---|---|---|---|
| A | `aya_dataset` | CohereLabs/aya_dataset | `data/{train,test}-*.parquet` | 1,153 Hindi | Apache-2.0 | Hindi SFT |
| A/B | `indic_align_instruct` | ai4bharat/indic-align | `dolly/Dolly.parquet`, `oasst/oasst.parquet`, `anudesh/anudesh1.parquet` | 15,011 / 19,909 / 36,820 | CC BY 4.0 | Hindi SFT; Anudesh also feeds benign prompts in B |
| B | `indic_align_toxic` | ai4bharat/indic-align | `hhrlhf/hh-rlhf.parquet`, `toxicmatrix/toxic_prompts_sarvam.parquet` | 32,669 / 90,352 | CC BY 4.0 | Harmful prompts (prompts only) |
| B | `textdetox_toxicity` | textdetox/multilingual_toxicity_dataset | `hi`, `hin`, `en` | 5,000 / 4,363 / 5,000 | OpenRAIL++ | Angry-user rants |
| B | `textdetox_lexicon` | textdetox/multilingual_toxic_lexicon | `hi`, `hin`, `en` | 133 / 209 / 3,386 | OpenRAIL++ | Bad-word reward |
| B | `setu_nsfw` | github.com/AI4Bharat/setu (sparse) | `setu/data/filter_data/nsfw/*.txt` | Hindi 173, English 403 | MIT | Bad-word reward |
| C | `aya_redteaming` | CohereLabs/aya_redteaming | `aya_hin.jsonl`, `aya_eng.jsonl` | 915 Hindi | Apache-2.0 | Unsafe-reply rate |
| C | `indicsafe` | github.com/ppattnayak/indicsafe | `data/Full_translated_prompts.csv` | 500 Hindi | MIT (repo) | Unsafe rate + over-refusal |
| C | `suraksha_eval` | SurakshaAI/SurakshaEval | all | 116 Hindi | CC BY-NC-SA 4.0 | Unsafe-reply rate only (no refuse labels) |
| D | `milu` | ai4bharat/MILU | all | 14,831 Hindi test | CC BY 4.0 | Knowledge MCQ (no-regression check) |
| D | `aya_evaluation_suite` | CohereLabs/aya_evaluation_suite | `dolly_human_edited/test-*.parquet` | 200 Hindi | Apache-2.0 | Helpfulness win rate |
| ref | `dolly_15k` | databricks/databricks-dolly-15k | `databricks-dolly-15k.jsonl` | — | CC BY-SA 3.0 | Leakage check only (maps Aya-Eval `source_id` to English prompt) |

## Schemas (as downloaded)

- **Aya Dataset:** `inputs, targets, language, language_code, annotation_type, user_id`. Filter with `language == "Hindi"`.
- **IndicAlign (Dolly, OASST, HH-RLHF, Toxic-Matrix):** `doc_id, num_turns, eng_Latn, hin_Deva, hin_Latn, …` (plus 20+ other language columns). Each language cell is a list of turns `[[user, assistant], …]`. `hin_Latn` is machine transliteration, so do not use it as Hinglish (see [DATA_ANALYSIS.md](DATA_ANALYSIS.md)).
- **Anudesh:** `id, interactions, num_turns`. `interactions` = `[[user, assistant], …]`, mixed languages; filter by script.
- **textdetox toxicity:** `text, toxic` (0/1). **textdetox lexicon:** `text`.
- **Setu NSFW:** one entry per line, some multi-word.
- **Aya Red-teaming:** `prompt, language, harm_category (JSON list string), global_or_local, literal_translation, semantic_translation, explanation`.
- **IndicSafe:** `Prompt (English), Category, Language, Translated_Prompt, category`. Normalise `Category` by stripping a trailing "Prompt(s)".
- **Aya Evaluation Suite (dolly_human_edited):** `id, inputs, targets, language (hin), script, source_id`.

## Built sets (`data/processed/`, made by `scripts/build_data.py`)

| File | Rows (CPU build, before Hinglish rewrites) | Schema |
|---|---|---|
| `A_sft.json` | ~13.2K | `id, messages (system/user/assistant), source, lang` |
| `B_pool.json` | ~3.5K (→ ~5K with rewrites) | `prompt (messages), reference (refuse/help), bucket, lang, allowed_terms, source, seed_id` |
| `C_safety.json` | ~1.76K | `id, prompt, set, category, expected (refuse/help/none), lang, allowed_terms` |
| `C_mini.json` | 180 | stratified subset of C for checkpoint scoring |
| `D_ayaeval.json` | 200 | `id, prompt, reference_answer, lang` |
| `D_milu.json` | 14,831 | `question, option1-4, target, domain, subject` |
| `manifest.json` | – | counts per set / bucket / language / source, filters applied |
| `cache/` | – | `synthetic.json` (Aya rewrites), `guard_prompt_labels.json`, reused on re-runs |

Every row's split comes from a hash of its id (`tameez.sources.role`), so test rows can never drift into training.

## Folder conventions

```
data/
  raw/              downloaded as-is, never edited (git-ignored)
  processed/        built sets + manifest (git-ignored; rebuilt by scripts)
  lexicons/         merged lexicon (git-ignored) + whitelist.txt (tracked)
  tricky_seeds.json hand-written tricky-benign seeds (tracked)
```
