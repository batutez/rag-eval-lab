## 2026-11-06 — Repo setup (first commit)

- **Goal:** Create a separate, public repo to learn RAGAS + the DataMorgana method for building an evaluation set
- **Tried:** Not applicable for this log
- **Result:** Not applicable for this log
- **Decisions:** Not applicable for this log
- **Learned:** Not applicable for this log
- **Next:** Fix folder tracking; learn W&B (M0).

## 2026-11-08 — W&B basics (M0)

- **Goal:** Learn W&B experiment tracking before using it on real runs.
- **Tried:** `sandbox/m0_wandb.py`: a run with a config, logged metrics, a Table and an Artifact. Then re-ran with changed artifact content.
- **Result:** All runs appear in the W&B project; the artifact got versions `v0` and `v1`.
- **Decisions:** Use W&B for this repo. 
- **Learned:**
  - Each script execution is a new run with a random name, so set `name=` in `wandb.init`.
  - Same artifact name with changed content creates a new version; old versions stay in W&B's cloud, linked to the run that made them.
  - GitHub and W&B visibility are separate settings.
- **Next:** M1: chunk one document. (Weave: explore at M3 if needed.)

## 2026-11-09 — Tokenizer Experiment (M1a)

- **Goal:** Learn what is the token size (character-wise) for Turkish embedding models.
- **Tried:** I did the following experiment.
  - Selected one document from the corpus. (Yapay Zeka Eylem Planı).
  - Selected six embedding models. The models are;
    - intfloat/multilingual-e5-large
    - google/embeddinggemma-300m
    - BAAI/bge-m3
    - trmteb/turkish-embedding-model
    - Qwen/Qwen3-Embedding-0.6B
    - sentence-transformers/LaBSE
  - Chunk documents into paragraphs. So each chunk contains one paragraph. 
  - For each model; 
    - Every paragraph is tokenized and for every paragraph;
      - number of tokens and length of paragraphs (character-wise) are calculated.
      - Average character per token is calculated.
  - Results are logged inside W%B
- **Result:**
  - Doc01: 290 paragraphs, 96,714 characters.
  - Average characters per token: LaBSE 4.67 · BERTurk 4.57 · e5 4.38 · bge 4.38 · Gemma 3.69 · Qwen 3.08.
  - Longest paragraph: 847 tokens (e5/bge), 865 (BERTurk), 809 (LaBSE).
  - Paragraphs over the limit: 1 for each 512-token model (e5, BERTurk, LaBSE); 0 for Gemma, bge, Qwen.
  - W&B: 6 runs in group `m1a-tokenizers`.
- **Decisions:**
    Max Chunk size estimatation per model = token limit × chars/token × 0.85 (15 % margin, overhead not measured!)

    - intfloat/multilingual-e5-large: about 1,900 characters
    - trmteb/turkish-embedding-model: about 2,000 characters (limit unverified)
    - sentence-transformers/LaBSE: about 2,000 characters (limit unverified)
    - google/embeddinggemma-300m: about 6,400 characters
    - BAAI/bge-m3: about 30,000 characters
    - Qwen/Qwen3-Embedding-0.6B: about 85,000 characters

  Selecting 
- **Learned:**
  - e5 and bge-m3 share the same tokenizer (XLM-RoBERTa): identical tokens and counts.
  - Turkish suffixes are split off as separate tokens; splits follow frequency, not grammar.
  - Gemma needs ~20% more tokens than XLM-R and splits numbers digit by digit.
  - Qwen is byte-level: Turkish letters (ç, ı, ş, ü…) often become separate pieces, so it is the most token-expensive.
  - BERTurk lowercases and strips diacritics (ç→c, ü→u, â→a); "zekâ" becomes "zeka". All other tokenizers break "zekâ" into odd pieces.
  - Fewer tokens ≠ better retrieval: LaBSE is the most compact but scores lowest on TR-MTEB retrieval (46.47 vs 60.62 for e5-large).
  - A token limit includes special tokens and prefixes; tools cut longer input silently.
- **Next:**
  - Verify the BERTurk and LaBSE limits (512 inferred from warnings) and BERTurk's `strip_accents` setting.
  - Decide on text normalisation (â, ligatures like ﬂ).
  - M1: structure-aware chunking of Doc01 with a per-chunk token check.


## 2026-11-10 — Chunking Doc01 (M1)

- **Goal:** Turn Doc01 into retrievable chunks with metadata, ready for embedding in M2.
- **Tried:** `sandbox/m1_1_chunking.py` reads the md file block by block (blank-line separated), tracks the current page and headings, and stores each paragraph with its metadata. `sandbox/m1_2_option_c_experiment.ipynb` tested sentence packing, the fallback cut and overlap.
- **Result:**
  - 305 paragraphs found; 256 kept after the short-paragraph rule; 270 chunks (10 paragraphs over 1,000 characters were split).
  - Chunk text: 42 to 985 characters (median 261). Longest paragraph (3,915 characters) became 6 chunks.
  - Heading path + chunk: at most 1,187 characters, roughly 270 e5 tokens (estimate); exact token check runs with the e5 tokenizer in the script.
  - Output: `outputs/doc01_yapay_zeka_eylem_plani_chunks_max1000_ov1.jsonl` (git-ignored) and W&B group `m1-chunking`.
- **Decisions:**
  - Paragraphs under 40 characters are dropped (sidebar numbers, labels, diagram labels). Known losses: the end of one Eylem 8 sentence ("Programı ile destekleyecektir.") and "TÜBİTAK" at the end of Eylem 16's responsible list.
  - Long paragraphs: sentence packing up to 1,000 characters, one-sentence overlap, fallback cut at a space for sentences over 1,000 (not triggered in Doc01).
  - Each chunk carries: chunk_id, doc_id, page_pdf, page_printed, heading_path, eylem, part / n_parts, char_start / char_end of its paragraph.
  - Embedding text = heading path + blank line + chunk text, so the embedding sees which section and Eylem a chunk belongs to.
  - Pieces of one paragraph share the paragraph's character span (evaluation at paragraph level for split paragraphs).
- **Learned:**
  - Filtering by length must happen only for paragraphs, not before page markers and headings are read; otherwise all page numbers and section context are lost.
  - A fixed character window cuts inside words and sentences; packing whole sentences avoids that.
  - Overlap only affects the 10 split paragraphs here: 270 chunks with overlap vs 269 without. Its benefit is unknown until M2.
- **Next:**
  - M2: embed the chunks with e5 and test retrieval on the in-scope Doc01 questions of eval-set-v1.
  - Known issues to revisit: flat heading levels (EKSEN replaces "05 Eylemler" in the path), page-break fragments, ligatures (ﬂ).
