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