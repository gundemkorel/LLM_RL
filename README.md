# LLM Explanation RL Environment

This repo is a starting point for training a tool-using LLM policy (π) to generate explanations for a black-box model (f),
rewarded by distributional alignment between f(x) and an explanation-induced model g(E):
R = -KL(f(x) || g(E)).

## Structure
- `src/envs/explain_env.py` — Gym-like environment exposing the KL reward.
- `src/models/blackbox.py` — Opaque model f with `predict_proba(x)`.
- `src/models/g_explainer.py` — g(E) classifier turning text into P(y|E).
- `src/tools/*` — Explainability tool wrappers (SHAP/LIME/PDP) as callable Python functions.
- `src/agents/agent_langchain.py` — (stub) Agent wiring to call tools and produce E.
- `src/training/train_ppo.py` — PPO loop using TRL-like API (stubbed for now).
- `src/utils/reward.py` — KL and helpers.
- `tests/*` — Basic tests to sanity-check the pieces.

## Quickstart
1. Create a virtualenv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. Run unit tests:
   ```bash
   pytest -q
   ```

3. (Later) Train PPO once you connect a real LLM and TRL:
   ```bash
   python -m src.training.train_ppo
   ```

## Notes
- Files ship with minimal, runnable stubs (sklearn logistic for f, a tiny bag-of-words for g).
- Tool wrappers return deterministic toy outputs if SHAP/LIME are not installed; replace with real implementations.
- Use this skeleton with a code-generation model (e.g., Codex-style) by giving it **one file at a time** and asking for implementations marked `# TODO`.
