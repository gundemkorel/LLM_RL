from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.data.datasets import TabularDataset
from src.envs.explain_env import ExplainEnv
from src.models.blackbox import BlackBoxModel


try:  # Optional torch / transformers / trl stack
    import torch
except Exception:  # pragma: no cover - torch not available in minimal envs
    torch = None  # type: ignore

try:  # pragma: no cover - transformers optional
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore

try:  # pragma: no cover - trl optional
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
except Exception:  # pragma: no cover
    AutoModelForCausalLMWithValueHead = None  # type: ignore
    PPOConfig = None  # type: ignore
    PPOTrainer = None  # type: ignore


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s - %(message)s",
            stream=sys.stdout,
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:  # pragma: no cover - random imported lazily
        import random

        random.seed(seed)
    except Exception:  # pragma: no cover
        pass

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU CI unlikely
            torch.cuda.manual_seed_all(seed)


def format_prompt(x: np.ndarray, p: np.ndarray) -> str:
    x_str = ", ".join(f"{xi:.3f}" for xi in x)
    p_str = ", ".join(f"{pi:.3f}" for pi in p)
    return (
        "You are an explanation agent. Given features x and model predictions p, "
        "produce a plain language explanation.\n"
        f"x = [{x_str}]\n"
        f"p = [{p_str}]\n"
        "Explanation:"
    )



def _build_default_dataset(
    seed: int,
    n_features: int = 10,
    dataset_size: int = 128,
) -> tuple[TabularDataset, BlackBoxModel]:
    X, y = make_classification(
        n_samples=dataset_size,
        n_features=n_features,
        n_informative=min(6, n_features),
        n_redundant=min(2, max(n_features - 6, 0)),
        n_classes=2,
        random_state=seed,
    )
    dataset = TabularDataset.from_arrays(X, y, feature_names=None)
    blackbox = BlackBoxModel.from_training(
        dataset=dataset,
        estimator_factory=lambda: LogisticRegression(max_iter=1000, random_state=seed),
    )
    return dataset, blackbox


def _to_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if torch is not None and isinstance(value, torch.Tensor):  # pragma: no cover - torch optional
        if value.numel() == 0:
            return default
        return float(value.detach().float().cpu().item())
    if isinstance(value, (list, tuple)) and value:
        return _to_float(value[0], default)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover
            return default
    return default


def _extract_stat(stats: Dict[str, object], keys: Sequence[str], default: float = float("nan")) -> float:
    for key in keys:
        if key in stats:
            return _to_float(stats[key], default)
    return default


class DummyTokenizer:
    pad_token_id: int = 0
    eos_token_id: int = 1

    def __init__(self) -> None:
        self._offset = 2

    def encode(self, text: str) -> List[int]:
        return [self._offset + (ord(ch) % 256) for ch in text]

    def decode(self, tokens: Iterable[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for token in tokens:
            if skip_special_tokens and token in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(chr((int(token) - self._offset) % 256))
        return "".join(chars)

    def batch_decode(self, sequences: Sequence[Sequence[int]], skip_special_tokens: bool = True) -> List[str]:
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def __call__(self, texts: Sequence[str] | str, return_tensors: str | None = None, padding: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t) for t in texts]
        max_len = max((len(e) for e in encoded), default=0)
        arrs = []
        for e in encoded:
            sequence = list(e)
            if return_tensors == "pt" and torch is not None:
                tensor = torch.tensor(sequence, dtype=torch.long)
                if padding:
                    pad_len = max_len - len(sequence)
                    if pad_len > 0:
                        tensor = torch.cat([
                            tensor,
                            torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        ])
                arrs.append(tensor)
            else:
                if padding:
                    pad_len = max_len - len(sequence)
                    if pad_len > 0:
                        sequence = sequence + [self.pad_token_id] * pad_len
                arrs.append(sequence)
        if return_tensors == "pt" and torch is not None:
            input_ids = torch.stack(arrs) if arrs else torch.empty((0, 0), dtype=torch.long)
            attention_mask = (input_ids != self.pad_token_id).long()
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": arrs, "attention_mask": [[1] * len(seq) for seq in arrs]}


class DummyValueHeadModel:
    def __init__(self, tokenizer: DummyTokenizer, seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.rng = np.random.default_rng(seed)
        self._vocab = [
            "evidence",
            "suggests",
            "feature",
            "impact",
            "class",
            "because",
            "low",
            "high",
            "risk",
            "signal",
        ]

    def generate_text(self, prompt: str, max_new_tokens: int) -> Tuple[List[int], List[int], str]:
        query_tokens = self.tokenizer.encode(prompt)
        length = int(self.rng.integers(1, max(2, max_new_tokens + 1)))
        words = [self.rng.choice(self._vocab) for _ in range(length)]
        response_text = " ".join(words)
        response_tokens = self.tokenizer.encode(response_text) + [self.tokenizer.eos_token_id]
        return query_tokens, response_tokens, response_text

    # mimic torch API used in code paths
    def to(self, *_args, **_kwargs) -> "DummyValueHeadModel":
        return self

    def eval(self) -> "DummyValueHeadModel":
        return self

    def train(self) -> "DummyValueHeadModel":  # pragma: no cover - no-op
        return self


class DummyPPOTrainer:
    def __init__(self, model: DummyValueHeadModel, tokenizer: DummyTokenizer, target_kl: float, lr: float, grad_accumulation_steps: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.target_kl = target_kl
        self.lr = lr
        self.grad_accumulation_steps = grad_accumulation_steps
        self._step = 0

    def step(self, queries: Sequence[Sequence[int]], responses: Sequence[Sequence[int]], rewards: Sequence[float]) -> Dict[str, float]:
        self._step += 1
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        # Construct simple proxies for KL metrics so logs remain meaningful.
        avg_response_tokens = float(np.mean([len(r) for r in responses])) if responses else 0.0
        kl = avg_response_tokens / 100.0
        ref_kl = kl / 2.0
        return {
            "kl": kl,
            "ref_kl": ref_kl,
            "mean_reward": mean_reward,
            "learning_rate": self.lr,
            "target_kl": self.target_kl,
        }


@dataclass
class PPOTrainingConfig:
    model_name: str = "dummy"
    batch_size: int = 2
    rollout_steps: int = 2
    lr: float = 1e-5
    target_kl: float = 0.1
    seed: int = 0
    max_new_tokens: int = 32
    grad_accumulation_steps: int = 1


def _init_real_trl_stack(cfg: PPOTrainingConfig):  # pragma: no cover - depends on optional deps
    assert torch is not None
    assert AutoTokenizer is not None
    assert AutoModelForCausalLMWithValueHead is not None
    assert PPOConfig is not None and PPOTrainer is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model_name, **model_kwargs)
    model.to(device)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg.model_name, **model_kwargs)
    ref_model.to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    config_kwargs = {
        "model_name": cfg.model_name,
        "learning_rate": cfg.lr,
        "batch_size": cfg.batch_size,
        "mini_batch_size": max(1, cfg.batch_size // cfg.grad_accumulation_steps),
        "target_kl": cfg.target_kl,
    }
    try:
        ppo_config = PPOConfig(
            **config_kwargs,
            gradient_accumulation_steps=cfg.grad_accumulation_steps,
        )
    except TypeError:
        ppo_config = PPOConfig(**config_kwargs)
        if hasattr(ppo_config, "gradient_accumulation_steps"):
            setattr(ppo_config, "gradient_accumulation_steps", cfg.grad_accumulation_steps)

    trainer = PPOTrainer(ppo_config, model, ref_model=ref_model, tokenizer=tokenizer)
    return trainer, tokenizer, device


def _init_dummy_stack(cfg: PPOTrainingConfig) -> Tuple[DummyPPOTrainer, DummyTokenizer, None]:
    warnings.warn(
        "transformers/trl not available - falling back to dummy PPO components."
    )
    tokenizer = DummyTokenizer()
    model = DummyValueHeadModel(tokenizer, seed=cfg.seed)
    trainer = DummyPPOTrainer(model=model, tokenizer=tokenizer, target_kl=cfg.target_kl, lr=cfg.lr, grad_accumulation_steps=cfg.grad_accumulation_steps)
    return trainer, tokenizer, None


def initialize_trainer(cfg: PPOTrainingConfig):
    have_trl = (
        torch is not None
        and AutoTokenizer is not None
        and AutoModelForCausalLMWithValueHead is not None
        and PPOConfig is not None
        and PPOTrainer is not None
    )
    if not have_trl:
        return _init_dummy_stack(cfg)
    try:
        return _init_real_trl_stack(cfg)
    except Exception as exc:  # pragma: no cover - best effort fallback
        warnings.warn(f"Falling back to dummy PPO stack due to: {exc}")
        return _init_dummy_stack(cfg)


def _sample_with_dummy(
    model: DummyValueHeadModel,
    tokenizer: DummyTokenizer,
    prompts: Sequence[str],
    max_new_tokens: int,
) -> Tuple[List[List[int]], List[List[int]], List[str], List[int], List[int]]:
    query_tensors: List[List[int]] = []
    response_tensors: List[List[int]] = []
    response_texts: List[str] = []
    prompt_lens: List[int] = []
    response_lens: List[int] = []
    for prompt in prompts:
        query_tokens, response_tokens, response_text = model.generate_text(prompt, max_new_tokens)
        query_tensors.append(query_tokens)
        response_tensors.append(response_tokens)
        response_texts.append(response_text)
        prompt_lens.append(len(query_tokens))
        response_lens.append(len(response_tokens))
    return query_tensors, response_tensors, response_texts, prompt_lens, response_lens


def _sample_with_trl(
    trainer,
    tokenizer,
    prompts: Sequence[str],
    device,
    max_new_tokens: int,
):  # pragma: no cover - depends on optional deps
    query_tensors: List[torch.Tensor] = []
    response_tensors: List[torch.Tensor] = []
    response_texts: List[str] = []
    prompt_lens: List[int] = []
    response_lens: List[int] = []

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        prompt_len = input_ids.shape[-1]

        policy_model = getattr(trainer, "model", trainer)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id

        with torch.no_grad():
            output = policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )

        generated = output[0, prompt_len:]
        if generated.numel() == 0:
            filler = pad_token_id if pad_token_id is not None else 0
            generated = torch.tensor([filler], device=device)

        query_tensors.append(input_ids.squeeze(0))
        response_tensors.append(generated)
        prompt_lens.append(int(prompt_len))
        response_lens.append(int(generated.shape[-1]))
        response_text = tokenizer.decode(generated.detach().cpu().tolist(), skip_special_tokens=True)
        response_texts.append(response_text)

    return query_tensors, response_tensors, response_texts, prompt_lens, response_lens


def sample_model_responses(
    trainer,
    tokenizer,
    prompts: Sequence[str],
    device,
    max_new_tokens: int,
):
    if isinstance(trainer, DummyPPOTrainer):
        return _sample_with_dummy(trainer.model, tokenizer, prompts, max_new_tokens)
    return _sample_with_trl(trainer, tokenizer, prompts, device, max_new_tokens)


def _convert_rewards(rewards: Sequence[float], device, use_torch: bool):
    if not use_torch:
        return list(rewards)
    tensors: List[torch.Tensor] = []
    for r in rewards:
        tensor = torch.tensor([r], dtype=torch.float32, device=device)
        tensors.append(tensor)
    return tensors


def train(cfg: PPOTrainingConfig) -> List[Dict[str, float]]:
    _ensure_logging()
    set_seed(cfg.seed)

    trainer, tokenizer, device = initialize_trainer(cfg)
    use_torch = torch is not None and not isinstance(trainer, DummyPPOTrainer)


    dataset, blackbox = _build_default_dataset(seed=cfg.seed)
    env = ExplainEnv(dataset=dataset, blackbox=blackbox, seed=cfg.seed)

    stats_history: List[Dict[str, float]] = []

    buffer_queries = []
    buffer_responses = []
    buffer_rewards = []
    buffer_prompt_lens: List[int] = []
    buffer_response_lens: List[int] = []

    def flush_buffers(step_idx: int) -> None:
        if not buffer_queries:
            return
        batch_queries = list(buffer_queries)
        batch_responses = list(buffer_responses)
        batch_rewards = list(buffer_rewards)
        batch_prompt_lens = list(buffer_prompt_lens)
        batch_response_lens = list(buffer_response_lens)

        rewards_tensor = _convert_rewards(batch_rewards, device, use_torch)
        train_stats = trainer.step(batch_queries, batch_responses, rewards_tensor)
        stats_history.append({k: _to_float(v) for k, v in train_stats.items()})

        kl = _extract_stat(train_stats, ["kl", "ppo/kl", "objective/kl"])
        ref_kl = _extract_stat(train_stats, ["ref_kl", "ppo/ref_kl", "objective/ref_kl"])
        mean_reward = float(np.mean(batch_rewards)) if batch_rewards else float("nan")
        prompt_tokens = float(np.mean(batch_prompt_lens)) if batch_prompt_lens else 0.0
        response_tokens = float(np.mean(batch_response_lens)) if batch_response_lens else 0.0

        logging.info(
            "update=%d | mean_reward=%.4f | kl=%.4f | ref_kl=%.4f | prompt_tokens=%.2f | response_tokens=%.2f",
            step_idx,
            mean_reward,
            kl,
            ref_kl,
            prompt_tokens,
            response_tokens,
        )

        buffer_queries.clear()
        buffer_responses.clear()
        buffer_rewards.clear()
        buffer_prompt_lens.clear()
        buffer_response_lens.clear()

    for rollout_idx in range(cfg.rollout_steps):
        prompts: List[str] = []
        for _ in range(cfg.batch_size):
            obs = env.reset()
            prompt_text = format_prompt(obs["x"], obs["p"])
            prompts.append(prompt_text)

        queries, responses, responses_text, prompt_lens, response_lens = sample_model_responses(
            trainer,
            tokenizer,
            prompts,
            device,
            cfg.max_new_tokens,
        )

        rewards: List[float] = []
        for response_text in responses_text:
            step_out = env.step(response_text)
            rewards.append(float(step_out.reward))

        buffer_queries.extend(queries)
        buffer_responses.extend(responses)
        buffer_rewards.extend(rewards)
        buffer_prompt_lens.extend(prompt_lens)
        buffer_response_lens.extend(response_lens)

        if (rollout_idx + 1) % max(1, cfg.grad_accumulation_steps) == 0:
            flush_buffers(rollout_idx + 1)

    flush_buffers(cfg.rollout_steps)

    return stats_history


def parse_args(argv: Sequence[str] | None = None) -> PPOTrainingConfig:
    parser = argparse.ArgumentParser(description="Train PPO on ExplainEnv")
    parser.add_argument("--model_name", type=str, default="dummy")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--rollout_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    args = parser.parse_args(args=argv)
    return PPOTrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        target_kl=args.target_kl,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        grad_accumulation_steps=max(1, args.grad_accumulation_steps),
    )


if __name__ == "__main__":
    config = parse_args(None)
    history = train(config)
    if history:
        last_stats = history[-1]
        logging.info("Final stats: %s", {k: round(v, 4) for k, v in last_stats.items()})
