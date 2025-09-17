from __future__ import annotations

from src.training.train_ppo import PPOTrainingConfig, train


def test_dummy_training_smoke():
    cfg = PPOTrainingConfig(
        model_name="dummy",
        batch_size=2,
        rollout_steps=2,
        lr=1e-4,
        target_kl=0.1,
        seed=0,
        max_new_tokens=8,
        grad_accumulation_steps=1,
    )
    history = train(cfg)
    assert history, "Expected at least one PPO update"
    for stats in history:
        assert "kl" in stats
        assert "ref_kl" in stats


def test_dummy_training_with_accumulation():
    cfg = PPOTrainingConfig(
        model_name="dummy",
        batch_size=2,
        rollout_steps=3,
        lr=1e-4,
        target_kl=0.1,
        seed=123,
        max_new_tokens=6,
        grad_accumulation_steps=2,
    )
    history = train(cfg)
    # Two updates: one after steps {0,1} and a final flush for step {2}
    assert len(history) == 2
    for stats in history:
        assert "kl" in stats
        assert "ref_kl" in stats
