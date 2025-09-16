from src.envs.explain_env import ExplainEnv

def test_env_step_runs():
    env = ExplainEnv(n_features=6, seed=42)
    obs = env.reset()
    e = "features suggest class 1 with strong signal"
    out = env.step(e)
    assert hasattr(out, 'reward')
    assert out.done is True
