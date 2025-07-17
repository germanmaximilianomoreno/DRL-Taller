params = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "step": 10000,
    "n_eval_episodes": 10,
}