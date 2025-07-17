params = {
    "env": "FrozenLake-v1",
    "algorithm": "A2C",
    "policy": "MlpPolicy",
    "learning_rate": 7e-4,
    "gamma": 0.99,
    "n_steps": 5,          
    "vf_coef": 0.25,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5,
    "steps": 100000,
    "save_dir" : "./frozenlake/frozenlake_a2c/models",
    "check_freq" : 1000
}
