params = {
    "env": "FrozenLake-v1",
    "algorithm": "A2C",
    "policy": "MlpPolicy",  # También podrías probar "CnnPolicy" si adaptás la entrada
    "learning_rate": 7e-4,
    "gamma": 0.99,
    "n_steps": 5,            # Un poco mayor que para CartPole por la naturaleza del entorno
    "vf_coef": 0.25,
    "ent_coef": 0.01,
    "max_grad_norm": 0.5
}
