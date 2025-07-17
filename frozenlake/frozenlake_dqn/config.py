params = {
    "policy": "MlpPolicy",
    "learning_rate": 5e-4,               # Más baja para evitar oscilaciones
    "gamma": 0.99,
    "buffer_size": 20000,               # Más grande → mejor diversidad de experiencias
    "exploration_fraction": 0.2,        # Explora durante más tiempo (20% del total)
    "exploration_final_eps": 0.01,      # Explora un poquito siempre
    "target_update_interval": 250,      # Menos frecuente = más estabilidad
    "train_freq": 1,                    # Aprende más seguido (cada paso)
    "learning_starts": 500,             # Empieza a aprender más temprano
    "batch_size": 64,                   # Más muestras por batch para mejorar el gradiente
    "steps": 100000,
    "save_dir" : "./frozenlake/frozenlake_dqn/models",
    "check_freq" : 1000
}