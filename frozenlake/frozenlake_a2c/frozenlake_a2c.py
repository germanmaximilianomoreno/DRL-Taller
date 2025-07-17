import os
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from dotenv import load_dotenv
import wandb
from config import params

# === CARGAR VARIABLES DE ENTORNO ===
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    raise ValueError("WANDB_API_KEY no está definido en el archivo .env")

# === INICIALIZAR WANDB ===
wandb.init(
    project="prueba_maxi",
    name="A2C_FrozenLake",
    config=params
)

# === CALLBACK PERSONALIZADO PARA GUARDAR EL MEJOR MODELO Y LOGS ===
class CustomWandbCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(CustomWandbCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            all_rewards = []
            for env in self.training_env.envs:
                logger_env = env.envs[0] if isinstance(env, DummyVecEnv) else env
                if hasattr(logger_env, 'get_episode_rewards'):
                    all_rewards.extend(logger_env.get_episode_rewards())

            if all_rewards:
                mean_reward = np.mean(all_rewards[-self.check_freq:])
                wandb.log({'mean_reward': mean_reward, 'steps': self.num_timesteps})

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model'))

        return True

# === CONFIGURAR ENTORNO CON MONITOR ===
def make_env():
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = Monitor(env)  # Log automático de episodios y recompensas
    return env

env = DummyVecEnv([make_env])  # Se necesita DummyVecEnv incluso para un solo entorno

# === CREAR MODELO A2C ===
model = A2C(
    policy=params["policy"],
    env=env,
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    n_steps=params["n_steps"],
    vf_coef=params["vf_coef"],
    ent_coef=params["ent_coef"],
    max_grad_norm=params["max_grad_norm"],
    use_rms_prop=True,
    verbose=1,
)

# === CALLBACK DE ENTRENAMIENTO ===
save_dir = params["save_dir"]
os.makedirs(save_dir, exist_ok=True)
custom_callback = CustomWandbCallback(params["check_freq"], save_path=save_dir)

# === ENTRENAMIENTO CON MEDICIÓN DE TIEMPO ===
start_time = time.time()
model.learn(total_timesteps=params["steps"], callback=custom_callback)
end_time = time.time()
training_time = end_time - start_time

# === RESULTADOS EN CONSOLA ===
print(f"Tiempo total de entrenamiento: {training_time:.2f} segundos")

# === LOG FINAL EN WANDB ===
wandb.log({
    "training_time_seconds": training_time
}, step=params["steps"])

# === CIERRE DE ENTORNOS Y SESIÓN ===
env.close()
wandb.finish()