import os
import time
import gymnasium as gym
import numpy as np
import psutil
import GPUtil
from stable_baselines3 import DQN
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
    project="frozenlake_rl",
    name="DQN_FrozenLake_maxi",
    config=params
)

# === CALLBACK PERSONALIZADO PARA LOG Y GUARDADO DEL MEJOR MODELO ===
class CustomWandbCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            all_rewards = []
            for env in self.training_env.envs:
                monitor_env = env.envs[0] if isinstance(env, DummyVecEnv) else env
                if hasattr(monitor_env, 'get_episode_rewards'):
                    all_rewards.extend(monitor_env.get_episode_rewards())

            if all_rewards:
                mean_reward = np.mean(all_rewards[-self.check_freq:])
                elapsed_time = time.time() - self.start_time
                fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0

                # === USO DE GPU ===
                gpus = GPUtil.getGPUs()
                gpu_util = gpus[0].load * 100 if gpus else 0.0
                gpu_mem = gpus[0].memoryUsed if gpus else 0.0

                # === USO DE CPU y RAM ===
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent

                # === Log a wandb ===
                wandb.log({
                    "mean_reward_percent": mean_reward * 100,
                    "steps": self.num_timesteps,
                    "elapsed_time_sec": elapsed_time,
                    "fps": fps,
                    "gpu_util (%)": gpu_util,
                    "gpu_mem (MB)": gpu_mem,
                    "cpu (%)": cpu_usage,
                    "ram (%)": ram_usage
                })

                # Guardar el mejor modelo
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

# === CREAR MODELO DQN ===
model = DQN(
    policy=params["policy"],
    env=env,
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    buffer_size=params.get("buffer_size", 10000),
    learning_starts=params.get("learning_starts", 1000),
    batch_size=params.get("batch_size", 32),
    tau=params.get("tau", 1.0),
    train_freq=params.get("train_freq", 4),
    target_update_interval=params.get("target_update_interval", 1000),
    verbose=1,
)

# === CALLBACK Y ENTRENAMIENTO ===
save_dir = params["save_dir"]
os.makedirs(save_dir, exist_ok=True)
callback = CustomWandbCallback(check_freq=params["check_freq"], save_path=save_dir)

start_time = time.time()
model.learn(total_timesteps=params["steps"], callback=callback)
training_time = time.time() - start_time

print(f"Tiempo total de entrenamiento: {training_time:.2f} segundos")

# === LOG FINAL EN WANDB ===
wandb.log({
    "training_time_seconds": training_time
}, step=params["steps"])

# === CIERRE ===
env.close()
wandb.finish()