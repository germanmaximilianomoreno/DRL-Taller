import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from dotenv import load_dotenv
import wandb
from config import params  # Asegurate de que este archivo existe

# === CARGAR VARIABLES DE ENTORNO ===
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    raise ValueError("WANDB_API_KEY no está definido en el archivo .env")

# === INICIAR WANDB ===
wandb.init(
    project="frozenlake_rl",
    name="PPO_FrozenLake",
    config=params
)

# === CALLBACK PERSONALIZADO PARA LOGS ===
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info and "episode" in info:
                reward = info["episode"]["r"]
                wandb.log({"episode_reward": reward, "steps": self.num_timesteps})
        return True

# === ENTORNO DE ENTRENAMIENTO ===
env = Monitor(gym.make("FrozenLake-v1", is_slippery=True))

# === CREAR Y ENTRENAR MODELO PPO ===
model = PPO(
    policy=params["policy"],
    env=env,
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    n_steps=params["n_steps"],
    batch_size=params["batch_size"],
    ent_coef=params["ent_coef"],
    vf_coef=params["vf_coef"],
    max_grad_norm=params["max_grad_norm"],
    gae_lambda=params["gae_lambda"],
    clip_range=params["clip_range"],
    verbose=1,
)

reward_callback = RewardLoggerCallback()

start_time = time.time()
model.learn(total_timesteps=100000, callback=reward_callback)
end_time = time.time()

training_time = end_time - start_time

# === EVALUACIÓN DEL MODELO (sin render para evitar fallos) ===
eval_env = Monitor(gym.make("FrozenLake-v1", is_slippery=True))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

# === MOSTRAR RESULTADOS EN CONSOLA ===
print("\n✅ RESULTADOS DE EVALUACIÓN ===")
print(f"Recompensa promedio: {mean_reward:.2f}")
print(f"Desviación estándar: {std_reward:.2f}")
print(f"Tiempo total de entrenamiento: {training_time:.2f} segundos")

# === LOG FINAL EN WANDB ===
wandb.log({
    "mean_reward_eval": mean_reward,
    "std_reward_eval": std_reward,
    "training_time_seconds": training_time
}, step=100000)

# === CIERRE ===
env.close()
eval_env.close()
wandb.finish()