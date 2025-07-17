import os
import time
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor  #Correcto Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from dotenv import load_dotenv
import wandb
from config import params  # Asegurate que este archivo existe y es válido

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
    name="A2C_FrozenLake",
    config=params
)

# === CALLBACK PERSONALIZADO PARA LOGS ===
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_step = 0
        self.episode_count = 0  # ← contador de episodios

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        for idx, info in enumerate(infos):
            if dones[idx]:  # cuando termina un episodio
                if "episode" in info:
                    reward = info["episode"]["r"]
                    self.episode_count += 1  # ← aumentamos el número de episodio
                    wandb.log({
                        "recompensa_por_episodio": reward,
                        "movimientos_por_episodio": self.episode_step,
                        "movimientos_globales": self.num_timesteps,
                        "numero_episodio": self.episode_count
                    })
                self.episode_step = 0  # reiniciamos el contador del episodio
            else:
                self.episode_step += 1  # contamos los pasos dentro del episodio
        return True

# === ENTORNO DE ENTRENAMIENTO ===
env = Monitor(gym.make("FrozenLake-v1", is_slippery=True))

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

# === ENTRENAMIENTO CON MEDICIÓN DE TIEMPO ===
reward_callback = RewardLoggerCallback()
start_time = time.time()
model.learn(total_timesteps=params["step"], callback=reward_callback)
end_time = time.time()
training_time = end_time - start_time

# === EVALUACIÓN ===
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=params["n_eval_episodes"])

# === RESULTADOS EN CONSOLA ===
print("\nRESULTADOS DE EVALUACIÓN ===")
print(f"Recompensa promedio: {mean_reward:.2f}")
print(f"Desviación estándar: {std_reward:.2f}")
print(f"Tiempo total de entrenamiento: {training_time:.2f} segundos")

# === LOG FINAL EN WANDB ===
wandb.log({
    "mean_reward_eval": mean_reward,
    "std_reward_eval": std_reward,
    "training_time_seconds": training_time
}, step=params["step"])

# === CIERRE DE ENTORNOS Y SESIÓN ===
env.close()
wandb.finish()