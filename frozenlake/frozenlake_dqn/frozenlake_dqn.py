import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time
import wandb
from dotenv import load_dotenv
import os
from config import params

# === CARGAR VARIABLES DE ENTORNO ===
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# === INICIALIZAR WANDB ===
wandb.init(
    project="frozenlake_rl",
    name="DQN_FrozenLake",
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

# === ENTRENAMIENTO ===
reward_callback = RewardLoggerCallback()
model = DQN(
    policy=params["policy"],
    env=env,
    learning_rate=params["learning_rate"],
    gamma=params["gamma"],
    buffer_size=params["buffer_size"],
    exploration_fraction=params["exploration_fraction"],
    exploration_final_eps=params["exploration_final_eps"],
    target_update_interval=params["target_update_interval"],
    train_freq=params["train_freq"],
    learning_starts=params["learning_starts"],
    batch_size=params["batch_size"],
    verbose=1,
)

start_time = time.time()
model.learn(total_timesteps=params["step"], callback=reward_callback)
end_time = time.time()

# === EVALUACIÓN ===
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=params["n_eval_episodes"])
print(f"\nRecompensa promedio: {mean_reward}, desviación estándar: {std_reward}")
print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

wandb.log({
    "mean_reward_eval": mean_reward,
    "std_reward_eval": std_reward,
    "training_time_seconds": end_time - start_time
}, step=params["step"])

env.close()
wandb.finish()