import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics as Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time
import wandb
from dotenv import load_dotenv
import os
from config import params

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

# === INICIALIZAR WANDB ===
wandb.init(
    project="cartpole_rl",
    name="A2C_CartPole",
    config=params
)

# === CALLBACK PERSONALIZADO ===
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [None])[0]
        if info and "episode" in info:
            reward = info["episode"]["r"]
            wandb.log({"episode_reward": reward, "steps": self.num_timesteps})
        return True

# === CREAR ENTORNO ===
env = gym.make("CartPole-v1")
env = Monitor(env)

# === ENTRENAMIENTO ===
reward_callback = RewardLoggerCallback()
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

start_time = time.time()
model.learn(total_timesteps=10000, callback=reward_callback)
end_time = time.time()

# === EVALUACIÓN ===
eval_env = Monitor(gym.make("CartPole-v1", render_mode="human"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print(f"\nRecompensa promedio: {mean_reward}, desviación estándar: {std_reward}")
print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

wandb.log({
    "mean_reward_eval": mean_reward,
    "std_reward_eval": std_reward,
    "training_time_seconds": end_time - start_time
}) 

# === DEMOSTRACIÓN DEL AGENTE ENTRENADO ===
obs, _ = eval_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()
    time.sleep(0.03)
eval_env.close()

env.close()

# === FINALIZAR SESIÓN WANDB ===
wandb.finish()