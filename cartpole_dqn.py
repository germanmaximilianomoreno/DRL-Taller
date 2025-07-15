import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics as Monitor
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import psutil
import time
#import GPUtil  # opcional, si hay GPU

# === CALLBACK PERSONALIZADO ===
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [None])[0]
        if info and "episode" in info:
            self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_start(self):
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        self.memory_usage.append(psutil.virtual_memory().percent)
        self.gpu_usage.append(None)

# === CREAR ENTORNO ===
env = gym.make("CartPole-v1")
env = Monitor(env)

# === ENTRENAMIENTO ===
reward_callback = RewardLoggerCallback()
model = DQN(
    "MlpPolicy", 
    env, 
    learning_rate=1e-3,              # Más alto para que aprenda más rápido al principio
    buffer_size=50000,               # Tamaño del buffer de experiencia
    learning_starts=1000,            # No aprende hasta tener suficientes experiencias
    batch_size=32,
    tau=1.0,
    gamma=0.99,                      # Factor de descuento
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,        # Fracción del tiempo explorando
    exploration_final_eps=0.02,      # Valor mínimo de epsilon
    verbose=1,
    tensorboard_log="./dqn_cartpole_tensorboard/"
)

start_time = time.time()
model.learn(total_timesteps=10000, callback=reward_callback)
end_time = time.time()

# === EVALUACIÓN ===
eval_env = Monitor(gym.make("CartPole-v1", render_mode="human"))  # render_mode agregado
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"\nRecompensa promedio: {mean_reward}, desviación estándar: {std_reward}")
print(f"Tiempo total de entrenamiento: {end_time - start_time:.2f} segundos")

# === DEMOSTRACIÓN DEL AGENTE ENTRENADO ===
import time

obs, _ = eval_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    eval_env.render()
    time.sleep(0.03)

eval_env.close()

# === GRÁFICAS ===
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(reward_callback.episode_rewards)
plt.title("Recompensa por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")

plt.subplot(1, 3, 2)
plt.plot(reward_callback.cpu_usage, label="CPU %")
plt.plot(reward_callback.memory_usage, label="RAM %")
plt.title("Uso de CPU y RAM")
plt.xlabel("Checkpoint")
plt.ylabel("% de uso")
plt.legend()

if any(reward_callback.gpu_usage):
    plt.subplot(1, 3, 3)
    plt.plot(reward_callback.gpu_usage)
    plt.title("Uso de GPU")
    plt.xlabel("Checkpoint")
    plt.ylabel("% de uso")

plt.tight_layout()
plt.show()

env.close()
