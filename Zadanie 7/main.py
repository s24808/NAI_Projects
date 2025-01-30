# autorzy: Filip Labuda, Jędrzej Stańczewski
# instrukcja użycia: wystarczy odpalić kod w jakimś IDE
# (wymagane zainstalowanie importowanych bibliotek)
# opis: implementacja Deep Q-Network do nauki gry w Breakout z Atari przy użyciu reinforcement learningu
# sieć neuronowa przewiduje wartości Q dla możliwych akcji, tworzony jest model główny i docelowy
# doświadczenia agenta przechowywane są w buforze, sieć neuronowa jest aktualizowana co X klatek (update_target_network)
# trening w założeniu kończy się po osiągnięciu limitu epizodów (dla wartości 0 jest to nieskończoność)
# lub osiągnięcia 40 punktów nagrody
# pliki wideo z gry na początku treningu oraz po tysiącu epizodów załączone w repozytorium (tylko tysiąc ze względu
# na ograniczenia sprzętowe.


import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers import AtariPreprocessing
import numpy as np
import tensorflow as tf
import ale_py
import gc

# konfiguracja hiperparametrów
seed = 42  # ziarno dla powtarzalności
gamma = 0.99  # współczynnik dyskontowania
epsilon = 1.0  # początkowa wartość eksploracji
epsilon_min = 0.1  # minimalna wartość epsilon
epsilon_max = 1.0  # maksymalna wartość epsilon
epsilon_interval = (epsilon_max - epsilon_min)  # zakres epsilon
batch_size = 32  # wielkość wsadu treningowego
max_steps_per_episode = 10000  # maksymalna liczba kroków w epizodzie
max_episodes = 0  # liczba epizodów (0 oznacza brak limitu)
render_after_episodes = 1000  # po ilu epizodach włączyć renderowanie

# tworzenie środowiska gry
env = gym.make("BreakoutNoFrameskip-v4", frameskip=1)
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
env.reset(seed=seed)

num_actions = 4  # liczba możliwych akcji

def create_q_model():
    """tworzy sieć neuronową q-network."""
    return keras.Sequential([
        layers.Lambda(lambda tensor: keras.ops.transpose(tensor, [0, 2, 3, 1]), output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
        layers.Conv2D(32, 8, strides=4, activation="relu"),
        layers.Conv2D(64, 4, strides=2, activation="relu"),
        layers.Conv2D(64, 3, strides=1, activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(num_actions, activation="linear"),
    ])

# inicjalizacja sieci neuronowych
model = create_q_model()
model_target = create_q_model()
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# bufor pamięci doświadczeń
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# parametry epsilon-greedy
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4  # co ile kroków aktualizować model
update_target_network = 10000  # co ile klatek aktualizować model docelowy
loss_function = keras.losses.Huber()  # funkcja straty

while True:
    # restart środowiska z renderowaniem po określonej liczbie epizodów
    if episode_count == render_after_episodes:
        env.close()
        env = gym.make("BreakoutNoFrameskip-v4", frameskip=1, render_mode="human")
        env = AtariPreprocessing(env)
        env = FrameStack(env, 4)
        print(f"restart środowiska - renderowanie włączone po {episode_count} epizodach!")
        print(f"średnia nagroda przed restartem: {np.mean(episode_reward_history)}")

    # reset gry i inicjalizacja stanu
    observation, _ = env.reset()
    state = np.array(observation)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # eksploracja lub wybór najlepszej akcji
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)  # losowa akcja
        else:
            state_tensor = keras.ops.convert_to_tensor(state)
            state_tensor = keras.ops.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = keras.ops.argmax(action_probs[0]).numpy()

        # zmniejszanie epsilon
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # wykonanie akcji i otrzymanie nowego stanu
        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward

        # zapis doświadczeń do bufora
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # aktualizacja modelu co kilka kroków
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = keras.ops.convert_to_tensor([float(done_history[i]) for i in indices])

            future_rewards = model_target.predict(state_next_sample)
            updated_q_values = rewards_sample + gamma * keras.ops.amax(future_rewards, axis=1)
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            masks = keras.ops.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = keras.ops.sum(keras.ops.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # aktualizacja modelu docelowego co 10 000 klatek
        if frame_count % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            print(f"running reward: {running_reward:.2f}, episode: {episode_count}, frame count: {frame_count}")

        # ograniczanie długości bufora doświadczeń
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    print(f"epizod {episode_count}, nagroda: {episode_reward}")

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    gc.collect()
    episode_count += 1

    # sprawdzenie warunku sukcesu
    if running_reward > 40:
        print(f"solved at episode {episode_count}!")
        break

    if max_episodes > 0 and episode_count >= max_episodes:
        print(f"stopped at episode {episode_count}!")
        break
