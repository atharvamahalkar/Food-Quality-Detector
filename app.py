import numpy as np
import serial
import time
import matplotlib.pyplot as plt
from collections import deque

# ==============================
#   ENVIRONMENT CLASS
# ==============================

class PHSystem:
    """
    Interface between real pH sensor and RL agent.
    If real sensor is connected -> reads live data via Serial.
    If no serial found -> falls back to simulated environment.
    """

    def __init__(self, port='COM3', baudrate=9600, target_pH=7.0, simulate=False):
        self.target = target_pH
        self.simulate = simulate
        self.ser = None

        if not simulate:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                print(f"✅ Connected to pH sensor on {port}")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Serial connection failed: {e}")
                print("Switching to simulation mode.")
                self.simulate = True

        self.reset()

    def reset(self):
        self.pH = np.random.uniform(6.0, 8.0)
        return self.pH

    def read_sensor(self):
        """Read live pH sensor data or simulate if unavailable."""
        if self.simulate:
            self.pH += np.random.normal(0, 0.05)
        else:
            try:
                line = self.ser.readline().decode().strip()
                if line:
                    self.pH = float(line)
            except:
                pass
        return np.clip(self.pH, 0, 14)

    def step(self, action):
        """
        0 = do nothing
        1 = add acid (decrease pH)
        2 = add base (increase pH)
        """
        current_ph = self.read_sensor()

        # Simulate actuator effect (for prototype testing)
        if self.simulate:
            if action == 1:
                current_ph -= np.random.uniform(0.05, 0.15)
            elif action == 2:
                current_ph += np.random.uniform(0.05, 0.15)

        # reward closer to target (max = 0)
        reward = -abs(current_ph - self.target)
        done = False
        self.pH = np.clip(current_ph, 0, 14)

        return self.pH, reward, done


# ==============================
#   Q-LEARNING AGENT
# ==============================

class QLearningAgent:
    def __init__(self, num_states=100, num_actions=3, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_states, num_actions))

    def get_state_index(self, ph_value):
        return int((ph_value / 14) * (self.num_states - 1))

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state_idx])

    def update(self, s, a, r, s_next):
        best_next = np.argmax(self.q_table[s_next])
        td_target = r + self.gamma * self.q_table[s_next, best_next]
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.alpha * td_error


# ==============================
#   TRAINING LOOP
# ==============================

def train_agent(simulate=True, episodes=2000):
    env = PHSystem(simulate=simulate)
    agent = QLearningAgent()
    rewards_history = []
    recent_rewards = deque(maxlen=100)

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    reward_line, = ax[0].plot([], [], label='Reward')
    ph_line, = ax[1].plot([], [], label='pH Value')
    ax[0].set_title("Training Rewards")
    ax[1].set_title("Live pH Readings")
    ax[1].axhline(7.0, color='r', linestyle='--', label='Target (7.0)')
    ax[0].legend(); ax[1].legend()

    ph_log = []
    reward_log = []

    for ep in range(episodes):
        ph = env.reset()
        total_reward = 0
        ep_ph = []

        for step in range(200):
            s = agent.get_state_index(ph)
            a = agent.choose_action(s)
            next_ph, r, done = env.step(a)
            s_next = agent.get_state_index(next_ph)
            agent.update(s, a, r, s_next)

            ph = next_ph
            total_reward += r
            ep_ph.append(ph)

            # Live plot update
            if step % 10 == 0:
                reward_log.append(total_reward)
                ph_log.append(ph)
                reward_line.set_data(range(len(reward_log)), reward_log)
                ph_line.set_data(range(len(ph_log)), ph_log)
                ax[0].relim(); ax[0].autoscale_view()
                ax[1].relim(); ax[1].autoscale_view()
                plt.pause(0.001)

        recent_rewards.append(total_reward)
        agent.epsilon *= agent.epsilon_decay
        avg_reward = np.mean(recent_rewards)
        rewards_history.append(avg_reward)

        print(f"Episode {ep+1}/{episodes} | Avg Reward: {avg_reward:.3f} | ε: {agent.epsilon:.3f}")

    plt.ioff()
    plt.show()
    print("✅ Training Completed.")
    return agent, rewards_history


# ==============================
#   MAIN EXECUTION
# ==============================

if __name__ == "__main__":
    # Set simulate=False to use real pH sensor via Serial
    agent, rewards = train_agent(simulate=True, episodes=300)
// automated update 13 Jan 2026 15:01:02
