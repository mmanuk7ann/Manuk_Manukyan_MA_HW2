
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import ABC, abstractmethod
import logging

# Constants
TOTAL_RUNS = 20000
BANDIT_PROBS = [1, 2, 3, 4]
EPSILON = 0.1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Bandit(ABC):

    @abstractmethod
    def __init__(self, p):
        self.true_p = p
        self.est_p = 0.0
        self.n = 0

    @abstractmethod
    def __repr__(self):
        return f'Bandit with true p={self.true_p}'

    @abstractmethod
    def pull(self):
        return 1 if np.random.random() < self.true_p else 0

    @abstractmethod
    def update(self, x):
        self.n += 1
        self.est_p = self.est_p + (x - self.est_p) / self.n

    @abstractmethod
    def experiment(self):
        bandits = [self.__class__(p) for p in BANDIT_PROBS]
        rewards = np.zeros(TOTAL_RUNS)
        explore_count = 0
        exploit_count = 0
        optimal_count = 0
        optimal_idx = np.argmax([b.true_p for b in bandits])

        for i in range(TOTAL_RUNS):
            if np.random.rand() < EPSILON:
                explore_count += 1
                j = np.random.randint(len(bandits))
            else:
                exploit_count += 1
                j = np.argmax([b.est_p for b in bandits])

            optimal_count += 1 if j == optimal_idx else 0
            reward = bandits[j].pull()
            rewards[i] = reward
            bandits[j].update(reward)

        return bandits, rewards, explore_count, exploit_count, optimal_count

    @abstractmethod
    def report(self):
        results = self.experiment()
        bandits, rewards, explores, exploits, optimals = results

        for i, b in enumerate(bandits):
            logger.info(f"Bandit {i} estimate: {b.est_p:.4f}")

        logger.info(f"Total reward: {np.sum(rewards)}")
        logger.info(f"Average reward: {np.mean(rewards):.4f}")
        logger.info(f"Explores: {explores}")
        logger.info(f"Exploits: {exploits}")
        logger.info(f"Optimal choices: {optimals}")

        cum_rewards = np.cumsum(rewards)
        win_rates = cum_rewards / (np.arange(TOTAL_RUNS) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(win_rates, label="Observed win rate")
        plt.axhline(y=max(BANDIT_PROBS), color='r', linestyle='--', label="Max possible")
        plt.legend()
        plt.title("Win Rate Convergence")
        plt.xlabel("Trials")
        plt.ylabel("Win Rate")
        plt.grid(True)
        plt.show()


class Visualization:

    def plot1(self, rewards):
        plt.figure(figsize=(12, 6))
        for label, data in rewards.items():
            cum_data = np.cumsum(data)
            rate = cum_data / (np.arange(len(data)) + 1)
            plt.plot(rate, label=label)
        plt.xlabel("Trial")
        plt.ylabel("Win Rate")
        plt.title("Bandit Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot2(self, rewards_eg, regrets_eg, rewards_ts, regrets_ts):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(rewards_eg, label="Epsilon-Greedy")
        ax1.plot(rewards_ts, label="Thompson")
        ax1.set_title("Rewards Comparison")
        ax1.set_xlabel("Trials")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(regrets_eg, label="Epsilon-Greedy")
        ax2.plot(regrets_ts, label="Thompson")
        ax2.set_title("Regrets Comparison")
        ax2.set_xlabel("Trials")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


class EpsilonGreedy(Bandit):

    def __init__(self, rewards, trials):
        self.rewards = rewards
        self.trials = trials
        self.k = len(rewards)
        self.estimates = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        self.total_reward = 0
        self.history = []
        self.regrets = []
        self.optimal = np.argmax(rewards)
        self.arm_rewards = [[] for _ in range(self.k)]

    def __repr__(self):
        return f"EpsilonGreedy({self.k} arms)"

    def pull(self, arm):
        return np.random.normal(loc=self.rewards[arm], scale=1.0)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]

    def experiment(self):
        for t in range(1, self.trials + 1):
            eps = 1.0 / t if t > 0 else 1.0
            if np.random.rand() < eps:
                arm = np.random.randint(self.k)
            else:
                arm = np.argmax(self.estimates)

            reward = self.pull(arm)
            self.arm_rewards[arm].append(reward)
            self.update(arm, reward)

            self.total_reward += reward
            self.history.append(self.total_reward)
            regret = self.rewards[self.optimal] - self.rewards[arm]
            self.regrets.append(regret)

        return self

    def report(self):
        logger.info(f"Total reward: {self.total_reward:.2f}")
        logger.info(f"Average reward: {self.total_reward / self.trials:.4f}")
        logger.info(f"Total regret: {sum(self.regrets):.2f}")

        pd.DataFrame({
            'Bandit': [np.argmax(self.estimates)],
            'Reward': [self.total_reward],
            'Algorithm': ['EpsilonGreedy']
        }).to_csv("epsilon_results.csv", index=False)

        win_rates = np.array(self.history) / (np.arange(self.trials) + 1)
        plt.plot(win_rates)
        plt.axhline(y=max(self.rewards), color='r', linestyle='--')
        plt.title("Epsilon-Greedy Performance")
        plt.xlabel("Trials")
        plt.ylabel("Win Rate")
        plt.grid(True)
        plt.show()


class ThompsonSampling(Bandit):

    def __init__(self, rewards, trials):
        self.rewards = rewards
        self.trials = trials
        self.k = len(rewards)
        self.alpha = np.ones(self.k)
        self.beta = np.ones(self.k)
        self.total_reward = 0
        self.history = []
        self.regrets = []
        self.optimal = np.argmax(rewards)

    def __repr__(self):
        return f"ThompsonSampling({self.k} arms)"

    def pull(self, arm):
        return np.random.normal(loc=self.rewards[arm], scale=1.0)

    def update(self, arm, reward):
        if reward > self.rewards[arm]:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self):
        for t in range(self.trials):
            samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.k)]
            arm = np.argmax(samples)
            reward = self.pull(arm)
            self.update(arm, reward)

            self.total_reward += reward
            self.history.append(self.total_reward)
            regret = self.rewards[self.optimal] - self.rewards[arm]
            self.regrets.append(regret)

        return self

    def report(self):
        logger.info(f"Total reward: {self.total_reward:.2f}")
        logger.info(f"Average reward: {self.total_reward / self.trials:.4f}")
        logger.info(f"Total regret: {sum(self.regrets):.2f}")

        pd.DataFrame({
            'Bandit': [np.argmax(self.alpha)],
            'Reward': [self.total_reward],
            'Algorithm': ['ThompsonSampling']
        }).to_csv("thompson_results.csv", index=False)

        win_rates = np.array(self.history) / (np.arange(self.trials) + 1)
        plt.plot(win_rates)
        plt.axhline(y=max(self.rewards), color='r', linestyle='--')
        plt.title("Thompson Sampling Performance")
        plt.xlabel("Trials")
        plt.ylabel("Win Rate")
        plt.grid(True)
        plt.show()


def comparison():
    rewards = BANDIT_PROBS
    trials = TOTAL_RUNS

    # Run Epsilon-Greedy
    eg = EpsilonGreedy(rewards, trials).experiment()
    arm_data = {f"Arm {i}": r for i, r in enumerate(eg.arm_rewards)}

    viz = Visualization()
    viz.plot1(arm_data)
    eg.report()

    # Run Thompson Sampling
    ts = ThompsonSampling(rewards, trials).experiment()
    ts.report()

    # Compare both
    viz.plot2(eg.history, np.cumsum(eg.regrets),
              ts.history, np.cumsum(ts.regrets))


if __name__ == '__main__':
    logging.info("Starting bandit simulations")

    # Test logging levels
    logging.debug("Debug message")
    logging.info("Info message")
    logging.warning("Warning message")
    logging.error("Error message")
    logging.critical("Critical message")

    # Run experiments
    comparison()