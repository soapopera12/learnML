# 12 Reinforcement learning

**Reinforcement Learning (RL)** is a type of machine learning where an agent learns to make decisions by interacting with an environment in order to achieve a goal. The agent takes actions in the environment, receives feedback in the form of rewards or penalties, and adjusts its behavior over time to maximize cumulative rewards.

### Key Concepts in Reinforcement Learning:

1. **Agent**: The learner or decision-maker, which interacts with the environment.
   
2. **Environment**: The external system with which the agent interacts. It provides the agent with feedback based on its actions.
   
3. **State (S)**: A representation of the environment at a specific point in time. The agent observes the state before taking actions.
   
4. **Action (A)**: The choices or decisions the agent makes. Based on the current state, the agent selects an action that influences the state of the environment.

5. **Reward (R)**: A signal from the environment that the agent receives after taking an action. Rewards guide the agent towards desirable outcomes by reinforcing good actions with positive rewards and discouraging bad actions with negative ones.

6. **Policy (π)**: A strategy or rule that the agent follows to choose actions in different states. The policy can be deterministic (fixed actions for states) or stochastic (actions chosen based on probabilities).

7. **Value Function (V)**: A prediction of future rewards. It helps the agent evaluate the desirability of being in a specific state, guiding it toward states with higher long-term rewards.

8. **Q-Value (Q)**: Also known as the action-value function, it represents the expected cumulative reward for taking a specific action in a given state and then following a policy.

9. **Exploration vs Exploitation**: 
   - **Exploration** involves trying new actions to discover their effects, even if it might not provide immediate rewards.
   - **Exploitation** means using the agent’s current knowledge to maximize the reward by choosing the best-known action.

### Types of Reinforcement Learning Algorithms:

1. **Model-Free RL**:
   - **Q-Learning**: A value-based method that learns the Q-values for each state-action pair and chooses actions based on the highest Q-values.
   - **SARSA (State-Action-Reward-State-Action)**: A variant of Q-Learning that updates the Q-values using the agent’s actual policy.
   - **Deep Q-Network (DQN)**: Combines Q-Learning with deep neural networks, allowing the agent to handle more complex environments.

2. **Model-Based RL**: Involves learning a model of the environment (i.e., a function that predicts the next state and reward) and using this model to make decisions.

3. **Policy-Based RL**: The agent directly learns the policy without estimating value functions.
   - **REINFORCE**: A basic policy-gradient algorithm that adjusts the policy parameters based on the rewards.
   - **Proximal Policy Optimization (PPO)**: An improved policy-gradient method that optimizes policies in a more stable and efficient way.

### Applications of Reinforcement Learning:
- Game playing (e.g., AlphaGo, OpenAI’s Dota 2 bots)
- Robotics (autonomous control of robots)
- Self-driving cars
- Financial trading
- Healthcare (personalized treatment plans)
  
### Example of RL in Action:
Imagine an agent in a video game learning to avoid obstacles while moving towards a goal. The game environment provides feedback after each action:
- If the agent reaches the goal, it gets a reward.
- If it hits an obstacle, it receives a penalty.
Over time, the agent learns which actions are most likely to lead it to the goal while avoiding penalties, optimizing its performance.

Reinforcement learning focuses on long-term success by balancing immediate rewards with future outcomes.

## 12.1 Actor-critic model