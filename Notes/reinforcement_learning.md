
## Reinforcement Learning

There is usually a agent that performs some action which can lead to certain reward

1. Agent: A worker who explores different actions to get rewards and reach the goal.
    a. Value based: Has no policy, picks actions greedily based on state values.
    b. Policy based: Has no value function, uses the policy function to pick actions.
    c. Actor critic: Uses both value and policy functions.
    d. Model free: Uses policy and/or value functions but has no model.
    e. Model based: Uses policy and/or value functions and has a model.
2. Policy: Agent’s behaviour function which is a map from state to action.
3. Value function: Represents how good is each state and/or action.
4. Model: Agent’s representation of the environment. It predicts what the environment will do next. The predictions are of the next state and next immediate reward.
5. Reward: Indicates how well the agent is doing at time step t.
6. Goal: Select actions to maximise total future reward.
7. Environment: The external system the agent interacts with
    a. Fully Observable Environments: Agent directly observes environment
    b. Partially Observable Environments: Agent indirectly observes environment.
8. History: It is the sequence of observation
9. State: The state is the information used to determine what happens next.
    a. Environment state: The actual state of the environment.
    b. Agent State: The internal state of the agent.
    c. Information State: The set of all observations, actions, and rewards that summarize the agent’s knowledge.

    