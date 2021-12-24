from policy import *


"""
Theory - Monte Carlo 
The basic intuition is:

- We do not know the environment, so we sample an episode from beginning to end by running our current policy
- We try to estimate the action-values rather than the state values. This is because we are working model-free so just knowing state values won't help us select the best actions.
- The value of a state-action value is defined as the future returns from the first visit of that state-action
- Based on this we can improve our policy and repeat the process until the algorithm converges

"""

print(inspect.getsource(r.monte_carlo_learning))

# We do 100 iterations of monte carlo learning while maintaining a high exploration rate of 0.5:

for k in range(100):
    eps = 0.5
    r.monte_carlo_learning(epsilon=eps)

r.visualize_policy()

r.agent.action_function.max(axis=2).astype(int)

"""

Theory - Temporal Difference Learning

- Like Policy Iteration, we can back up state-action values from the successor state action without waiting for the episode to end.
- We update our state-action value in the direction of the successor state action value.
- The algorithm is called SARSA: State-Action-Reward-State-Action.
- Epsilon is gradually lowered (the GLIE property)

"""

print(inspect.getsource(r.sarsa_td))

p = Piece(piece='king')
env = Board()
r = Reinforce(p, env)
r.sarsa_td(n_episodes=10000, alpha=0.2, gamma=0.9)

r.visualize_policy()

"""

Theory - TD-lambda
In Monte Carlo we do a full-depth backup while in Temporal Difference Learning we de a 1-step backup. You could also choose a depth in-between: backup by n steps. But what value to choose for n?

- TD lambda uses all n-steps and discounts them with factor lambda
- This is called lambda-returns
- TD-lambda uses an eligibility-trace to keep track of the previously encountered states
- This way action-values can be updated in retrospect

"""

print(inspect.getsource(r.sarsa_lambda))

p = Piece(piece='king')
env = Board()
r = Reinforce(p, env)
r.sarsa_lambda(n_episodes=10000, alpha=0.2, gamma=0.9)

r.visualize_policy()

"""

Theory - Q-learning

In SARSA/TD0, we back-up our action values with the successor action value
In SARSA-max/Q learning, we back-up using the maximum action value.

"""

print(inspect.getsource(r.sarsa_lambda))

p = Piece(piece='king')
env = Board()
r = Reinforce(p, env)
r.q_learning(n_episodes=1000, alpha=0.2, gamma=0.9)

r.visualize_policy()

r.agent.action_function.max(axis=2).round().astype(int)
