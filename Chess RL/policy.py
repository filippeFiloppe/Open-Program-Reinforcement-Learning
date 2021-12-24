# Chess Reinforcement Learning (for Open Programme)

# This notebook is based on an online example for Reinforcement Learning on Chess
# found in Kaggle : https://www.kaggle.com/arjanso/reinforcement-learning-chess-1-policy-iteration
# My goal is to try and run the training and see what happens :). This notebook is for my own educational purposes
# I would describe myself as an average Chess player, I think that learning about RL from an already well known for me game is
# a good and efficient way to approach this new subject.

import numpy as np  # linear algebra
import inspect

from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce
from RLC.capture_chess.environment import Board

# Policy Iteration

# The environment
# - The state space is a 8 by 8 grid
# - The starting state S is the top-left square (0,0)
# - The terminal state F is square (5,7).
# - Every move from state to state gives a reward of minus 1
# - Naturally the best policy for this environment is to move from S to F in the lowest amount of moves possible

env = Board()
env.render()

# The agent
# - The agent is a chess Piece (king, queen, rook, knight or bishop)
# - The agent has a behavior policy determining what the agent does in what state

p = Piece(piece='king')

# Reinforce
# - The reinforce object contains the algorithms for solving move chess
# - The agent and the environment are attributes of the Reinforce object

r = Reinforce(p, env)

# Now if we want our agent to optimize its rewards, the way to do it is by guiding his
# behavior towards the states with highest reward values. The values are estimated using
# bootstrapping:
# - A state (s) is as valuable (V) as the successor state (s') plus the reward (R) for going from s to s'.
# - Since there can be multiple actions (a) and multiple successor states they are summed and weighted by their probability (pi).
# - In a non-deterministic environment, a given action could result in multiple successor states.
#   We don't have to take this into account for this problem because move chess is a deterministic game.
# - Successor state values are discounted with discount factor (gamma) that varies between 0 and 1.
# - This gives us the following formula:

# Also note that:
# - The successor state value is also en estimate.
# - Evaluating a state is bootstrapping because you are making an estimate based on another estimate
# - In the code you'll see a synchronous parameter that will be explained later in the policy evaluation section

print(inspect.getsource(r.evaluate_state))

# Demonstration
# - The initial value function assigns value 0 to each state
# - The initial policy gives an equal probability to each action
# - We evaluate state (0,0)

r.agent.value_function.astype(int)

state = (0, 0)
r.agent.value_function[0, 0] = r.evaluate_state(state, gamma=1)

r.agent.value_function.astype(int)

print(inspect.getsource(r.evaluate_policy))

r.evaluate_policy(gamma=1)

r.agent.value_function.astype(int)

eps = 0.1
k_max = 1000
value_delta_max = 0
gamma = 1
synchronous = True
value_delta_max = 0

for k in range(k_max):
    r.evaluate_policy(gamma=gamma, synchronous=synchronous)
    value_delta = np.max(np.abs(r.agent.value_function_prev - r.agent.value_function))
    value_delta_max = value_delta
    if value_delta_max < eps:
        print('converged at iter', k)
        break

r.agent.value_function.astype(int)

print(inspect.getsource(r.improve_policy))

r.improve_policy()
r.visualize_policy()

print(inspect.getsource(r.policy_iteration))

r.policy_iteration()

agent = Piece(piece='king')
r = Reinforce(agent, env)

r.policy_iteration(gamma=1, synchronous=False)

r.agent.value_function.astype(int)

agent = Piece(piece='rook')  # Let's pick a rook for a change.
r = Reinforce(agent, env)
r.policy_iteration(k=1, gamma=1)  # The only difference here is that we set k_max to 1.
