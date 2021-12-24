import gym
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

"""
Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. 
Adam combines the best properties of theAdaGrad and RMSProp algorithms to provide an optimization algorithm that 
can handle sparse gradients on noisy problems.
"""
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# gym (developed by OpenAI) is used for the environment and we need random for the random action our agent is going to perform.
# Environment setup
env = gym.make('CartPole-v0')
# 4 states available
states = env.observation_space.shape[0]
# the actions that this Agent (card holder) can perform are 2 , go left or go right
actions = env.action_space.n

# Now lets see how the Agent performs with no training for 10 episodes
# On average our maximum score that we cen get is around 38 , which is quite low :(
episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.choice([0, 1])
        n_sates, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))


# So in order to increase the score we need to build our Deep Learning Model:
# The model that we built requires 2 arguments: the states (4) and the actions (2) that we extracted

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
# Lets see how the model looks like
model.summary()


# Now the next step is to built the Agent, for that we will make use of the DQNAgent from rl.agents
# There are value based and policy based RL methods, for our case we will use the policy from BoltzmannQPolicy, which is a Q - learning based policy

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10,
                   target_model_update=1e-02)
    return dqn


# Time for training , set it to 500000 steps and learning rate of 0.001 (1e-3)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Now you can see that we reach a maximum score of 200, much better than before
scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

_ = dqn.test(env, nb_episodes=15, visualize=True)

# And this is how we can save the weights of the model for future use
dqn.save_weights('dqn_weights.h5f', overwrite=True)
