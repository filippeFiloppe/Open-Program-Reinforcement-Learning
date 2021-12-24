import pandas as pd
from policy import *
from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import Q_learning
from RLC.capture_chess.agent import Agent

"""
The environment: Capture Chess
In this python file we'll upgrade our environment to one that behaves more like real chess. It is mostly based on the Board object from python-chess. 
Some modifications are made to make it easier for the algorithm to converge:

- There is a maximum of 25 moves, after that the environment resets
- Our Agent only plays white
- The Black player is part of the environment and returns random moves
- The reward structure is not based on winning/losing/drawing but on capturing black pieces:
pawn capture: +1
knight capture: +3
bishop capture: +3
rook capture: +5
queen capture: +9
- Our state is represent by an 8x8x8 array
Plane 0 represents pawns
Plane 1 represents rooks
Plane 2 represents knights
Plane 3 represents bishops
Plane 4 represents queens
Plane 5 represents kings
Plane 6 represents 1/fullmove number (needed for markov property)
Plane 7 represents can-claim-draw
- White pieces have the value 1, black pieces are minus 1

"""

board = Board()

# Change the index of the first dimension to see the other pieces

board.layer_board[0, ::-1, :].astype(int)

"""

The Agent
The agent is no longer a single piece, it's a chess player
Its action space consist of 64x64=4096 actions:
There are 8x8 = 64 piece from where a piece can be picked up
And another 64 pieces from where a piece can be dropped.
Of course, only certain actions are legal. Which actions are legal in a certain state is part of the environment 
(in RL, anything outside the control of the agent is considered part of the environment).
We can use the python-chess package to select legal moves. 
(It seems that AlphaZero uses a similar approach
https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work)
  
"""

agent = Agent(network='conv', gamma=0.1, lr=0.07)
R = Q_learning(agent, board)
R.agent.fix_model()
R.agent.model.summary()
print(inspect.getsource(agent.network_update))

"""

Theory - Q learning with a Q-network

The Q-network is usually either a linear regression or a (deep) neural network.
The input of the network is the state (S) and the output is the predicted action value of each Action (in our case, 4096 values).
The idea is similar to learning with Q-tables. We update our Q value in the direction of the discounted reward + the max successor state action value
I used prioritized experience replay to de-correlate the updates. If you want to now more about it, check the link in the references
I used fixed-Q targets to stabilize the learning process.

"""

print(inspect.getsource(R.play_game))
pgn = R.learn(iters=750)



reward_smooth = pd.DataFrame(R.reward_trace)
reward_smooth.rolling(window=125, min_periods=0).mean().plot(figsize=(16, 9),
                                                             title='average performance over the last 125 steps')

with open("final_game.pgn", "w") as log:
    log.write(str(pgn))

board.reset()
bl = board.layer_board
bl[6, :, :] = 1 / 10  # Assume we are in move 10
av = R.agent.get_action_values(np.expand_dims(bl, axis=0))

av = av.reshape((64, 64))

p = board.board.piece_at(20)  # .symbol()

white_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
black_piece = ['_', 'p', 'n', 'b', 'r', 'q', 'k']

df = pd.DataFrame(np.zeros((6, 7)))

df.index = white_pieces
df.columns = black_piece

for from_square in range(16):
    for to_square in range(30, 64):
        from_piece = board.board.piece_at(from_square).symbol()
        to_piece = board.board.piece_at(to_square)
        if to_piece:
            to_piece = to_piece.symbol()
        else:
            to_piece = '_'
        df.loc[from_piece, to_piece] = av[from_square, to_square]

var = df[['_', 'p', 'n', 'b', 'r', 'q']]
