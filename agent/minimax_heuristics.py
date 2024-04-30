from simple_board import Simple_Board
from typing import Tuple
import numpy as np

"""
Minimax heuristic function - 1

Factors Considered:
	- Red vs Blue Ratio

Notes:
Scalar constant of this function should increase close to the end of the game.
Meaning that late game, we prioritise placing pieces to have most tiles in end.
However, we might need two different functions for this, because I suspect
red vs blue ratio will be important early game too. So we don't want a low
multiplier for early game and thus the factor is not really considered
"""
def v1_minimax_heuristic(
	simple_board: Simple_Board
) -> float:

	# Counts will have the amount of 0's, 1's and 2's. Could possibly use 0's
	# to aid calculations?
	counts = colour_count(simple_board)
	#num_of_empty = counts[0]
	num_of_red = counts[1]
	num_of_blue = counts[2]

	# If it is blue's turn, we want to calculate blue over red
	# If it is red's turn, we want to calculate red over blue
	# `best` state using this heuristic will just have the highest value
	if simple_board.is_blue_turn:
		return num_of_blue / num_of_red
	else:
		return num_of_red / num_of_blue


"""
Minimax heuristic function - 2

Factors Considered:
	- Approx opponent branching factor

Notes:
For benefit, need to find balance between speed and correctness

"""
def v2_minimax_heuristic():
	return 0


"""
Minimax heuristic function - 3

Factors Considered:
	- Exact opponent branching factor

Notes:
Way more computationally expensive than v2, but should give a better result
"""
def v3_minimax_heuristic():
	return 0

"""
Minimax heuristic function - 4

Factors Considered:
	- Agent approx branching factor 

Notes:
This is considering branching factor before opponent moves again

"""

def v4_minimax_heuristic():
	return 0

"""
Minimax heuristic function - 5

Factors Considered:
	- Exact agent branching factor

Notes:
Calculated before opponent makes moves

"""
def v5_minimax_heuristic():
	return 0


"""
Minimax heuristic function - 6

Factors Considered:
	- Amount of rows and columns filled

Notes:
The max amount of rows/cols filled is 11. The more the number, the greater
resistance to being completely wiped out by row/col deletion
"""
def v6_minimax_heuristic():
	return 0







# HELPER FUNCTIONS

# Pass in a board, and get the number of empty, red and blue squares
def colour_count(
	simple_board: Simple_Board
) -> np.array:
	
	# Flatten to 1D array, then count occurances of 0,1 and 2.
	flat_board = simple_board.board.ravel()
	counts = np.bincount(flat_board,minlength=3)

	# Will return a np.array with first value being the count of 0's, then 1's,
	# then 2's
	return counts
	
	