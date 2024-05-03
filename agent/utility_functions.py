from Bitboard import Bitboard
from referee.game.constants import BOARD_N

# Utility functions for minimax.
# Idea is to have multiple versions of the utility functions, then we can 
# combine these in a larger function with a multiplier for each function.

# Eg, if we want our utility function to be 
# 2*ratio_redvsblue + 3*opp_branch_factor,
# we could use utility = 2*v1_minimax_util + 3*v2_minimax_util

"""
Minimax utility function - 1


Factors Considered:
	- Red vs Blue Ratio

Notes:
Scalar constant of this function should increase close to the end of the game.
Meaning that late game, we prioritise placing pieces to have most tiles in end.
However, we might need two different functions for this, because I suspect
red vs blue ratio will be important early game too. So we don't want a low
multiplier for early game and thus the factor is not really considered
"""
def v1_minimax_util(
	bitboard: Bitboard,
	is_blue_turn: bool
) -> float:

	# Get the number of red tiles and the number of blue tiles using bitboard
	# count_tiles function
	red_counts = bitboard.count_tiles(bitboard.red_board)
	blue_counts = bitboard.count_tiles(bitboard.blue_board)
	
	# Return calling_players_tilecount/opp_player_tilecount. The higher the
	# number, the better move it is
	if is_blue_turn:
		return blue_counts/red_counts
	else:
		return red_counts/blue_counts


"""
Minimax utility function - 2

Factors Considered:
	- Approx opponent branching factor

Notes:
For benefit, need to find balance between speed and correctness

"""
def v2_minimax_util():
	return 0


"""
Minimax utility function - 3

Factors Considered:
	- Exact opponent branching factor

Notes:
Way more computationally expensive than v2, but should give a better result
"""
def v3_minimax_util():
	return 0

"""
Minimax utility function - 4

Factors Considered:
	- Agent approx branching factor 

Notes:
This is considering branching factor before opponent moves again

"""

def v4_minimax_util():
	return 0

"""
Minimax utility function - 5

Factors Considered:
	- Exact agent branching factor

Notes:
Calculated before opponent makes moves

"""
def v5_minimax_util():
	return 0


"""
Minimax utility function - 6

Factors Considered:
	- Amount of rows and columns filled

Notes:
The max amount of rows/cols filled is 11. The more the number, the greater
resistance to being completely wiped out by row/col deletion.

Big note: This is a greedy approach and not actually an optimal solution. It
turns out that this is basically a 'minimal cover' problem, which is 
NP-Complete. Will be too inefficient to calculate the exact amount, so we will
just approximate
"""
def v6_minimax_util(
	bitboard: Bitboard,
	is_blue_turn: bool
) -> int:

	# Pick the board depending on the player
	if is_blue_turn:
		board = bitboard.blue_board
	else:
		board = bitboard.red_board

	rows_needed = set()
	cols_needed = set()

	# Check if there is at least 1 set bit in each row
	for row in range(BOARD_N):
		row_mask = ((1 << BOARD_N) - 1) << (row * BOARD_N)
		if board & row_mask != 0:
			rows_needed.add(row)

	# Check if there is at least 1 set bit in each column
	for col in range(BOARD_N):
		col_mask = 0
		for row in range(BOARD_N):
			col_mask |= (1 << (row * BOARD_N + col))
		if board & col_mask != 0:
			cols_needed.add(col)

	min_clears_needed = min(len(rows_needed), len(cols_needed))

	return min_clears_needed

	



	
	