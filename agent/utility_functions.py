from agent.Bitboard import Bitboard
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


	# Need to normalise this value. I suppose the maximum ratio would be 121:0,
	# so for now we will divide by 121
	if is_blue_turn:
		return (blue_counts/red_counts)/(BOARD_N **2)
	else:
		return (red_counts/blue_counts)/(BOARD_N ** 2)



"""
Minimax utility function - 2

Factors Considered:
	- Oppoent next move branching factor
"""
def v2_minimax_util(
	board: Bitboard,
	seen_states: dict
) -> int:
	

	# Doesn't work. Never enters the if statement
	board_hash = board.get_hash()
	if board_hash in seen_states:
		return seen_states[board_hash]['branching_factor']
	return 0

"""
Minimax utility function - 4

Factors Considered:
	- Agent next move branching factor

Notes:
Calculated before opponent makes moves

"""
def v3_minimax_util():
	return 0


"""
Minimax utility function - 4

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
def v4_minimax_util(
	bitboard: Bitboard,
	is_blue_turn: bool
) -> float:

	# Pick the board depending on the player
	board = bitboard.blue_board if is_blue_turn else bitboard.red_board

	rows_needed = 0
	cols_needed = 0

	# Check if there is at least 1 set bit in each row
	for row in range(BOARD_N):
		row_mask = Bitboard.row_masks[row]
		if board & row_mask != 0:
			rows_needed += 1

	# Check if there is at least 1 set bit in each column
	for col in range(BOARD_N):
		col_mask = Bitboard.col_masks[col]
		if board & col_mask != 0:
			cols_needed += 1
	min_clears_needed = min(rows_needed, cols_needed)

	# Normalise and return

	return min_clears_needed / BOARD_N

	



	
	