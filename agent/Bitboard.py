# Represent the board with a NumPy int array, 0 = empty, 1 = red, 2 = blue

from dataclasses import dataclass
from referee.game.constants import BOARD_N
from referee.game.player import PlayerColor
from referee.game.coord import Coord
class Bitboard:

	def __init__(
		self
	):
		self.red_board = 0
		self.blue_board = 0
		self.full_mask = (1 << BOARD_N) - 1 # Used for clearing rows/cols
		
		# Size of the board, 121 bits
		self.total_bits = BOARD_N * BOARD_N
	
	"""
	Input:
		`index` - int representing a tile on the board
		`colour` - A PlayerColor depending on which players turn it is
	
	Output: No return, but alters the players board
	
	Desc: Changes a given tile from 0 to 1 on the players own board
	"""
	def set_tile(
		self,
		index: int,
		colour: PlayerColor
	):
		# On the players board, bitwise OR their board with a a mask of
		# a board with only the index tile switched on
		if colour == PlayerColor.RED:
			self.red_board |= (1 << index)
		elif colour == PlayerColor.BLUE:
			self.blue_board |= (1 << index) 

	"""
	Input:
		`index` - int representing a tile on the board

	Output: No return, but alters both players boards

	Desc: Changes a given tile from 1 to 0 on both boards. Due to the fact that
		placed pieces cannot overlap, only 1 player will have the indexed bit
		turned on. However, rather than doing an if to figure out which players
		turn it is, we can just turn this bit off in both boards
		"""
	def clear_tile(
		self,
		index: int
	):
		# Turn off the index bit in both boards
		self.red_board &= ~(1 << index)
		self.blue_board &= ~(1 << index)

	"""
	Input:
		`index` - int representing a tile on the board

	Output: Returns a PlayerColor type, or None if tile is empty

	Desc: Checks an indexed tile, and returns the PlayerColor if the tile is
		not empty. If the tile is empty, returns None
		"""
	def get_tile(
		self,
		index: int
	) -> PlayerColor | None:
		
		# If there is something in a tile, it will either be in the red or blue
		# board, but not both. If neither board contains a turned on bit at the
		# index, it is empty so we return None
		if self.red_board & (1 << index):
			return PlayerColor.RED
		elif self.blue_board & (1 << index):
			return PlayerColor.BLUE
		return None

	"""
	Input:
	
	Output: Returns a direct copy of the bitboard
	
	Desc: Takes a bitboard, and returns a duplicate of it. This duplicate can be
		modified without changing the parent bitboard
	"""
	def copy(
		self
	):
		new_board = Bitboard()
		new_board.red_board = self.red_board
		new_board.blue_board = self.blue_board
		return new_board
	
	"""
	Input: 
		`coord` - A Coord tpye, with row,column
		
	Output: An int representing a tile in the board
	
	Desc: Quality of life functin to convert a Coord to a tile in the bitboard.
		Useful as the PlaceActions from the referee are still in Coord type
	"""
	def get_index_from_coord(
		coord: Coord
	) -> int:
		return coord.r * BOARD_N + coord.c

	
	"""
	Input:
		
	Output: No output, but modifies the bitboard that is calling the function
	
	Desc: Checks over a bitboard to see if any rows or columns are filled. If
		they are filled, then it will set all the values back to 0
	"""
	def check_clear_filled_rowcol(
		self
	):
		# New bitboard with 1 for occupied tiles and 0 for empty
		full_board = self.red_board | self.blue_board
		rows_to_clear = []
		cols_to_clear = []

		# Check each row to see if it is full. If it is, append that mask
		for i in range(BOARD_N):
			row_mask = self.full_mask << (i * BOARD_N)
			if (full_board & row_mask) == row_mask:
				rows_to_clear.append(row_mask)

		# Check each col to see if it is full. More difficult because a full col
		# spans over all 11 rows. If full, append mask
		for j in range(BOARD_N):
			col_mask = 0
			for k in range(BOARD_N):
				col_mask |= (1 << (j + k * BOARD_N))
			if (full_board & col_mask) == col_mask:
				cols_to_clear.append(col_mask)

		# A mask means that some row/col is full. For each mask, use that mask
		# to change the value of each bit in the row/col to 0
		for mask in rows_to_clear + cols_to_clear:
			self.red_board &= ~mask
			self.blue_board &= ~mask

	"""
	Input:
	
	Output: No output, but prints into terminal
	
	Desc: Prints out a visual representation of both bitboards joined together
	"""
	def bitboard_display(
		self
	): 

		for i in range(BOARD_N ** 2):
			if self.red_board & (1 << i): # On bit in the red board
				print('R', end='')
			elif self.blue_board & (1 << i): # On bit in the blue board
				print('B', end='')
			else: # No on bit for current index 
				print('.', end='')
			if (i + 1) % BOARD_N == 0:
				print()
	"""
	Input:
		`bitboard` - A bitboard, either red or blue
		
	Output: Returns an int corresponding to the number of filled tiles in a 
		bitboard
		
	Desc: Uses Brian Kernighan's algorithm to count bits in a bitboard.
	Kernighan's algorithm:
	https://yuminlee2.medium.com/brian-kernighans-algorithm-count-set-bits-in-a-number-18ab05edca93#:~:text=The%20idea%20is%20to%20subtract,be%20the%20set%20bit%20count.
	"""
	def count_tiles(
		self,
		bitboard
	) -> int:
		
		count = 0
		while bitboard:
			bitboard &= bitboard - 1
			count += 1

		return count