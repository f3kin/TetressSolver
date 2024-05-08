# Represent the board with a NumPy int array, 0 = empty, 1 = red, 2 = blue

from dataclasses import dataclass
from referee.game.constants import BOARD_N
from referee.game.player import PlayerColor
from referee.game.coord import Coord
from referee.game.pieces import PieceType, _TEMPLATES
from referee.game import board
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
	
	"""
	Input:
		`colour` - A PlayerColor depending on which players turn it is
	Output: A list of ints

	Desc: Finds all on bits for a given players bitboard. Used to find where
		a player can place a piece
		"""
	def get_colour_indexes(
		self,
		color: PlayerColor
	) -> list[int]:
		
		indexes = []

		# Get the corresponding board of the current player
		if color == PlayerColor.RED:
			temp = self.red_board
		else:
			temp = self.blue_board

		# Check the last bit of the temp board. If it is 1, then append the
		# index. Then move temp right by 1 unit, and increase index
		index = 0
		while temp:
			if temp & 1:
				indexes.append(index)
			temp >>= 1
			index += 1
			
		return indexes


	"""
	Input:
		`tile_indexes` - A list of indexes representing the turned on bits in
			one of the bitboards
	Output: A set containing all the empty adjacent tiles to the tiles passed
		into the function
	Desc: Takes a list of indexes which represent where tiles are filled in one
		of the boards. No colour specifics here. If red is using this function, 
		we will be inputting reds tile indexes, and thus receive the adjacent
		tiles. Same goes for blue
	"""
	def get_adjacent_squares(
		self,
		tile_indexes: list[int]
	) -> set[int]:
		
		empty_adjacent_tiles = set()

		for index in tile_indexes:

			# Get left adjacent index
			offset = -1
			if (index % 3 == 0):
				offset = (BOARD_N -1)
			left_index = index + offset
			if self.get_tile(left_index) is None:
				empty_adjacent_tiles.add(left_index)

			# Get right adjacent index
			offset = 1
			if ((index + 1) % 3 == 0):
				offset = -(BOARD_N - 1)
			right_index = index + offset
			if self.get_tile(right_index) is None:
				empty_adjacent_tiles.add(right_index)

			# Get above adjacent index
			above_index = index - BOARD_N
			if (above_index < 0):
				above_index = BOARD_N**2 + above_index
			if self.get_tile(above_index) is None:
				empty_adjacent_tiles.add(above_index)


			# Get below adajcent index
			below_index = index + BOARD_N
			if (below_index >= BOARD_N**2):
				below_index = below_index - BOARD_N**2
			if self.get_tile(below_index) is None:
				empty_adjacent_tiles.add(below_index)

		return empty_adjacent_tiles

	def move_adj(
			self,
		index: int,
		move: str
	) -> int:
		
		# Calc 'row' and 'col' or given index
		row = index // BOARD_N
		col = index % BOARD_N

		if move == "right":
			# In case where index is on right most column, col 10
			if col == (BOARD_N - 1):
				new_index = row * BOARD_N
			# In case where not in the 10th col
			else:
				new_index = index + 1
		
		elif move == "left":
			# In case where index is on left most column, col 0
			if col == 0:
				new_index = row * BOARD_N + (BOARD_N - 1)
			# In case were index is not in the 0th col
			else:
				new_index = index - 1
			
		elif move == "down":
			# In case where index is on bottom row, 10th row
			if index + BOARD_N >= (BOARD_N **2):
				new_index = (index + BOARD_N) % BOARD_N
			# In case were index is not in the 10th row
			else:
				new_index = index + BOARD_N

		elif move == "up":
			# In case where index is on the top row, the 0th row
			if index < BOARD_N:
				new_index = (BOARD_N ** 2) - (BOARD_N - index)
			# In case where index is not on the 0th row
			else:
				new_index = index - BOARD_N

		return new_index



	"""
	Input:
		`c1,c2,c3,c1` - Coords in a grid, assuming already wrapped
		
	Output: No output, but modifies the board
	
	Desc: Takes 4 coordinates which have already been wrapped, and converts 
		them to indexes, before placing them on the players board.
	"""
	def place_four_tiles(
		self,
		colour: PlayerColor,
		c1: Coord,
		c2: Coord,
		c3: Coord,
		c4: Coord
	):
		# Take all 4 coords, convert them to indexes
		index1 = get_index_from_coord(c1)
		index2 = get_index_from_coord(c2)
		index3 = get_index_from_coord(c3)
		index4 = get_index_from_coord(c4)

		# Call set_tile 4 times with the 4 different indexes	
		self.set_tile(index1, colour)
		self.set_tile(index2, colour)
		self.set_tile(index3, colour)
		self.set_tile(index4, colour)

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


def Bitboard_to_OG(
	board: Bitboard
) -> Board:
	result = Board()
	for i in range(BOARD_N ** 2):
		if board.red_board & (1 << i): # On bit in the red board
			row = i//11
			col = i % 11
			result[(row,col)] == PlayerColor.RED
		elif self.blue_board & (1 << i): # On bit in the blue board
			row = i//11
			col = i % 11
			result[(row,col)] == PlayerColor.Blue

	return result


"""
------------------------------How does the bitboard work?-------------------------------
A bitboard has two boards, the red_board and blue_board.
A board is a binary int, where 1's represent a filled tile and 0 is empty.

So the board
B - R
R - -
R - B

would have
red_board =
0 0 1
1 0 0
1 0 0
which is
0 0 1 0 0 1 1 0 0
because we start on the right hand side and read left

and blue_board =
1 0 0
0 0 0
0 0 1
which is
1 0 0 0 0 0 0 0 1

To interact with a cell in the board, we use indexes. The index of a tile can
be calculated with 
r * BOARD_N + c

-----------------------------Check if there is something at index-----------------------
To check if there is a blue or red tile in a square, we use get_tile(index)

get_tile takes the index, say index 3 (4th tile)
it creates a new binary int 000000001, and then shifts the set bit to the left
by `index`. So
1 << index means:
000000001 left shift 3
= 000001000

It then takes this 1 << index, and does a bitwise AND with both red and blue boards.

first we check blue board
blue_board = 100000001
1 << index = 000001000
             |||||||||
			 000000000
The result is 0, so there is no set bit in blue board at index 3

now we check red_board
red_board  = 001001100
1 << index = 000001000
			 |||||||||
			 000001000
The result isn't 0, so there is a set bit in the red board. So we return playercolour.red

If neither board returned true, we would return None to signify no set tiles at the index


----------------------------------Placing a tile at an index -----------------------------
Laptop battery running out.
Same left shift thing as the check tile, but this time we bitwise OR the two binary ints

So anywhere with a 1 in the original board remains on
and the new index we are placing with the left shift also turns on in the orignal board.
We only perform this operation on the board of the colour we input.

Quality of life stuff:

I created a get_index_from_coord function so we dont have to manually calc indexes

There is a print board function too


Message me if you need any other help

"""