# Represent the board with a NumPy int array, 0 = empty, 1 = red, 2 = blue

from dataclasses import dataclass
from referee.game.constants import BOARD_N
from referee.game.player import PlayerColor
from referee.game.coord import Coord
from referee.game.pieces import PieceType, _TEMPLATES
from referee.game import board
class Bitboard:

	# Create 11 row and 11 column masks for clearing rows and columns.
	# Creates here because they are the same for all board, and saves
	# computation time when creating new bitboards

	row_masks = [(1 << BOARD_N) - 1 << (i * BOARD_N) for i in range(BOARD_N)]
	#for i in range(BOARD_N):
	#	print(bin(row_masks[i]))

	col_masks = [0 for _ in range(BOARD_N)]
	
	for j in range(BOARD_N):
		for k in range(BOARD_N):
			col_masks[j] |= (1 << (j + k * BOARD_N))
		#print(bin(col_masks[j]))

	def __init__(
		self
	):
		self.red_board = 0
		self.blue_board = 0

		#print(len(bin(self.red_board)))
		# Size of the board, 121 bits
		self.total_bits = BOARD_N * BOARD_N

	# def clone(self):
    #     # Create a new instance of the Bitboard class
	# 	cloned_board = Bitboard()
	# 	cloned_board.red_board = self.red_board
	# 	cloned_board.blue_board = self.blue_board
    #     # Copy the relevant attributes from the original instance to the new one
    #     # For example:
    #     # cloned_board.attribute1 = self.attribute1
    #     # cloned_board.attribute2 = self.attribute2
    #     # ...
	# 	return cloned_board
	
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
		self,
		changed_indexes: list[int]
	):
		# Create the full board showing which tiles are filled in
		full_board = self.red_board | self.blue_board

		combined_masks = 0

		# Find which rows and cols have been affected by the new placement
		rows_to_check = set(index // BOARD_N for index in changed_indexes)
		cols_to_check = set(index % BOARD_N for index in changed_indexes)

		# Check to see if affected rows/cols are full
		#print(rows_to_check)
		for row in rows_to_check:
			
			row_mask = Bitboard.row_masks[row] # Precomputed row all set bits
			masked_row_board = full_board & row_mask
			# If row is full in the full board, then add it to the combined
			# mask we will use to clear rows/cols at the end
			#print(bin(full_board & row_mask))
			print("masked row board = " + str(bin(masked_row_board)))
			print("row mask = " + str(bin(row_mask)))
			if masked_row_board == row_mask:
				combined_masks |= row_mask

		for col in cols_to_check:

			col_mask = Bitboard.col_masks[col] # Precomputed col all set bits
			masked_col_board = full_board & col_mask
			# If col is full in the full board, then add it to the combined
			# mask we will use to clear rows/cols at the end
			#print(masked_col_board)
			#print(col_mask)

			if masked_col_board == col_mask:
				combined_masks |= col_mask

		# 'Overlay' the negative combined mask over the red and blue board,
		# turning off the bits that need to be cleared
		# Combined mask will the rows and cols that were full in the fill board
		# turned on, so it just maps all bits we need to turn off
		self.red_board &= ~combined_masks
		self.blue_board &= ~combined_masks

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
				print('R', end=' ')
			elif self.blue_board & (1 << i): # On bit in the blue board
				print('B', end=' ')
			else: # No on bit for current index 
				print('.', end=' ')
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

	def get_hash(
		self
	):
		return hash((self.red_board, self.blue_board))

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


#def Bitboard_to_OG(
#	board: Bitboard
#) -> Board:
#	result = Board()
#	for i in range(BOARD_N ** 2):
#		if board.red_board & (1 << i): # On bit in the red board
#			row = i//11
#			col = i % 11
#			result[(row,col)] == PlayerColor.RED
#		elif self.blue_board & (1 << i): # On bit in the blue board
#			row = i//11
#			col = i % 11
#			result[(row,col)] == PlayerColor.Blue

#	return result


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