# Represent the board with a NumPy int array, 0 = empty, 1 = red, 2 = blue

from dataclasses import dataclass

from referee.game.pieces import Piece, PieceType, create_piece
from referee.game.coord import Coord, Direction
from referee.game.player import PlayerColor
from referee.game.actions import Action, PlaceAction
from referee.game.exceptions import IllegalActionException
from referee.game.constants import *
from referee.game.board import Board
import numpy as np

class Simple_Board:

	"""
	In simple_board, each square has either 0, 1 or 2.
		0 - Empty square
		1 - Red square
		2 - Blue square
	"""

	# TODO Change board size to 12x12, with the 12th element of each row and
	# col representing the number of elements in said row or col
	def __init__(
		self
	):
		self.board = np.zeros((12, 12), dtype=np.int8)
		self.is_blue_turn = False

	def check_square(
		self,
		position: Coord
	) -> np.int8:
		return self.board[position.r][position.c]

	def change_square(
		self,
		position: Coord,
		is_blue_turn: bool
	) -> None: 
		

		# Find who's turn it is and set the colour
		if (is_blue_turn):
			colour = 2
		else:
			colour = 1

		# If square being changed is not 0, (eg red-1 or blue-2), then 
		# change it to 0. Coloured squares being modified will only ever be
		# changed to empty
		if (self.board[position.r][position.c] > 0):
			self.board[position.r][position.c] = 0
		# If square being changed is empty, change it to the turn colour
		else:
			self.board[position.r][position.c] = colour

	def copy_board(
		self
	):		
		new_board = Simple_Board()
		new_board.board = np.copy(self.board)

		# TODO Invert this to save switching later?
		new_board.is_blue_turn = self.is_blue_turn	
		return new_board
	
	# Check any altered rows/cols to see if they are full. If they are,
	# set everything to 0
	# IDEA. Maybe we have a 12th element in each row/col whch is a counter of
	# how many empty tiles in a row/col. 
	# eg row 0: 0,1,1,2,2,0,2,0,0,0,0,5
	# Here, the 5 means there are 5 0's in the row
	# When we make a PlaceAcion, this would be simple to update. Then, 
	# deletion becomes a lot quicker
 
	# Better, the 12th number represents the number of filled squares. Then
	# we can initialise the board as all 0's. Also, clearing a row or col means
	# everything can just be set to 0.

	# When a row or col is full, eg tiles filled = 11, then pass in that with
	# rowcol_deletion(row=2) or rowcol_deletion(col=5)
	# Maybe pass in a list of rows and a list of cols that are full
	
	def rowcol_deletion(
		self,	
		rows=[None],
		cols=[None]
	)-> None:
		
		for row in rows:
			self.board[row].fill(0)

		for col in cols:
			for board_row in self.board:
				self.board[board_row][col] = 0
			


	# THis is really ugly. Gotta be a better way to do it
	# NOT SURE IF WE NEED THIS NOW I THINK ABOUT IT
	# I think the flow will just be using action()/update() to update the
	# referee board and visa versa
	def board_to_simple(
		board: Board
	)->Simple_Board:
		
		new_simple_board = Simple_Board()
		for r in range(BOARD_N):
			for c in range(BOARD_N):
				cell_state = board.__getitem__(Coord(r,c))
				if cell_state is None:
					new_simple_board[r][c] = 0
				elif cell_state is PlayerColor.RED:
					new_simple_board[r][c] = 1
					# BOARD_N + 1 index holds the num of pieces in the row/col
					# increase row and col counter by 1 for each piece
					new_simple_board[r][BOARD_N + 1] += 1
					new_simple_board[BOARD_N + 1][c] += 1
				elif cell_state is PlayerColor.BLUE:
					new_simple_board[r][c] = 2
					# BOARD_N + 1 index holds the num of pieces in the row/col
	 				# increase row and col counter by 1 for each piece
					new_simple_board[r][BOARD_N + 1] += 1
					new_simple_board[BOARD_N + 1][c] += 1
		return new_simple_board
	

	# Function should take a placement, and update the boards row and col 12
	# to hold the corrent nums for tiles in given rows and cols
	def increment_rowcol_piece_counter(
		
	):
		
		# Psuedo
  
		"""
		for tile_coord in Placement
			self.board[tile_coord.r][BOARD_N + 1] += 1
			self.board[BOARD_N + 1][tile_coord.c] += 1
		"""
		return 0

def v1_minimax_eval(
	simple_board:Simple_Board		
) -> float:

	# Some function adding opp branching factor + ratio of redvsblue tiles
	
	return 0