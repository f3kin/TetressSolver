# Represent the board with a NumPy int array, 0 = empty, 1 = red, 2 = blue

from dataclasses import dataclass

from .pieces import Piece, PieceType, create_piece
from .coord import Coord, Direction
from .player import PlayerColor
from .actions import Action, PlaceAction
from .exceptions import IllegalActionException
from .constants import *
import numpy as np

class new_board:

	def __init__(
		self,
		initial_player: PlayerColor = PlayerColor.RED
	):
		self.board = np.zeros((11, 11), dtype=int)
		self._turn_color: PlayerColor = initial_player
		

	# THREE MAIN FUNCTIONS ARE:
		 
	# Need to be able to copy a board
 
	# Rendering the board would be useful too. Maybe just a need to be able to convert new board into old board for these functions
 
	"""
	Desc: Function takes row and col and returns the value stored there in board
	Input: 
		`row`: An int corresponding to the row number
		`col`: An int corresponding to the col number
	Output: The int8 value stored at the coordinates [0,1,2]

	"""
	def get_square(
		self,
		row: int,
		col: int
	) -> np.int8:
		return self.new_board[row][row]
	
	"""
	Desc: Change the value of a square on the board
	Input:
		`row`: An int corresponding to the row number
		`col`: An int corresponding to the col number
		`new_val`: The value you want to change the current square to
	Output: No output, but modifies the current board

	Notes: This will be used by another function. Will be called in 3 different ways.
		- Player1 places. Update empty square/s (0) to Player1 Color (eg (1))
		- Player2 places. Update empty square/s (0) to Player2 Color (eg (2))
		- For deletion, when either P1 or P2 delete a square, (1/2) to empty (0)
	"""
	def modify_square(
		self,
		row: int,
		col: int,
		new_val: np.int8
	) -> None:
		np.board[row][col] = new_val


	"""
	Desc: Takes the current board, and creates a deep copy of it for children to modify
	Input:
	Output: A deep copy of current board

	Notes: Could just be done with "BOARD".copy() but will be nicer to use a function
	"""

	#TODO Fix this
	def copy_board(
		self,
		board: new_board
	) -> new_board:
		return board.copy()
	
