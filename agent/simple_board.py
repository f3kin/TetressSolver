# Represent the board with a NumPy int array, 0 = empty, 1 = red, 2 = blue

from dataclasses import dataclass

from ..referee.game.pieces import Piece, PieceType, create_piece
from ..referee.game.coord import Coord, Direction
from ..referee.game.player import PlayerColor
from ..referee.game.actions import Action, PlaceAction
from ..referee.game.exceptions import IllegalActionException
from ..referee.game.constants import *
import numpy as np

class simple_board:

	"""
	In simple_board, each square has either 0, 1 or 2.
		0 - Empty square
		1 - Red square
		2 - Blue square
	"""

	# TODO Big time todo, check that we order coords in r,c and not c,r

	def __init__(
		self
	):
		self.board = np.zeros((11, 11), dtype=np.int8)
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
		new_board = simple_board()
		new_board.board = np.copy(self.board)

		# TODO Invert this to save switching later?
		new_board.is_blue_turn = self.is_blue_turn	
		return new_board
		


