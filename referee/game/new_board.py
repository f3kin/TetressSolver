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
		
		