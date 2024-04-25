# Class to represent a placement on the board

from dataclasses import dataclass

from .simple_board import Simple_Board
from referee.game.pieces import Piece, PieceType, _TEMPLATES
from referee.game.coord import Coord

class Placement:

	# To save computations, maybe if a move is valid, we just place it?
	# Feels like double handling if we test if a move, then create the move
	# again to place it
	def is_valid_move(
		board: Simple_Board,
		piece_type: PieceType,
		origin: Coord
	) -> bool:

		# Don't need to check if we are placing in a valid 'adjacent' tile,
		# because these are the only tiles we will try and place in
		
		# TODO Add board wrapping to this?
		# Check if all spaces in the shape are empty
		for offset in _TEMPLATES[piece_type]:
			if (board.check_square(origin + offset)):
				return False
		# If all 4 squares are empty, the move is valid
		return True

	
