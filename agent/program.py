# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent


# Depends entirely on if we can use deque
from collections import deque
from typing import Tuple, Optional
from agent.Bitboard import Bitboard

"""
Areas for improvement

    
Whole expand/minimax logic, we need to cache board states and use in next level
of minimax. MASSIVE MASSIVE TIME SAVE

Move ordering, when expanding, have some sort of measure for how good a move is,
so create that child first. This will save a lot of time in alpha, beta pruning

We also really need to think about killer moves, book moves (during whole game
eg moves you always make, like a checkmate in chess).



"""

OPENING = 3 # TODO Change this to another value
END_GAME = 75
DEPTH_VALUE = 5

DIRECTIONS = ["up", "down", "left", "right"]

from referee.game import PlayerColor, Action, PlaceAction, Coord
from agent.Bitboard import *


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Tetress game events.
    """
    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self.board = Bitboard()
        self._color = color
        self.num_moves = -1
        self.state = OPENING
        if self._color is PlayerColor.RED:
            self.book_moves = [PlaceAction(Coord(1,1), Coord(2,1), Coord(2,2), Coord(2,3)
                                ),PlaceAction(Coord(3,3), Coord(4,3), Coord(4,4), Coord(5,4)
                                ),PlaceAction(Coord(5,5), Coord(6,5), Coord(6,6), Coord(7,6)
                                ),PlaceAction(Coord(7,7), Coord(8,7), Coord(8,8), Coord(9,8)
                                ),PlaceAction(Coord(9,9), Coord(10,9), Coord(10,10), Coord(0,10)
                                )]
        else:
            self.book_moves = [PlaceAction(Coord(1,9), Coord(2,9), Coord(2,8), Coord(3,8)
                                ),PlaceAction(Coord(3,7), Coord(4,7), Coord(4,6), Coord(5,6)
                                ),PlaceAction(Coord(5,5), Coord(6,5), Coord(6,4), Coord(7,4)
                                ),PlaceAction(Coord(7,3), Coord(8,3), Coord(8,2), Coord(9,2)
                                ),PlaceAction(Coord(9,1), Coord(10,1), Coord(10,0), Coord(0,0)
                                )]
        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        self.num_moves += 1

        # If there are still possible book moves
        if self.book_moves:
            
            valid_move = None

            for move in self.book_moves:
                
                book_move_indexes = [get_index_from_coord(move.c1), 
                                    get_index_from_coord(move.c2),
                                    get_index_from_coord(move.c3),
                                    get_index_from_coord(move.c4)]
                
                if self.board.valid_book_move(book_move_indexes, self._color, self.num_moves):
                    valid_move = move

                    break

            if valid_move:
                self.book_moves.remove(valid_move)
                return valid_move

        # If there are no more book moves, move onto search
        return search(self.board, self._color, self.num_moves) 
        

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords

        changed_indexes = [get_index_from_coord(c1), 
                           get_index_from_coord(c2),
                           get_index_from_coord(c3),
                           get_index_from_coord(c4)]
        self.board.place_four_tiles(color, c1,c2,c3,c4)
        self.board.check_clear_filled_rowcol(changed_indexes)


def search(board, color, num_moves):
    if color == PlayerColor.RED:
        result = minimax(board, True, 0, float('-inf'), float('inf'), True, num_moves)
    else:
        result = minimax(board, False, 0, float('-inf'), float('inf'), True, num_moves)
    coords = get_coord_from_index(result[2])
    action = PlaceAction(coords[0], coords[1], coords[2], coords[3])
    return action


def get_coord_from_index(
    indexes: list[int]
) -> list[Optional[PlaceAction]]:
    result = []
    if indexes is not None:
        for i in range(4):
            result.append(Coord(indexes[i]//11, indexes[i]%11))
        return result



###### General Expansion logic (FROM EXPANSION BRANCH) ######

"""
Input: 
    `board` - A bitboard
    `color` - The color of the player calling

Output: A list of all possible children (after placing 1 tetromino) of a board

Desc: Takes a board and color, finds the tiles on the board of that color, and
    performs the tile expansion from each one
"""

def expand(
        board: Bitboard,
        isRed: bool,
    ) -> list[Bitboard]:

    moves = []
    visited = set()
    if isRed:
        player_tiles = board.get_colour_indexes(PlayerColor.RED)
    else:
        player_tiles = board.get_colour_indexes(PlayerColor.BLUE)

    for index in player_tiles:
        if index in visited: #Checks we havent expanded from here already
            continue
        all_index_placements = iterative_expand(board, index, isRed) 
        moves.extend(all_index_placements)
        visited.add(index) 
    
    return moves

def iterative_expand(
    board: Bitboard, 
    index: int, 
    isRed: bool
):

    queue = deque([(board, index, [index], int(1))])
    all_shapes = []
    seen_hashes = set()

    while queue:
        current_board, current_index, shape, depth = queue.popleft() 

        if depth == 5:
            
            current_board.check_clear_filled_rowcol(shape[1:]) 

            board_hash = current_board.get_hash()
            if board_hash not in seen_hashes:   
                seen_hashes.add(board_hash)
                all_shapes.append((current_board.copy(), shape[1:]))
            continue

        for direction in DIRECTIONS:
            new_index = current_board.move_adj(current_index, direction)
            if current_board.get_tile(new_index) is None and new_index not in shape:

                new_board = current_board.copy()
                new_board.set_tile(new_index, isRed)
                new_shape = shape + [new_index]
                queue.append((new_board, new_index, new_shape, depth + 1))

    return all_shapes

def minimax(
    board: Bitboard, 
    isRed: bool,
    depth: int, 
    alpha: int, 
    beta: int, 
    maximizingPlayer: bool, 
    num_moves: int,
    past = {}
) -> Tuple[Optional[int], Optional[Bitboard], Optional[list]]:

    board_key = board.get_hash()
    if board_key in past:
        return past[board_key]
    if maximizingPlayer:
        return max_value(board.copy(), isRed, depth, alpha, beta, num_moves, past)
    else:
        return min_value(board.copy(), isRed, depth, alpha, beta, num_moves, past)


def max_value(board, isRed, depth, alpha, beta, num_moves, past):
    maxEval = float('-inf')
    best_move = None
    best_coords = None
    if cutoff_test(depth, num_moves):
        eval_score = evaluation(board, isRed, v1_coefficient=10, v3_coefficient=5, v5_coefficient=5, v6_coefficient=2)
        return eval_score, board, None
    for child in expand(board, isRed): #returns a tuple (Bitboard, coords)
        eval_score, _, __ = minimax(child[0], not isRed, depth+1, alpha, beta, False, num_moves, past) 
        if eval_score is not None and eval_score > maxEval:
            maxEval = eval_score
            best_move = child[0]  
            best_coords = child[1]
        alpha = max(alpha, maxEval) 
        beta = min(beta, eval_score)
        if beta <= alpha:
            maxEval = beta
            break
    past[board.get_hash()] = maxEval, best_move, best_coords
    return maxEval, best_move, best_coords


def min_value(board, isRed, depth, alpha, beta, num_moves, past):
    minEval = float('inf')
    best_move = None
    best_coords = None
    if cutoff_test(depth, num_moves):
        eval_score = evaluation(board, isRed, v1_coefficient=10, v3_coefficient=5, v5_coefficient=5, v6_coefficient=2)
        return eval_score, board, None
        
    for child in expand(board, isRed):
        eval_score, _, __ = minimax(child[0], not isRed, depth+1, alpha, beta, True,num_moves, past)
        if eval_score is not None and eval_score < minEval:
            minEval = eval_score
            best_move = child[0]
            best_coords = child[1]
        alpha = max(alpha, minEval) 
        beta = min(beta, eval_score)
        if beta <= alpha:
            minEval = alpha
            break
    past[board.get_hash()] = minEval, best_move, best_coords
    return minEval, best_move, best_coords


def cutoff_test(depth, num_moves):
    if num_moves < 10:
        comp = 1
    else: 
        comp = DEPTH_VALUE
    if depth > comp:  
        return True



"""
Input:
    `board` - A bitboard
    `index` - An int, relating to one of the boards tiles
    `player_colour` - The PlayerColour type of the current agent
    `depth` - In 0-4. Represents the number of tiles placed 
    `current_shape` - The current formation of tiles placed in the expand.
                      Will either be a full shape (at depth 4), or partial
                      shape (at depths 1-3)
    `all_shapes` - The set of shapes created by the expanding. Only full shapes

Output: No output, but modifies the all_shapes set

Desc: Takes an index, and starts expanding outwards in all directions. Adds 
      searches of depth 4 to the all_shapes set and returns it
      So the return of this function contains all possible board configs after
      expanding from 1 of the players tiles

"""
def expand_out_sexy_style(
    board: Bitboard,
    index: int,
    player_colour: PlayerColor,
    depth: int,
    current_shape: list[int], 
    all_shapes: list[(Bitboard, list[int])],
    seen_hashes: set
):
    # Add all of the boards of depth 4 and return
    if depth == 5:
        
        # See if piece fills up rows or columns and delete them
        board.check_clear_filled_rowcol(current_shape[1:])
        # Hash the board and check for duplicates. If none, add the board and
        # shape as a child
        board_hash = board.get_hash()
        if board_hash not in seen_hashes:
            seen_hashes.add(board_hash)
            all_shapes.append((board, current_shape[1:]))
        return
    

    for direction in DIRECTIONS:

        # Move out 1 in each direction from the given index to get new index
        new_index = board.move_adj(index, direction)

        # If new_index is empty, we set the tile, update the shape, copy the
        # board to get a new board, then clean up the original board
        if board.get_tile(new_index) is None and new_index not in current_shape:

            new_board = board.copy()
            new_board.set_tile(new_index, player_colour)
            new_shape = current_shape + [new_index]
            expand_out_sexy_style(new_board, new_index, player_colour, depth + 1, new_shape, all_shapes, seen_hashes)


"""
Inputs:
    `board`
    `start_index`
    `player_colour`
    
Output: Returns all possible shapes made expanding from one tile

Desc: Function will be called on each of the players tiles in the 'expand'
    function. Called onces for each of the tiles. At the end, all possible
    children boards from a given board will exist.
"""
def init_expand_from_tile(
    board: Bitboard, 
    start_index: int,
    player_colour: PlayerColor
) -> list[Bitboard]:

    
    return iterative_expand(board, start_index, player_colour)
        

def evaluation(
    board: Bitboard, 
    isRed: bool,
    v1_coefficient: int,
    v3_coefficient: int,
    v5_coefficient: int,
    v6_coefficient: int
) -> float:

    is_blue_turn = not isRed
    red_blue_ration = v1_minimax_util(board, is_blue_turn)
    opp_branching_factor = v3_minimax_util(board, is_blue_turn)
    player_branching_factor = v5_minimax_util(board, is_blue_turn)
    rows_cols_filled = v6_minimax_util(board, is_blue_turn)

    goodness = (v1_coefficient * red_blue_ration +
                v3_coefficient * opp_branching_factor +
                v5_coefficient * player_branching_factor +
                v6_coefficient * rows_cols_filled)
    return goodness


def endgame_search(board, color):
    return search(board,color)


# Utility functions
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
		return (blue_counts/red_counts)/(BOARD_N)
	else:
		return (red_counts/blue_counts)/(BOARD_N)



"""
Minimax utility function - 3

Factors Considered:
	- Exact opponent branching factor

Notes:
Way more computationally expensive than v2, but should give a better result
"""
def v3_minimax_util(
	board: Bitboard,
	is_blue_turn: bool
) -> float: 
	
	from agent.program import expand

	
	is_red_turn = not is_blue_turn

	possible_moves = expand(board, is_red_turn)
	return len(possible_moves)/2500

"""
Minimax utility function - 5

Factors Considered:
	- Exact agent branching factor

Notes:
Calculated before opponent makes moves

"""
def v5_minimax_util(
	board: Bitboard,
	is_blue_turn: bool
) -> float:
	
	from agent.program import expand

	
	possible_moves = expand(board, is_blue_turn)

	return len(possible_moves)/2500

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
	return min_clears_needed / BOARD_N