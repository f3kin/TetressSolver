# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent


# Depends entirely on if we can use deque
from collections import deque
from typing import Tuple, Optional


"""
Areas for improvement

    
Whole expand/minimax logic, we need to cache board states and use in next level
of minimax. MASSIVE MASSIVE TIME SAVE

Move ordering, when expanding, have some sort of measure for how good a move is,
so create that child first. This will save a lot of time in alpha, beta pruning

We also really need to think about killer moves, book moves (during whole game
eg moves you always make, like a checkmate in chess).



"""





MID_GAME = 70
OPENING = 3 # TODO Change this to another value
END_GAME = 75
MAX_TURN = 0
MIN_TURN = 1
DEPTH_VALUE = 3

DIRECTIONS = ["up", "down", "left", "right"]

from referee.game import PlayerColor, Action, PlaceAction, Coord
from agent.Bitboard import *
from agent.utility_functions import *


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
        # TODO: IMPLEMENT PRECOMPUTATION HERE
        self.board = Bitboard()
        self._color = color
        self.num_moves = -1
        self.state = OPENING
        if self._color is PlayerColor.RED:
            # Create these book moves. Try and place these moves on first 3 turns. If no moves can be made, then need to do normal search
            self.book_moves = [PlaceAction( #TODO: make these moves legitimate, will have to be slightly more complex, i.e. first red is open, blue is always a response to the red book move
                        Coord(0, 0), 
                        Coord(1, 0), 
                        Coord(2, 0), 
                        Coord(3, 0)
                    ), PlaceAction(
                        Coord(3, 1), 
                        Coord(4, 1), 
                        Coord(5, 1), 
                        Coord(6, 1)
                    ), PlaceAction(
                        Coord(6, 2), 
                        Coord(7, 2), 
                        Coord(8, 2), 
                        Coord(9, 2)
                    )]
        else:
            self.book_moves = [PlaceAction( #TODO: make these moves legitimate, will have to be slightly more complex, i.e. first red is open, blue is always a response to the red book move
                        Coord(0, 10), 
                        Coord(1, 10), 
                        Coord(2, 10), 
                        Coord(3, 10)
                    ), PlaceAction(
                        Coord(3, 9), 
                        Coord(4, 9), 
                        Coord(5, 9), 
                        Coord(6, 9)
                    ), PlaceAction(
                        Coord(6, 8), 
                        Coord(7, 8), 
                        Coord(8, 8), 
                        Coord(9, 8)
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

        # If there are still book moves available, play them and remove
        # the move from bookmoves. 
        if self.book_moves:
            return self.book_moves.pop(0)
        elif self.num_moves < END_GAME:
            return search(self.board, self._color) 
        else:
            return endgame_search(self.board, self._color)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords
        self.board.place_four_tiles(color, c1,c2,c3,c4)

        #print(f"Testing: {color} played PLACE action: {c1}, {c2}, {c3}, {c4}")

def search(board, color):
    # Minimax goes here
    result = minimax(board, color, 0, float('-inf'), float('inf'), True)
    coords = get_coord_from_index(result[2])
    action = PlaceAction(coords[0], coords[1], coords[2], coords[3])
    return action


def get_coord_from_index(
    indexes: list[int]
) -> list[Optional[PlaceAction]]:
    result = []
    #print(indexes)
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
        color: PlayerColor
    ) -> list[Bitboard]:

    moves = []

    # player_tiles is the list of indexes corresponding to the players tile
    if color == PlayerColor.RED:
        player_tiles = board.get_colour_indexes(PlayerColor.RED)
    else:
        player_tiles = board.get_colour_indexes(PlayerColor.BLUE)
        
    # For each tile, expand it
    for index in player_tiles:
        visited = {index}  #TODO: Implement set functionality
        all_index_placements = init_expand_from_tile(board, index, color)
        moves.extend(all_index_placements)
        
    # TODO Look at this
    # Sort children by some heuristic estimation before returning
    #children.sort(key=lambda x: heuristic_estimate(x))
    
    #print(moves)
    return moves


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

def iterative_expand(
    board: Bitboard, 
    index: int, 
    player_colour: PlayerColor
):

    queue = deque([(board, index, [index], 1)])
    all_shapes = []
    seen_hashes = set()

    while queue:
        current_board, current_index, shape, depth = queue.popleft()

        if depth == 5:

            current_board.check_clear_filled_rowcol(shape)
            board_hash = current_board.get_hash()
            if board_hash not in seen_hashes:
                seen_hashes.add(board_hash)
                all_shapes.append((current_board.copy(), shape[1:]))
            continue

        for direction in DIRECTIONS:
            new_index = current_board.move_adj(current_index, direction)
            if current_board.get_tile(new_index) is None and new_index not in shape:
                current_board.set_tile(new_index, player_colour)
                new_shape = shape + [new_index]
                queue.append((current_board, new_index, new_shape, depth + 1))
                current_board.clear_tile(new_index)

    return all_shapes



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

    seen_hashes = set()
    
    all_shapes = iterative_expand(board, start_index, player_colour)
    #all_shapes = []
    #expand_out_sexy_style(board, start_index, player_colour, 1, [start_index], all_shapes, seen_hashes)
    return all_shapes #[1::]
        




###### Functions specific to Minimax #####

#TODO: develop a better way of saving + storing child nodes, use acutal board type
# class Move:
#     def __init__(self, value:int, board:board) -> None:
#         self.value = value
#         self.board = board
    
def minimax(
    board: Bitboard, 
    color: PlayerColor,
    depth: int, 
    alpha: int, 
    beta: int, 
    maximizingPlayer: bool, 
    past = {}
) -> Tuple[Optional[int], Optional[Bitboard], Optional[list]]:
    # Add a call to the evaluate function
    if cutoff_test(board, depth):
        eval_score = evaluation(board, color)
        return eval_score, None, None
        
    # Check if the board state has been visited before
    board_key = hash(board)
    if board_key in past:
        return past[board_key]

    if maximizingPlayer:
        return max_value(board, color, depth, alpha, beta, past)
    else:
        return min_value(board, color, depth, alpha, beta, past)

def max_value(board, color, depth, alpha, beta, past):
    maxEval = float('-inf')
    best_move = None
    best_coords = None
    for child in expand(board, color):
        eval_score, _, coords = minimax(child[0], color, depth+1, alpha, beta, False, past)
        if eval_score is not None and eval_score > maxEval:
            maxEval = eval_score
            best_move = child[0]  # Update the best move
            best_coords = child[1]
        alpha = max(alpha, eval_score or alpha)  # Use `alpha` if `eval_score` is None
        if beta <= alpha:
            break
    # Store maxEval, best_move in past
    past[hash(board)] = maxEval, best_move, best_coords
    return maxEval, best_move, best_coords

def min_value(board, color, depth, alpha, beta, past):
    minEval = float('inf')
    best_move = None
    best_coords = None
    for child in expand(board, color):
        eval_score, _, coords = minimax(child[0], color, depth+1, alpha, beta, True, past)
        if eval_score is not None and eval_score < minEval:
            minEval = eval_score
            best_move = child[0]
            best_coords = child[1]
        beta = min(beta, eval_score or beta)  # Use `beta` if `eval_score` is None
        if beta <= alpha:
            break
    # Store minEval, best_move in past
    past[hash(board)] = minEval, best_move, best_coords
    return minEval, best_move, best_coords

# Will evaluate a board state and assign it a value
def evaluation(
    board: Bitboard, 
    colour: PlayerColor
) -> float:

    # Different factor multiples
    v1_constant = 2
    v6_constant = 1

    goodness = v1_constant * v1_minimax_util(board, colour) + v6_constant * v6_minimax_util(board, colour)
    
    return goodness

# Checks if the move is a completed game(very unlikely), or we have reached our desired depth
def cutoff_test(board, depth):
    if depth > DEPTH_VALUE:
        return True
    else:
        return finished(board)

#Simply checks if the game is over
def finished(board):
    #TODO: implement me
    return False
        

#returns the best move based on an end game scenario
def endgame_search(board, color):
    # May not be needed as we could simply modify the heuristic 
    #TODO: Implement me
    return search(board,color)
