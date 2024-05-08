# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import queue

MID_GAME = 75
OPENING = 1 # TODO Change this to another value
END_GAME = 3
MAX_TURN = 0
MIN_TURN = 1
DEPTH_VALUE = 3

DIRECTIONS = ["up", "down", "left", "right"]

from referee.game import PlayerColor, Action, PlaceAction, Coord
from agent.Bitboard import Bitboard 


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
        if self.num_moves < OPENING:
            return self.book_moves[self.num_moves]
        else:

            # Current children holds all children of board to a depth of 1, placing one tile. There is 118 children, maybe right?
            childen = search(self.board, self._color) # How do we get a placeAction from search?
            
            
            
            return 0


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There is only one action type, PlaceAction
        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords
        self.board.place_four_tiles(color, c1,c2,c3,c4)
        
        # Here we are just printing out the PlaceAction coordinates for
        # demonstration purposes. You should replace this with your own logic
        # to update your agent's internal game state representation.
        print(f"Testing: {color} played PLACE action: {c1}, {c2}, {c3}, {c4}")





###### General Expansion logic (FROM EXPANSION BRANCH) ######

"""
Input: 
    `board` - A bitboard
    `color` - The color of the player calling

Output: A list of all possible children (after placing 1 tetromino) of a board

Desc: Takes a board and color, finds the tiles on the board of that color, and
    performs the tile expansion from each one
"""

def search(board, color):
    return expand(board, color)

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
        visited = {index}  # Add back in the set functionality
        all_index_placements = init_expand_from_tile(board, index, color)
        moves.extend(all_index_placements)
    

    #for value in moves:
    #    # Print all boards with move
        
    #    value.bitboard_display()
    #    print("\n\n")

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
    all_shapes: list[Bitboard],
    seen_hashes: set
):
    # Add all of the boards of depth 4 and return
    if depth == 5:
        print(current_shape)
        #new_board = board.copy()
        #for i in range(4):
        #    new_board.set_tile(current_shape[i], player_colour)
        ##new_board.bitboard_display()
        board_hash = board.get_hash()
        if board_hash not in seen_hashes:
            seen_hashes.add(board_hash)
            all_shapes.append(board)
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

    # Will have repeats, need to convert to a set
    seen_hashes = set()
    all_shapes = []
    expand_out_sexy_style(board, start_index, player_colour, 1, [start_index], all_shapes, seen_hashes)
    #print(all_shapes)
    return all_shapes[1::]
        
    

        

    
