# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import queue

MID_GAME = 2
OPENING = 1
END_GAME = 3
MAX_TURN = 0
MIN_TURN = 1
DEPTH_VALUE = 3

DIRECTIONS = ["up", "down", "left", "right"]

from referee.game import PlayerColor, Action, PlaceAction, Coord
from Bitboard import Bitboard


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
        self.num_moves = 0
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

        #TODO: PUT ACTION CHOOSING LOGIC
        #if (self.num_moves > 3): #TODO: Make this state logic more complex
        #    if (self.num_moves > 140):
        #        self.state = END_GAME
        #    else:
        #        self.state = MID_GAME
        #match self.state:
        #    case OPENING:
        #        return self.book_moves[num_moves]
        #    case MID_GAME:
        #        return search(board, self._color)
        #    case END_GAME:
        #        return search(board, self._color)
        if self.state is OPENING:
            self.num_moves += 1
            return self.book_moves[self.num_moves - 1]
        else:
            search_result = search(self.board, self._color) # How do we get a placeAction from search?


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There is only one action type, PlaceAction
        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords
        num_moves += 1
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
def expand(
        board: Bitboard,
        color: PlayerColor
    ) -> set[Bitboard]:

    moves = set()

    # player_tiles is the list of indexes corresponding to the players tile
    if color is PlayerColor.RED:
        player_tiles = board.get_colour_indexes(PlayerColor.RED)
    else:
        player_tiles = board.get_colour_indexes(PlayerColor.BLUE)
        
    # For each tile, expand it
    for index in player_tiles:
        visited = {index}  # Add back in the set functionality
        moves.update(init_expand_from_tile(board, index, color))
    return moves

# Expand from a single square
"""
Inputs:
    `board` - A bitboard
    `start_index` - The index of a tile
    `visited` - 
    `moves` -
    `steps` - 

Output: Result of dfs_expand function

Desc: Takes a board and the index of a given tile of the calling players, and
    passes it into the depth first search"""
#def square_expand(board, start_index, visited, moves, steps):
#    return dfs_expand(board, start_index, visited, moves, steps)


#def dfs_expand(board, start_square, visited, moves, steps):
#    end_squares = []
#    calculate_diamond(start_square, end_squares) #TODO: Bitboard - Generate outer squares of diamond bitboard style
#    #TODO: Finlay - Add adjacent red square checking, fix diamond interior
#    for end_square in end_squares:
#        visited = set()
#        visited.add(start_square)
#        path = []
#        result = dfs_travel(board, start_square, end_square, path, visited, steps) 
#        moves.append(result)


#def expandable(board, square):
#    #TODO: Implement
#    return True


#Given Start coord, end coord travel to form a move with dfs
#def dfs_travel(board, start_square, end_square, path, visited, steps):
#    if steps == 5:
#        return None
#    if start_square == end_square:
#        return path + [end_square]
    
#    distance_to_end = abs(end_square.r - start_square.r) + abs(end_square.c - start_square.c) #TODO: Bitboard - manhattan distance between two points
    
#    # Define possible directions (up, down, right, left) prioritized based on proximity to the end square
#    #TODO: Bitboard - define these numerically
#    directions = [
#        (1, 0),  # Down
#        (0, 1),  # Right
#        (-1, 0), # Up
#        (0, -1)  # Left
#    ]
    
#    #TODO: Bitboard - HARD, change the following logic into bitboard 
#    # Sort directions based on their proximity to the end square
#    directions.sort(key=lambda d: abs((start_square.r + d[0]) - end_square.r) + abs((start_square.c + d[1]) - end_square.c))
#    # Explore all possible directions from the current square
#    for direction in directions:
#        next_square = Coord((start_square.r + direction[0]) % 11, (start_square.c + direction[1]) % 11)
        
#        # Check if the next square is within the bounds of the board
#        if next_square.r < 0 or next_square.r >= 11 or next_square.c < 0 or next_square.c >= 11:
#            continue  # Skip to the next direction if the next square is out of bounds
        
#        # Check if the next square is not visited and isn't filled
#        if (next_square not in visited) and (board.get(next_square) != PlayerColor.RED) and (board.get(next_square) != PlayerColor.BLUE):
#            # Create a new set for each recursive call
#            new_visited = set(visited)
#            new_visited.add(next_square)
#            # Recursively search from the next square
#            result = dfs_travel(board, next_square, end_square, path + [start_square], new_visited, steps+1)
#            # If a path is found, return it
#            if result:
#                return result
    
#    # If no path is found, return None
#    return None



#def calculate_diamond(square, end_squares):
#    directions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
#    # Initialize curr_square with the initial square
#    curr_square = Coord(square.r, square.c + 4)
#    # Append a copy of curr_square to end_squares
#    end_squares.append(curr_square)
#    for i in range(4):
#        for j in range(4):
#            # Create a new list for each curr_square iteration
#            if (i == 3 and j == 3):
#                return
#            new_square = Coord((curr_square.r + directions[i][0]) % 11, (curr_square.c + directions[i][1]) %11)
#            end_squares.append(new_square)
#            #print(new_square)
#            curr_square = new_square
            

            

#def UCS_expand(board, square):
#    #TODO: test this by displaying the children printed and showing duplicates etc.
#    # Directions for moving up, down, left, right
#    #TODO: Use purple book to look at effiency ideas
#    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#    visited = set()  # Keep track of visited nodes
#    frontier = queue.PriorityQueue()  # Priority queue for UCS
#    frontier.put((0, square))  # (cost, position) tuple
#    steps = 4
#    #TODO: Implement expanding logic for adjacent red squares
#    #TODO: Update this for our board setup
#    # Run until there are no more nodes to explore or we reach the step limit
#    while not frontier.empty():
#        cost, current = frontier.get()
#        if cost > steps:
#            continue
#        if current not in visited:
#            visited.add(current)
#            # Explore neighbors
#            for direction in directions:
#                next_row = current[0] + direction[0]
#                next_col = current[1] + direction[1]
#                if 0 <= next_row < len(board) and 0 <= next_col < len(board[0]):
#                    if board[next_row][next_col]:  # We are hitting an obstacle
#                        next_position = (next_row, next_col)
#                        frontier.put((cost + 1, next_position))

#    return visited




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
    current_shape: Bitboard, 
    all_shapes: set[Bitboard]
):
    
    # Add all of the boards of depth 4 and return
    if depth == 4:
        all_shapes.add(current_shape[:])
        return
    

    for direction in DIRECTIONS:

        # Move out 1 in each direction from the given index to get new index
        new_index = board.move_adj(index, direction)

        # If new_index is empty, we set the tile, update the shape, copy the
        # board to get a new board, then clean up the original board
        if board.get_tile(new_index) is None:
            board.set_tile(new_index, player_colour)
            new_shape = current_shape + [new_index]
            expand_out_sexy_style(board.copy(), new_index, player_colour, depth + 1, new_shape, all_shapes)
            board.clear_tile(new_index)



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
) -> set[Bitboard]:

    # Will have repeats, need to convert to a set
    all_shapes = set()
    expand_out_sexy_style(board, start_index, player_colour, 1, [start_index], all_shapes)
    return all_shapes
        
    

        

    
    





###### Functions specific to Minimax Branch #####

#TODO: develop a better way of saving + storing child nodes, use acutal board type
class Move:
    def __init__(
        self, 
        value:int, 
        board:Bitboard
    ) -> None:
        self.value = value
        self.board = board
    
def search(
    board: Bitboard, 
    color: PlayerColor
):
    return minimax(board, color, 0, float('-inf'), float('inf'), True) # Returns the placeaction of the best move to make

def minimax(
    board: Bitboard, 
    color: PlayerColor, 
    depth: int, 
    alpha, 
    beta, 
    maximizingPlayer, 
    past = {}
):
    if cutoff_test(board, depth):
        return None #, evaluation(board, color), 
    #TODO: make this more efficient through hashing, won't work otherwise
    #if board in past:
        #return past[board]
    if maximizingPlayer:
        move, score = max_value(board, color, depth, alpha, beta, past)
        return move
    else:
        move, score = min_value(board, color, depth, alpha, beta, past)
        return move

def max_value(
    board: Bitboard, 
    color: PlayerColor, 
    depth: int, 
    alpha, 
    beta, 
    past
):
    maxEval = float('-inf')
    best_move = None
    for child in expand(board):
        eval, _ = minimax(child, color, depth+1, alpha, beta, False)
        if eval > maxEval:
            maxEval = eval
            best_move = child  # Update the best move
        alpha = max(alpha, eval)
        if beta <= alpha:
            break
    #TODO: store maxEval, best_move in past
    return maxEval, best_move

def min_value(
    board: Bitboard, 
    color: PlayerColor, 
    depth: int, 
    alpha, 
    beta, 
    past
):
    minEval = float('inf')
    best_move = None
    for child in expand(board):
        eval, _ = minimax(child, color, depth+1, alpha, beta, True)
        if eval < minEval:
            minEval = eval
            best_move = child
        beta = min(beta, eval)
        if beta <= alpha:
            break
    #TODO: store minEval, best_move in past
    return minEval, best_move

# Will evaluate a board state and assign it a value
def evaluation(
    board: Bitboard, 
    colour: PlayerColor
):
    # will be generated by Nic in his own branch, returns an integer score for the node where higher is better
    return 0

# Checks if the move is a completed game(very unlikely), or we have reached our desired depth
def cutoff_test(
    board: Bitboard, 
    depth: int
):
    if depth > DEPTH_VALUE:
        return True
    elif finished(board):
        return True
    return False
    #Unsure if this is both for winning and losing
    
    return True

#returns the best move based on an end game scenario
def endgame_search(
    board: Bitboard, 
    color: PlayerColor
):
    # May not be needed as we could simply modify the heuristic 
    return PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                )

#Simply checks if the game is over
def finished(
    board: Bitboard):
    #TODO: implement me
    return False