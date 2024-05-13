# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent


# Depends entirely on if we can use deque
from collections import deque
from typing import Tuple, Optional
from referee.game.constants import BOARD_N
from referee.game.player import PlayerColor
from referee.game.coord import Coord

##BITBOARD CODE


class Bitboard:

	# Create 11 row and 11 column masks for clearing rows and columns.
	# Creates here because they are the same for all board, and saves
	# computation time when creating new bitboards

	row_masks = [(1 << BOARD_N) - 1 << (i * BOARD_N) for i in range(BOARD_N)]

	col_masks = [0 for _ in range(BOARD_N)]
	
	for j in range(BOARD_N):
		for k in range(BOARD_N):
			col_masks[j] |= (1 << (j + k * BOARD_N))

	def __init__(
		self
	):
		self.red_board = 0
		self.blue_board = 0
		self.total_bits = BOARD_N * BOARD_N

	
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
		isRed: bool
	):
		# On the players board, bitwise OR their board with a a mask of
		# a board with only the index tile switched on
		if isRed:
			self.red_board |= (1 << index)
		else:
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

			if masked_row_board == row_mask:
				combined_masks |= row_mask

		for col in cols_to_check:

			col_mask = Bitboard.col_masks[col] # Precomputed col all set bits
			masked_col_board = full_board & col_mask
			# If col is full in the full board, then add it to the combined
			# mask we will use to clear rows/cols at the end

			if masked_col_board == col_mask:
				#print("Test")
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
		isRed = False
		if colour == PlayerColor.RED:
			isRed = True
		self.set_tile(index1, isRed)
		self.set_tile(index2, isRed)
		self.set_tile(index3, isRed)
		self.set_tile(index4, isRed)

	def get_hash(
		self
	):
		return hash((self.red_board, self.blue_board))
	

	def valid_book_move(
		self,
		indexes: list[int],
		colour: PlayerColor,
		turn_counter: int
	) -> bool:

		if colour == PlayerColor.RED:
			player_board = self.red_board
		else:
			player_board = self.blue_board

		
		adjacent_found = False

		if turn_counter == 0:
			adjacent_found = True

		for index in indexes:
			if (self.red_board | self.blue_board) & (1 << index):
				return False
		
		for index in indexes:
			if adjacent_found:
				return True
			
			adjacent_indexes = [
				self.move_adj(index, 'up'),
				self.move_adj(index, 'down'),
				self.move_adj(index, 'left'),
				self.move_adj(index, 'right'),
			]
			for adj_index in adjacent_indexes:
				if player_board & (1 << adj_index):
					adjacent_found = True
		return adjacent_found

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

OPENING = 3 # TODO Change this to another value
END_GAME = 75
DEPTH_VALUE = 5

DIRECTIONS = ["up", "down", "left", "right"]

from referee.game import PlayerColor, Action, PlaceAction, Coord


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
