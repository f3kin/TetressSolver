# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

import queue

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
        self._color = color
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

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a PLACE action")
                return PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                )
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a PLACE action")
                return PlaceAction(
                    Coord(2, 3), 
                    Coord(2, 4), 
                    Coord(2, 5), 
                    Coord(2, 6)
                )

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after an agent has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        # There is only one action type, PlaceAction
        place_action: PlaceAction = action
        c1, c2, c3, c4 = place_action.coords

        # Here we are just printing out the PlaceAction coordinates for
        # demonstration purposes. You should replace this with your own logic
        # to update your agent's internal game state representation.
        print(f"Testing: {color} played PLACE action: {c1}, {c2}, {c3}, {c4}")


###### General Expansion logic ######
# Uses Nic's board format
def expand(board, color):
    color_val = 0
    moves = []
    #TODO: modify color_val based on input color, Nic's colour rep
    for square in board:
        # We have found a square to expand
        if square == color_val & expandable(board, square): #It is one of our colour squares
            visited = {square}  # Initialize visited set with the start square
            square_expand(square, visited, moves, 0)  # Perform DFS from the start square
    return list(moves)

# Expand from a single square
def square_expand(start_square, visited, moves, steps):
    #Initial diamond expanding logic
    return dfs_expand(start_square, visited, moves, steps)


def dfs_expand(start_square, visited, moves, steps):
    # Base case: stop exploring if we've reached the maximum number of steps
    if steps == 5:
        return
    # Store the squares visited 
    # End squares - squares at the edge of the diamond
    end_squares = []
    calculate_diamond(board, start_square, end_squares)
    #TODO: Add adjacent red square checking
    for end_square in end_squares:
        visited = set()
        visited.add(start_square)
        path = []
        dfs_travel(start_square, end_square, path, visited) #TODO: get this returning stuff right
        moves.add(path)

def expandable(board, square):
    #TODO: Implement
    return True

def dfs_travel(start_square, end_square, path, visited):
    # Base case: if the current square is the end square, return the path
    if start_square == end_square:
        return path + [end_square]
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Define possible directions (up, down, right, left)
    
    # Explore all possible directions from the current square
    for direction in directions:
        next_square = (start_square[0] + direction[0], start_square[1] + direction[1])
        
        # Check if the next square is not visited and isnt filled
        if next_square not in visited | !(next_square.colour):
            # Add the next square to the visited set and update the path
            visited.add(next_square)
            result = dfs_travel(next_square, end_square, path + [start_square], visited)
            
            # If a path is found, return it
            if result:
                return result
    
    # If no path is found, return None
    return None


def calculate_diamond(board, square, end_squares):
    directions = [(-1, 0), (0, -1), (1,0), (0, 1)]
    #TODO: Implement edge cases - in DFS travel, find next closest square to end square
    curr_square = square + (0,4)
    end_squares.append(curr_square)
    for i in range(4):
        for j in range(4):
            curr_square += directions[i]
            end_squares.append(curr_square)

            

'''
def UCS_expand(board, square):
    #TODO: test this by displaying the children printed and showing duplicates etc.
    # Directions for moving up, down, left, right
    #TODO: Use purple book to look at effiency ideas
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()  # Keep track of visited nodes
    frontier = queue.PriorityQueue()  # Priority queue for UCS
    frontier.put((0, square))  # (cost, position) tuple
    steps = 4
    #TODO: Implement expanding logic for adjacent red squares
    #TODO: Update this for our board setup
    # Run until there are no more nodes to explore or we reach the step limit
    while not frontier.empty():
        cost, current = frontier.get()
        if cost > steps:
            continue
        if current not in visited:
            visited.add(current)
            # Explore neighbors
            for direction in directions:
                next_row = current[0] + direction[0]
                next_col = current[1] + direction[1]
                if 0 <= next_row < len(board) and 0 <= next_col < len(board[0]):
                    if board[next_row][next_col]:  # We are hitting an obstacle
                        next_position = (next_row, next_col)
                        frontier.put((cost + 1, next_position))

    return visited
'''