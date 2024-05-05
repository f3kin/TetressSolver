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
    moves = []
    #TODO: modify color_val based on input color, Nic's colour rep
    for square in board:
        if (board[square] == color) & expandable(board, square): #It is one of our colour squares
            visited = {square}  # Initialize visited set with the start square
            square_expand(board, square, visited, moves, 0)  # Perform DFS from the start square
    return list(moves)

# Expand from a single square
def square_expand(board, start_square, visited, moves, steps):
    #Initial diamond expanding logic
    return dfs_expand(board, start_square, visited, moves, steps)


def dfs_expand(board,start_square, visited, moves, steps):
    #print("start square:")
    #print(start_square)
    # Base case: stop exploring if we've reached the maximum number of steps
    # Store the squares visited 
    # End squares - squares at the edge of the diamond
    end_squares = []
    calculate_diamond(start_square, end_squares)
    #TODO: Add adjacent red square checking
    for end_square in end_squares:
        visited = set()
        visited.add(start_square)
        path = []
        result = dfs_travel(board, start_square, end_square, path, visited, steps) #TODO: get this returning stuff right
        # print(start_square)
        # print(result[1::])
        # print(end_square)
        # print("\n")
        moves.append(result)

def expandable(board, square):
    #TODO: Implement
    return True

def dfs_travel(board, start_square, end_square, path, visited, steps):
    # Base case: if the current square is the end square, return the path
    if steps == 5:
        return None
    if start_square == end_square:
        return path + [end_square]
    
    # Calculate the distance from the current square to the end square
    distance_to_end = abs(end_square.r - start_square.r) + abs(end_square.c - start_square.c)
    
    # Define possible directions (up, down, right, left) prioritized based on proximity to the end square
    directions = [
        (1, 0),  # Down
        (0, 1),  # Right
        (-1, 0), # Up
        (0, -1)  # Left
    ]
    
    # Sort directions based on their proximity to the end square
    directions.sort(key=lambda d: abs((start_square.r + d[0]) - end_square.r) + abs((start_square.c + d[1]) - end_square.c))
    # Explore all possible directions from the current square
    for direction in directions:
        next_square = Coord((start_square.r + direction[0]) % 11, (start_square.c + direction[1]) % 11)
        
        # Check if the next square is within the bounds of the board
        if next_square.r < 0 or next_square.r >= 11 or next_square.c < 0 or next_square.c >= 11:
            continue  # Skip to the next direction if the next square is out of bounds
        
        # Check if the next square is not visited and isn't filled
        if (next_square not in visited) and (board.get(next_square) != PlayerColor.RED) and (board.get(next_square) != PlayerColor.BLUE):
            # Create a new set for each recursive call
            new_visited = set(visited)
            new_visited.add(next_square)
            # Recursively search from the next square
            result = dfs_travel(board, next_square, end_square, path + [start_square], new_visited, steps+1)
            # If a path is found, return it
            if result:
                return result
    
    # If no path is found, return None
    return None



def calculate_diamond(square, end_squares):
    #Seems to work now
    directions = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    # Initialize curr_square with the initial square
    curr_square = Coord(square.r, square.c + 4)
    # Append a copy of curr_square to end_squares
    end_squares.append(curr_square)
    for i in range(4):
        for j in range(4):
            # Create a new list for each curr_square iteration
            if (i == 3 and j == 3):
                return
            new_square = Coord((curr_square.r + directions[i][0]) % 11, (curr_square.c + directions[i][1]) %11)
            end_squares.append(new_square)
            #print(new_square)
            curr_square = new_square
            

            

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