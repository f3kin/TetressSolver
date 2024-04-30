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
    children = [] #TODO: Increase child storing efficiency from a list
    #TODO: modify color_val based on input color, Nic's colour rep
    for square in board:
        # We have found a square to expand
        #TODO: Implement improved efficiency from expanding one square at a time
        if square == color_val:
            new_children = square_expand(board, square)
            #TODO: change to adding more efficiently 
            for child in new_children:
                children.append(child)
    
# Expand from a single square
def square_expand(board, square):
    #Initial diamond expanding logic
    return UCS_expand(board,square)

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
