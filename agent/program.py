# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

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
        # TODO: IMPLEMENT PRECOMPUTATION HERE
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
        #TODO: PUT ACTION CHOOSING LOGIC
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

###### Functions specific to Minimax #####
def search(board, colour):
    return minimax(board, colour) # Returns the placeaction of the best move to make

def minimax(state, game, a, b):
    if cutoff_test(state):
        return evaluation(state)
    return max_value(state, game, a, b) # Assumes we are always maximising, think this is correct ?
    
def max_value(state, game, a, b):
    # State - current game state (probably the board)
    # Game - game description
    # a - the best score for max along the path to state
    # b - the best score for min along the path to state
    if cutoff_test(state):
        return evaluation(state)
    children = expand(state)
    for child in children:
        a = max(a, min_value(child, game, a, b))
        if a >= b:
            return b
    return a

def min_value(state, game, a ,b):
    if cutoff_test(state):
        return evaluation(state)
    children = expand(state)
    for child in children:
        b = min(b,max_value(child, game, a, b))
        if b <= a:
            return a
    return b

def cutoff_test(state):
    return 0