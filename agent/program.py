# COMP30024 Artificial Intelligence, Semester 1 2024
# Project Part B: Game Playing Agent

MID_GAME = 2
OPENING = 1
END_GAME = 3
MAX_TURN = 0
MIN_TURN = 1
DEPTH_VALUE = 3

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
        self.num_moves = 0
        self.state = OPENING
        self.book_moves = [PlaceAction( #TODO: make these moves legitimate, will have to be slightly more complex, i.e. first red is open, blue is always a response to the red book move
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                ), PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                ), PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
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

        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        #TODO: PUT ACTION CHOOSING LOGIC
        if (self.num_moves > 3): #TODO: Make this state logic more complex
            if (self.num_moves > 140):
                self.state = END_GAME
            else:
                self.state = MID_GAME
        match self.state:
            case OPENING:
                return self.book_moves[num_moves]
            case MID_GAME:
                return search(board, self._color)
            case END_GAME:
                return endgame_search(board, self._color)



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

###### Functions specific to Minimax #####

#TODO: develop a better way of saving + storing child nodes, use acutal board type
class Move:
    def __init__(self, value:int, board:board) -> None:
        self.value = value
        self.board = board
    
def search(board, color):
    return minimax(board, color, 0, float('-inf'), float('inf'), True) # Returns the placeaction of the best move to make

def minimax(board, color, depth, alpha, beta, maximizingPlayer , past = {}):
    if cutoff_test(board, depth):
        return evaluation(board, color), None
    #TODO: make this more efficient through hashing, won't work otherwise
    #if board in past:
        #return past[board]
    if maximizingPlayer:
        return max_value(board, color, depth, alpha, beta, past)
    else:
        return min_value(board, color, depth, alpha, beta, past)

def max_value(board, color, depth, alpha, beta, past):
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

def min_value(board, color, depth, alpha, beta, past):
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
def evaluation(board, colour):
    # will be generated by Nic in his own branch, returns an integer score for the node where higher is better
    return 0

# Checks if the move is a completed game(very unlikely), or we have reached our desired depth
def cutoff_test(board, depth):
    if depth > DEPTH_VALUE:
        return True
    elif finished(board):
        return True
    return False
    #Unsure if this is both for winning and losing
    
    return True

#returns the best move based on an end game scenario
def endgame_search(board, color):
    # May not be needed as we could simply modify the heuristic 
    return PlaceAction(
                    Coord(3, 3), 
                    Coord(3, 4), 
                    Coord(4, 3), 
                    Coord(4, 4)
                )

def expand(board, colour):
    #TODO: Implement me from other branch

    return board

#Simply checks if the game is over
def finished(board):
    #TODO: implement me
    return False