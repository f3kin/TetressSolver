from referee.game import PlayerColor, Action, PlaceAction, Coord

def expand(board, color):
    moves = []
    #TODO: modify color_val based on input color, Nic's colour rep
    for square in board:
        if board[square] == color & expandable(board, square): #It is one of our colour squares
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


expand(board, PlayerColour.RED)