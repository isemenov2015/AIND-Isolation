"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def center_distance(game, moves_list):
    """
    Returns average euclidean distance for moves in move_list from the
    board center
    """
    if len(moves_list) == 0:
        return -float("Inf")
    center_coord = (float(game.width) / 2, float(game.height) / 2)
    dist = np.sqrt(np.sum((moves_list - center_coord)**2))
    #print(dist)
    return dist

def check_killer(game, moves_list):
    """
    'Killer' / 'Antikiller' strategy implementation
    Returns +Inf, if there is a move that wins the game in the moves_list
    Returns 0 otherwise
    """
    for move in moves_list:
        new_board = game.forecast_move(move)
        if new_board.is_loser(new_board.active_player):
            return float("Inf")
        #if new_board.is_winner(new_board.inactive_player):
        #    return -float("Inf")
        #if new_board.is_winner(new_board.active_player):
        #    return float("Inf")
    return 0

def players_distance(game, player):
    """
    Returns euclidean distance between the player and its opponent
    """
    player_pos = np.array(game.get_player_location(player))
    opp_pos = np.array(game.get_player_location(game.get_opponent(player)))
    #print(player_pos, opp_pos, np.sqrt((player_pos[0] - opp_pos[0])**2 + 
    #                                   (player_pos[1] - opp_pos[1])**2))
    return np.linalg.norm(player_pos - opp_pos)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # DONE: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #n_moves = game.width * game.height - len(game.get_blank_spaces())
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
        #own_dist = center_distance(game, np.array(own_moves))
        #opp_dist = center_distance(game, np.array(opp_moves))
        #free_squares = math.sqrt(len(game.get_blank_spaces()))
    n_own_moves = len(own_moves) + 0.01
    n_opp_moves = len(opp_moves) + 0.01
    killer_score = check_killer(game, own_moves)
        #print(killer_score)
    #if killer_score != 0:
    #    print('Killer worked!')
    score = n_own_moves - n_opp_moves + killer_score
    #score = (n_own_moves - n_opp_moves) + \
    #        n_own_moves / n_opp_moves + \
    #        (own_dist) * 5 / n_moves
    #print("Stage ", n_moves, "Score: ", score, 
    #      "Distance addon: ", (own_dist - opp_dist) * 10 / n_moves)
    return score


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = float(len(game.get_legal_moves(player)) + 0.01)
    opp_moves = float(len(game.get_legal_moves(game.get_opponent(player))) + 0.01)

    # DONE: finish this function!
    return own_moves + own_moves / opp_moves


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
    def terminal_test(self, game, depth):
        """
        Returns TRUE if no legal moves left for the active player or max
        depth is reached
        FALSE otherwise
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return len(game.get_legal_moves()) == 0 or depth <= 0
    
    def min_value(self, game, depth):
        """
        Returns the value of a win if the game is over
        Otherwise returns minimum value over all legal child nodes
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()        
        if self.terminal_test(game, depth):
            if len(game.get_legal_moves()) > 0:
                return self.score(game, self)
            else:
                return float("Inf")
        state_val = []
        for move in game.get_legal_moves():
            state_val.append(self.max_value(game.forecast_move(move), depth - 1))
        return min(state_val)

    def max_value(self, game, depth):
        """
        Returns the value of a loss if the game is over
        Otherwise returns maximum value over all legal child nodes
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            if len(game.get_legal_moves()) > 0:
                return self.score(game, self)
            else:
                return -float("Inf")
        state_val = []
        for move in game.get_legal_moves():
            state_val.append(self.min_value(game.forecast_move(move), depth - 1))
        return max(state_val)

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        """

        # DONE: finish this function!
        moves_list = game.get_legal_moves()
        if len(moves_list) == 0:
            return (-1, -1)
        maximum = -float("Inf")
        max_index = 0
        for i in range(len(moves_list)):
            value = self.min_value(game.forecast_move(moves_list[i]), depth - 1)
            if value > maximum:
                maximum = value
                max_index = i
        return moves_list[max_index]


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        depth = 0

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:
                depth += 1
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # DONE: finish this function!
        moves_list = game.get_legal_moves()
        if len(moves_list) == 0:
            return (-1, -1)
        max_index = 0
        for i in range(len(moves_list)):
            value = self.min_value(game.forecast_move(moves_list[i]), 
                                   depth - 1, alpha, beta)
            if value > alpha:
                alpha = value
                max_index = i
        return moves_list[max_index]
    
    def terminal_test(self, game, depth):
        """
        Returns TRUE if no legal moves left for the active player or max
        depth is reached
        FALSE otherwise
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        return len(game.get_legal_moves()) == 0 or depth <= 0

    def max_value(self, game, depth, alpha, beta):
        """
        Returns the value of a loss if the game is over
        Otherwise returns maximum value over all legal child nodes
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            return self.score(game, self)
        v = -float("Inf")
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), 
                                   depth - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def min_value(self, game, depth, alpha, beta):
        """
        Returns the value of a win if the game is over
        Otherwise returns minimum value over all legal child nodes
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        if self.terminal_test(game, depth):
            return self.score(game, self)
        v = float("Inf")
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), 
                                   depth - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v