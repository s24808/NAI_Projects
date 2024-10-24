# AUTHORS: Filip Labuda, Jędrzej Stańczewski
# RULES: Players take turns placing their pieces on a 5x5 board. Each player has 12 pieces to use.
#   After all pieces have been placed, the player who has formed the most three-piece combinations in a row wins.
# ENVIRONMENT SETUP: The program requires the easyAI library to run.
#   You can install it using the command "pip install easyAI" or "pip3 install easyAI".

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax


class ThreeInARowGame(TwoPlayerGame):
    """
    Attributes:
        players: A list of players.
        board_size: Size of the board (5x5).
        board: A list representing the board with each cell initialized to empty (' ').
        current_player: The player whose turn it is (1 for player 1, 2 for player 2).
        total_pieces: The total number of pieces each player has to place (12 per player).
        men_to_place: A list containing the number of pieces left to place for each player.
    """

    def __init__(self, players):
        """
        Initializes the game board and players.

        Args:
            players (list): A list containing the players (either AI or human).
        """
        self.players = players
        self.board_size = 5
        self.board = [' '] * (self.board_size ** 2)  # Initialize a 5x5 board
        self.current_player = 1  # Player 1 starts
        self.total_pieces = 12  # Total pieces each player has
        self.men_to_place = [self.total_pieces, self.total_pieces]  # Pieces left to place for each player

    def possible_moves(self):
        """
        Returns a list of possible moves (empty positions) on the board.

        Returns:
            list: A list of strings representing the indices of empty cells.
        """
        return [str(i + 1) for i, cell in enumerate(self.board) if cell == ' ']

    def make_move(self, move):
        """
        Places a player's piece on the board at the specified position.

        Args:
            move (str): The index of the cell (1-based) where the current player wants to place a piece.
        """
        index = int(move) - 1
        self.board[index] = self.current_player_symbol()
        self.men_to_place[self.current_player - 1] -= 1

    def unmake_move(self, move):
        """
        Removes a player's piece from the board (used for undoing a move).

        Args:
            move (str): The index of the cell (1-based) where the piece should be removed.
        """
        index = int(move) - 1
        self.board[index] = ' '
        self.men_to_place[self.current_player - 1] += 1

    def is_over(self):
        """
        Determines if the game is over.

        The game is over when both players have placed all their pieces or there are no possible moves left.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return sum(self.men_to_place) == 0 or not self.possible_moves()

    def show(self):
        """
        Displays the current state of the board as a 5x5 grid.
        """
        for i in range(self.board_size):
            row = '|'.join(self.board[i * self.board_size:(i + 1) * self.board_size])
            print(row)
            if i < self.board_size - 1:
                print('-' * (self.board_size * 2 - 1))

    def scoring(self):
        """
        Calculates the score of the current player based on the number of "three-in-a-row" they have.

        Returns:
            int: The score, where each "three-in-a-row" gives a point.
        """
        counts = self.count_three_in_a_row()
        score = counts[self.current_player - 1] - counts[1 - (self.current_player - 1)]
        return score

    def current_player_symbol(self):
        """
        Returns the symbol associated with the current player.

        Returns:
            str: 'X' for player 1 and 'O' for player 2.
        """
        return 'X' if self.current_player == 1 else 'O'

    def count_three_in_a_row(self):
        """
        Counts the number of occurrences where each player has three connected pieces horizontally, vertically, or diagonally.

        Returns:
            list: A list of two integers where the first element is the count for player 1 and the second is the count for player 2.
        """
        counts = [0, 0]  # Index 0 for Player 1, Index 1 for Player 2
        b = self.board
        size = self.board_size
        lines = []

        # Collect all possible lines of length 3
        # Horizontal lines
        for i in range(size):
            for j in range(size - 2):
                lines.append([b[i * size + j + k] for k in range(3)])

        # Vertical lines
        for j in range(size):
            for i in range(size - 2):
                lines.append([b[(i + k) * size + j] for k in range(3)])

        # Diagonal lines (top-left to bottom-right)
        for i in range(size - 2):
            for j in range(size - 2):
                lines.append([b[(i + k) * size + (j + k)] for k in range(3)])

        # Diagonal lines (top-right to bottom-left)
        for i in range(size - 2):
            for j in range(2, size):
                lines.append([b[(i + k) * size + (j - k)] for k in range(3)])

        # Count occurrences for each player
        for line in lines:
            if line.count('X') == 3 and line.count('O') == 0:
                counts[0] += 1
            elif line.count('O') == 3 and line.count('X') == 0:
                counts[1] += 1
        return counts

    def win_message(self):
        """
        Determines the winner based on the number of "three-in-a-row" sequences each player has and prints the result.

        Returns:
            str: A message indicating whether player 1 won, player 2 won, or if it's a tie.
        """
        counts = self.count_three_in_a_row()
        print(f"Player 1: {counts[0]}, player 2: {counts[1]}")
        if counts[0] > counts[1]:
            return "Player 1 wins!"
        elif counts[1] > counts[0]:
            return "Player 2 wins!"
        else:
            return "It's a tie!"


# Main game logic
if __name__ == "__main__":
    """
    Entry point for the game when executed as a script.
    Initializes the game with a human player and an AI player, then starts the game loop.
    """
    ai_algo = Negamax(2)  # Depth of 2
    game = ThreeInARowGame([Human_Player(), AI_Player(ai_algo)])
    game.play()
    print(game.win_message())
