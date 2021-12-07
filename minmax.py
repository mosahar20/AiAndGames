import socket
from random import choice
from time import sleep
import math
import copy
import numpy as np

# python Hex.py "a=good_agent;python agents\DefaultAgents\minmax.py" -v
LOSE = 1000  # Choose win value higher than possible score but lower than INF

class NaiveAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234

    I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
    J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]

    def run(self):
        """A finite-state machine that cycles through waiting for input
        and sending moves.
        """

        self._board_size = 0
        # self._board = []
        self._colour = ""
        self._turn_count = 1
        self._choices = []
        self._best_move = []

        states = {
            1: NaiveAgent._connect,
            2: NaiveAgent._wait_start,
            3: NaiveAgent._make_move,
            4: NaiveAgent._wait_message,
            5: NaiveAgent._close
        }

        res = states[1](self)
        while (res != 0):
            res = states[res](self)

    def eval_dijkstra(self, color):
        """Evaluation based on distance to border. Score computed by taking difference of dijkstra score of opponent and current player

        Returns:
            int: Board evaluation score
        """
        self.eval_count += 1
        return self.get_dijkstra_score(self.board.get_opposite_color(color)) - self.get_dijkstra_score(color)

    def get_dijkstra_score(self, color):
        """gets the dijkstra score for a certain color, differs from dijkstra eval in that it only considers the passed color

        Args:
            color (Hexboard.Color): What color to evaluate

        Returns:
            int: score of how many (shortest) path-steps remain to victory
        """
        scores = np.array([[LOSE for i in range(self._board_size)]
                           for j in range(self._board_size)])
        updated = np.array([[True for i in range(self._board_size)] for j in range(
            self._board_size)])  # Start updating at one side of the board

        #alignment of color (blue = left->right so (1,0))
        # alignment = (0, 1) if color == self.board.BLUE else (1, 0)
        alignment = (0, 1) if color == "B" else (1, 0)


        for i in range(self._board_size):
            # iterate over last row or column based on alignment of current color
            newcoord = tuple([i * j for j in alignment])

            updated[newcoord] = False
            # TODO 
            # if same color --> path starts at 0
            if self.is_color(newcoord, color):
                scores[newcoord] = 0
            # if empty --> costs 1 move to use this path
            elif self.is_empty(newcoord):
                scores[newcoord] = 1
            else:  # If other color --> can't use this path
                scores[newcoord] = LOSE

        scores = self.dijkstra_update(color, scores, updated)

        #self.board.print_dijkstra(scores)

        results = [scores[alignment[0] * i - 1 + alignment[0]][alignment[1]*i - 1 + alignment[1]]
                   for i in range(self.board.size)]  # take "other side" to get the list of distance from end-end on board
        best_result = min(results)

        # if best_result == 0:
        #     best_result = -500

        log.debug("Best score for color {}: {}".format(color, best_result))
        return best_result  # return minimum distance to get current score

    def is_empty(self, coordinates):
        """Checks if cell is empty

        Args:
            coordinates (tuple): Cell (x, y) coordinates

        Returns:
            bool: Whether specified cell is empty
        """
        #assert self.is_legal_move(coordinates)
        # TODO 
        return self.board[coordinates] == "0"


    def is_color(self, coordinates, color):
        """Checks if cell is filled with a certain color

        Args:
            coordinates (tuple): Cell (x, y) coordinates
            color (int): Player color

        Returns:
            bool: Whether specified cell contains the color
        """
        #assert self.is_legal_move(coordinates)
        # TODO 
        return self.board[coordinates] == color

    ###########################################################
    ###########################################################
    ###########################################################


    def _connect(self):
        """Connects to the socket and jumps to waiting for the start
        message.
        """

        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((NaiveAgent.HOST, NaiveAgent.PORT))

        return 2

    def _wait_start(self):
        """Initialises itself when receiving the start message, then
        answers if it is Red or waits if it is Blue.
        """

        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        if (data[0] == "START"):
            self._board_size = int(data[1])
            # self._board = [["0" for i in range(self._board_size)] for j in range(self._board_size)]
            self._board = np.full((self._board_size, self._board_size), "0")
            self._colour = data[2]

            if (self._colour == "R"):
                return 3
            else:
                return 4

        else:
            print("ERROR: No START message received.")
            return 0

    def _make_move(self):
        """Makes a random valid move. It will choose to swap with
        a coinflip.
        """

        if (self._turn_count == 2 and choice([0, 1]) == 1):
            msg = "SWAP\n"

        else:
            newBoard = copy.deepcopy(self._board)
            self.MinMax(newBoard, True, 1, -math.inf, math.inf)
            msg = f"{self._best_move[0]},{self._best_move[1]}\n"

        self._s.sendall(bytes(msg, "utf-8"))

        return 4

    def _wait_message(self):
        """Waits for a new change message when it is not its turn."""

        self._turn_count += 1

        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        if (data[0] == "END" or data[-1] == "END"):
            return 5
        else:

            if (data[1] == "SWAP"):
                self._colour = self.opp_colour()
                for x in range(self._board_size):
                    for y in range(self._board_size):
                        if self._board[x][y] != "0":
                            if (self._board[x][y] == "R"):
                                self._board[x][y] = "B"
                            else:
                                self._board[x][y] = "R"
                            break
            else:
                x, y = data[1].split(",")
                if (data[-1] == self._colour):
                    self._board[int(x)][int(y)] = self.opp_colour()
                else:
                    self._board[int(x)][int(y)] = self._colour
            # print(self._board)
            if (data[-1] == self._colour):
                return 3

        return 4

    def _close(self):
        """Closes the socket."""

        self._s.close()
        return 0

    def opp_colour(self):
        """Returns the char representation of the colour opposite to the
        current one.
        """

        if self._colour == "R":
            return "B"
        elif self._colour == "B":
            return "R"
        else:
            return "None"

    def MinMax(self, startBoard, maximizingPlayer, depth, alpha, beta):
        alpha = alpha
        beta = beta
        iterationBestMove = []

        if depth == 0 or not (any("0" in subl for subl in startBoard)):
            # return choice(range(0, 1))  # return heuristic
            return self.eval_dijkstra(self._colour)

        if maximizingPlayer:
            bestValue = -math.inf
            moves = self.get_moves(startBoard)
            for move in moves:
                updatedBoard = self.update_board(copy.deepcopy(
                    startBoard), move[0], move[1], self._colour)
                value = self.MinMax(updatedBoard, False, depth-1, alpha, beta)
                if value > bestValue:
                    iterationBestMove = move
                    bestValue = value
                beta = min(beta, bestValue)
                if beta <= alpha:
                    break
            self._best_move = iterationBestMove
            return bestValue
        else:
            bestValue = math.inf
            moves = self.get_moves(startBoard)
            for move in moves:
                updatedBoard = self.update_board(
                    copy.deepcopy(startBoard), move[0], move[1], "B")
                bestValue = min(bestValue, self.MinMax(
                    updatedBoard, True, depth-1, alpha, beta))
                beta = min(beta, bestValue)
                if beta <= alpha:
                    break
            return bestValue

    def get_moves(self, board):

        moves = []
        for x in range(self._board_size):
            for y in range(self._board_size):
                if board[x][y] == "0":
                    moves.append([x, y])

        return moves

    def update_board(self, updateboard, x, y, colour):
        updateboard[x][y] = colour
        return updateboard

    # def board_from_string(self, string_input, board_size=11, bnf=True):
    #         """Loads a board from a string representation index 2 from protocol. If bnf=True, it will
    #         load a protocol-formatted string. Otherwise, it will load from a
    #         human-readable-formatted board.
    #         """
    #         b =  self._board

    #         if (bnf):
    #             lines = string_input.split(",")
    #             for i, line in enumerate(lines):
    #                 for j, char in enumerate(line):
    #                     b[i,j]= char
    #         else:
    #             lines = [line.strip() for line in string_input.split("\n")]
    #             for i, line in enumerate(lines):
    #                 chars = line.split(" ")
    #                 for j, char in enumerate(chars):
    #                     b[i,j]= char

    #         return b

    # def get_neghbours(self, x,y):
    #     neghbour = []
    #     for i in range(0, 6):
    #         newNeghbour = [(self.x + self.I_DISPLACEMENTS[i]), (self.y + self.J_DISPLACEMENTS[i])]

    #         #not out of bounds
    #         if not (newNeghbour[0] < 0 or newNeghbour[0] >= self._board_size or
    #                 newNeghbour[1] < 0 or newNeghbour[1] >= self._board_size):

    #             neghbour.append(newNeghbour)

    #     return neghbour;


if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()
