import socket
from random import choice
from time import sleep
import math

//https://gsurma.medium.com/hex-creating-intelligent-opponents-with-minimax-driven-ai-part-1-%CE%B1-%CE%B2-pruning-cc1df850e5bd

class NaiveAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234

    def run(self):
        """A finite-state machine that cycles through waiting for input
        and sending moves.
        """
        
        self._board_size = 0
        self._colour = ""
        self._turn_count = 1
        self._choices = []
        
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
            self._board = [["" for i in range(self._board_size)] for j in range(self._board_size)]
            for i in range(self._board_size):
                for j in range(self._board_size):
                    self._choices.append((i, j))
            self._colour = data[2]

            print(self._board)

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
            move = choice(self._choices)
            msg = f"{move[0]},{move[1]}\n"
            
        
        self._s.sendall(bytes(msg, "utf-8"))

        return 4

    def _wait_message(self):
        """Waits for a new change message when it is not its turn."""

        self._turn_count += 1

        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        data2 = data[2].strip().split(",")
        print(data)
        if (data[0] == "END" or data[-1] == "END"):
            return 5
        else:

            if (data[1] == "SWAP"):
                self._colour = self.opp_colour()
            else:
                x, y = data[1].split(",")
                self._choices.remove((int(x), int(y)))
                self._board[int(x)][int(y)] = 1
                print(self._board)

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

    def MinMax(self, board, maximizingPlayer, depth, alpha, beta ):
        alpha = alpha
        beta = beta

        if depth == 0 or board.getIsBoardFull():
            return board
        

        if maximizingPlayer:
            bestValue= -math.inf
            for move in self._choices:
                updatedBoard = board
                bestValue = max(bestValue, self.MinMax(updatedBoard, False, depth-1, alpha, beta))
                beta = min(beta, bestValue)
                if beta <= alpha:
                    break
            return bestValue
        else: 
            bestValue= math.inf
            for move in self._choices:
                updatedBoard = board
                bestValue = min(bestValue, self.MinMax(updatedBoard, True, depth-1, alpha, beta))
                beta = min(beta, bestValue)
                if beta <= alpha:
                        break
            return bestValue




if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()
