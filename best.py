import socket
from random import choice
from time import sleep
import copy
import numpy as np
# import sys
import math



class AiAgent():
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.
    """

    HOST = "127.0.0.1"
    PORT = 1234

    def run(self):
        """A finite-state machine that cycles through waiting for input
        and sending moves.
        """
        
        self.MIN_VALUE = -math.inf
        self.MAX_VALUE = math.inf

        self.DEPTH = 2

        self.RED = "R"
        self.BLUE = "B"


        self._board_size = 0
        self._board = None
        self._colour = ""
        self._turn_count = 1
        self.winner = None
        # self._choices = []
        
        states = {
            1: AiAgent._connect,
            2: AiAgent._wait_start,
            3: AiAgent._make_move,
            4: AiAgent._wait_message,
            5: AiAgent._close
        }

        res = states[1](self)
        while (res != 0):
            res = states[res](self)

    def _connect(self):
        """Connects to the socket and jumps to waiting for the start
        message.
        """
        
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.connect((AiAgent.HOST, AiAgent.PORT))

        return 2

    def _wait_start(self):
        """Initialises itself when receiving the start message, then
        answers if it is Red or waits if it is Blue.
        """
        
        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        if (data[0] == "START"):
            self._board_size = int(data[1])

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
        # Swap randomly
        if (self._turn_count == 2 and choice([0, 0]) == 1):
            msg = "SWAP\n"
        else:
            move = self.getMove()
            print("-- bestValue; ", move[0])

            if (self._board[move[1], move[2]] == "0"):
                msg = f"{move[1]},{move[2]}\n"
            else:
                print("-----tried sending bad move-----")
                print(move[1], move[2])
                self._make_move()
        
        self._s.sendall(bytes(msg, "utf-8"))

        return 4

    def getMove(self):

        index = self.minmax(self._board, self.DEPTH, True, self.MIN_VALUE, self.MAX_VALUE)
        return index
    
    def minmax(self, board, depth, isMax, alpha, beta):
        nodeValue = None
        bestRow = -1
        bestCol = -1

        winner = self.get_winner(board)
        # print("--winner: ", winner)

        if(winner == self._colour):
            nodeValue = self.MAX_VALUE
            # TODO get best row and col
            print("--winner: ", winner)
            print(bestRow, bestCol)
            return nodeValue, bestRow, bestCol
	    
        elif(winner == self.opp_colour):
            nodeValue = self.MIN_VALUE
            print("--winner: ", winner)
            return nodeValue, bestRow, bestCol

        if(depth <= 0):
	        # Returning this state heuristic function value
	        # return val;
            a = self.get_heuristic(board);
            return a, bestRow, bestCol

        else:
            moves = self.get_available_moves(board)
            if (isMax):
                nodeValue = self.MIN_VALUE
                for move in moves:
                    updatedBoard = self.update_board(copy.deepcopy(board), move[0], move[1], self._colour)
                    nodeValue = max(nodeValue, self.minmax(
                        updatedBoard, depth - 1, not isMax, alpha, beta)[0])
                    if(nodeValue > alpha):
                        alpha = nodeValue;
                        bestRow = move[0] 
                        bestCol = move[1] 
                    if(beta <= alpha):
                        break
                    
                return alpha, bestRow, bestCol
            else:
                nodeValue = self.MAX_VALUE
                for move in moves:
                    updatedBoard = self.update_board(
                        copy.deepcopy(board), move[0], move[1], self.opp_colour())
                    nodeValue = min(nodeValue, self.minmax(
                        updatedBoard, depth - 1, not isMax, alpha, beta)[0])

                    if(nodeValue < beta):
                        beta = nodeValue;
                        bestRow = move[0]
                        bestCol = move[1]
                    if(beta <= alpha):
                        break
                    
                return beta, bestRow, bestCol

    def get_heuristic(self, board):
        score = 0

        # boolean[][] visited = new boolean[board.getSize()][board.getSize()]
        visitedBoard = np.full((self._board_size, self._board_size), False)
        score += self.findChain(board, self._colour, visitedBoard)
        return score
    
    def findChain(self, board, player, visited):
        chain = 0
        for i in range(self._board_size):
            for j in range(self._board_size):
                tempChain = 0

                if visited[i, j]:
                    continue

                for cell in self.get_neighbors((i, j)):
                    if cell == player and not visited[i, j]:
                        tempChain += 1

                        # TODO make sure it works correctly
                        # Blue player should play in X axis
                        if self._colour == self.BLUE and cell[0]==i:
                            tempChain += 2

                        if self._colour == self.RED and cell[1]==j:
                            tempChain += 2
                
                chain = max(chain, tempChain)
                tempChain = 0

        if(player == self._colour):
            chain *= 10
        
        return chain

        
    def _wait_message(self):
        """Waits for a new change message when it is not its turn."""

        self._turn_count += 1

        data = self._s.recv(1024).decode("utf-8").strip().split(";")
        if (data[0] == "END" or data[-1] == "END"):
            return 5
        
        else:
            #----- SWAP -----
            if (data[1] == "SWAP"):
                self._colour = self.opp_colour()
                #TODO switch to np.where
                # x, y = zip(np.where(self._board != "0"))
                for x in range(self._board_size):
                    for y in range(self._board_size):
                        if self._board[x][y] != "0":
                            if (self._board[x][y] == "R"):
                                self._board[x][y] = "B"
                            else:
                                self._board[x][y] = "R"
                            break

            else:
                #----- Update our local board -----
                x, y = data[1].split(",")
                if (data[-1] == self._colour):
                    self._board[int(x)][int(y)] = self.opp_colour()
                else:
                    self._board[int(x)][int(y)] = self._colour

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

    def update_board(self, updateboard, x, y, colour):
        updateboard[x][y] = colour
        return updateboard

    def get_available_moves(self, board):
        # TODO update to np.where()
        moves = []
        for x in range(self._board_size):
            for y in range(self._board_size):
                if board[x][y] == "0":
                    moves.append([x, y])

        return moves

    def get_winner(self, board):
        """Checks if the game has ended. It will attempt to find a red chain
        from top to bottom or a blue chain from left to right of the board.
        """
        self.winner = None
        self.visited = np.full((self._board_size, self._board_size), False)

        # Red
        # for all top tiles, check if they connect to bottom
        for idx in range(self._board_size):
            tile = board[0, idx]
            if (not self.visited[0, idx] and tile == self.RED and self.winner is None):
                self.DFS_colour(board, 0, idx, self.RED)
        
        # Blue
        # for all left tiles, check if they connect to right
        for idx in range(self._board_size):
            tile = board[idx, 0]
            if (not self.visited[idx, 0] and tile == self.BLUE and self.winner is None):
                self.DFS_colour(board, idx, 0, self.BLUE)

        return self.winner

    def DFS_colour(self, board, x, y, colour):
        """A recursive DFS method that iterates through connected same-colour
        tiles until it finds a bottom tile (Red) or a right tile (Blue).
        """
        I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
        J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]

        self.visited[x, y] = True

        # win conditions
        if (colour == self.RED):
            if (x == self._board_size-1):
                self.winner = colour
        elif (colour == self.BLUE):
            if (y == self._board_size-1):
                self.winner = colour
        else:
            return
        

        # end condition
        if (self.winner is not None):
            return

        # visit neighbours
        for idx in range(6):
            # TODO replace with get_neighbors
            x_n = x + I_DISPLACEMENTS[idx]
            y_n = y + J_DISPLACEMENTS[idx]
            if (x_n >= 0 and x_n < self._board_size and
                    y_n >= 0 and y_n < self._board_size):

                neighbour = board[x_n, y_n]
                if (not self.visited[x_n, y_n] and
                        neighbour == colour):
                    self.DFS_colour(board, x_n, y_n, colour)

    
    
    def get_neighbors(self, coordinates):
        """Gets all the neighboring cells of a given cell
        Args:
            coordinates (tuple): Cell (x, y) coordinates
        Returns:
            list: List of neighboring cell coordinates tuples
        """
        (x, y) = coordinates
        neighbors = []
        if x-1 >= 0:
            neighbors.append((x-1, y))
        if x+1 < self._board_size:
            neighbors.append((x+1, y))
        if x-1 >= 0 and y+1 <= self._board_size-1:
            neighbors.append((x-1, y+1))
        if x+1 < self._board_size and y-1 >= 0:
            neighbors.append((x+1, y-1))
        if y+1 < self._board_size:
            neighbors.append((x, y+1))
        if y-1 >= 0:
            neighbors.append((x, y-1))

        return neighbors

if (__name__ == "__main__"):
    agent = AiAgent()
    agent.run()
