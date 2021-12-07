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
        self._seen_board = dict()
        self.eval_count = 0
        self.depth = 2
        
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
            # self._board = [["0" for i in range(self._board_size)] for j in range(self._board_size)]
            self._board = np.full((self._board_size,self._board_size), "0")
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
            value = self.MinMax(newBoard, True, self.depth, -math.inf, math.inf, [0,0])
            # print(value)

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
    def opp_this_colour(self, colour):
        """Returns the char representation of the colour opposite to the
        current one.
        """
        
        if colour == "R":
            return "B"
        elif colour == "B":
            return "R"
        else:
            return "0"

    def MinMax(self, startBoard, maximizingPlayer, depth, alpha, beta, currentMove ):
        alpha = alpha
        beta = beta
        iterationBestMove = []


        if depth == 0 or  not (any("0" in subl for subl in startBoard)):
            turn_colour = self.opp_colour
            if maximizingPlayer:
                turn_colour = self._colour

            if np.array2string(self._board) in self._seen_board:
                return self._seen_board[np.array2string(self._board)]
            else:
                score =  self.eval_dijkstra(turn_colour)
                # print(score)
                self._seen_board[np.array2string(self._board)] = score
                return score
            
            # return choice(range(-10,10))  #return heuristic  
        
    
  
        if maximizingPlayer:
            bestValue= -math.inf
            moves = self.get_moves(startBoard)
            for move in moves:
                # print(move)
                # updatedBoard = self.update_board(copy.deepcopy(startBoard),move[0], move[1], self._colour)
                self._board[move[0]][move[1]] = self._colour
                value = self.MinMax(self._board, False, depth-1, alpha, beta, move)
                self._board[move[0]][move[1]] = "0"
                if value > bestValue:
                    iterationBestMove = move
                    bestValue = value
                    # print(bestValue)
                alpha = max(alpha, bestValue)
                if bestValue >= beta:
                    break
            self._best_move = iterationBestMove
            return bestValue
        else: 
            bestValue= math.inf
            moves = self.get_moves(startBoard)
            for move in moves:
                self._board[move[0]][move[1]] = self.opp_colour()
                # updatedBoard = self.update_board(copy.deepcopy(startBoard),move[0], move[1], "B")
                value = self.MinMax(self._board, True, depth-1, alpha, beta, move)
                self._board[move[0]][move[1]] = "0"
                if value < bestValue:
                    iterationBestMove = move
                    bestValue = value
                beta = min(beta, bestValue)
                if bestValue <= alpha:
                        break
            self._best_move = iterationBestMove
            return bestValue

    def eval_dijkstra(self, color):
        """Evaluation based on distance to border. Score computed by taking difference of dijkstra score of opponent and current player

        Returns:
            int: Board evaluation score
        """
        self.eval_count += 1
        return  self.get_dijkstra_score(self.opp_this_colour(color)) - self.get_dijkstra_score(color)

    def dijkstra_update(self, color, scores, updated):
        """Updates the given dijkstra scores array for given color
        Args:
            color (HexBoard.color): color to evaluate
            scores (int array): array of initial scores
            updated (bool array): array of which nodes are up-to-date (at least 1 should be false for update to do something)
        Returns:
            the updated scores
        """
        # print("Starting dijkstra algorithm")
        updating = True
        while updating: 
            updating = False
            for i, row in enumerate(scores): #go over rows
                for j, point in enumerate(row): #go over points 
                    if not updated[i][j]: 
                        neighborcoords = self.get_neighbors((i,j))  
                        for neighborcoord in neighborcoords:
                            target_coord = tuple(neighborcoord)
                            path_cost = LOSE #1 for no color, 0 for same color, INF for other color 
                            if self.is_empty(target_coord):
                                path_cost = 1
                            elif self.is_color(target_coord, color):
                                path_cost = 0
                            
                            if scores[target_coord] > scores[i][j] + path_cost: #if new best path to this neighbor
                                scores[target_coord] = scores[i][j] + path_cost #update score
                                updated[target_coord] = False #This neighbor should be updated
                                updating = True #make sure next loop is started
        return scores
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
        alignment = (1, 0) if color == "B" else (0, 1)


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
                   for i in range(self._board_size)]  # take "other side" to get the list of distance from end-end on board
        best_result = min(results)
      
        # if best_result == 0:
        #     best_result = -500

        # print("Best score for color {}: {}".format(color, best_result))
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
        return self._board[coordinates] == "0"

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
        return self._board[coordinates] == color



    def get_moves(self, board):
        
        # moves = []
        # for x in range(self._board_size):
        #     for y in range(self._board_size):
        #         if board[x][y] == "0":
        #             moves.append([x,y])

        moves = np.where(board == "0")
        moves = list(zip(moves[0], moves[1]))

        return moves

    def update_board(self, updateboard, x,y, colour):
        updateboard[x][y] = colour
        return updateboard


    # def get_neighbors(self, x,y):
    #     neighbor = []
    #     for i in range(0, 6):
    #         newNeighbor = [(self.x + self.I_DISPLACEMENTS[i]), (self.y + self.J_DISPLACEMENTS[i])]

    #         #not out of bounds
    #         if not (newNeighbor[0] < 0 or newNeighbor[0] >= self._board_size or
    #                 newNeighbor[1] < 0 or newNeighbor[1] >= self._board_size):
             
    #             neighbor.append(newNeighbor)

    #     return neighbor;     


    def get_neighbors(self, coordinates):
        """Gets all the neighbouring cells of a given cell
        Args:
            coordinates (tuple): Cell (x, y) coordinates
        Returns:
            list: List of neighbouring cell coordinates tuples
        """
        (x,y) = coordinates
        neighbors = []
        if x-1 >= 0: neighbors.append((x-1,y))
        if x+1 < self._board_size: neighbors.append((x+1,y))
        if x-1 >= 0 and y+1 <= self._board_size-1: neighbors.append((x-1,y+1))
        if x+1 < self._board_size and y-1 >= 0: neighbors.append((x+1,y-1))
        if y+1 < self._board_size: neighbors.append((x,y+1))
        if y-1 >= 0: neighbors.append((x,y-1))

        return neighbors

    def get_heuristic(self, move):
        if self._board.tobytes() in self._seen_board:
            return self._seen_board[self._board.tobytes()]
        else:
            myPath = self.beam(self._colour, move)
            oppPath = self.beam(self.opp_colour(), move)
            self._seen_board[self._board.tobytes()] = myPath - oppPath
            return self._seen_board[self._board.tobytes()]


    def beam(self, colour, move): 
        count = 0
        if(colour == "R"):
            count = (self._board[:, move[1]] == "R").sum()
        else:
            count = (self._board[move[0], :] == "B").sum()

        return count
            

    # def dijkstra(self, src, dst):
    #     stat_edges_explored = 0

    #     g = self._board

    #     assert(src < graph.INVALID_NODE)
    #     N = g.graph_get_num_nodes()
    #     assert(N < graph.INVALID_NODE)
    #     pred = [graph.INVALID_NODE]*N
    #     dist = [weight.weight_inf()]*N
    #     dist[src] = weight.weight_zero()
    #     pred[src] = src

    #     #create a priority queue for the discoverd nodes
    #     d = pq.DPQ_t(N)
    #     d.DPQ_insert(src, weight.weight_zero())
    #     f = []


    #     while not d.DPQ_is_empty():
    #         #pop the minimal dist node from the queue
    #         u = d.DPQ_pop_min()
    #         f.append(u)
    #         for v in g.get_graph_succs(u):
    #             stat_edges_explored += 1
    #             if(v.v not in f ):
    #                 if(v.w.weight_is_finite()):
    #                 if weight.weight_less(weight.weight_add(v.w ,dist[u]), dist[v.v]):
    #                     dist[v.v] = weight.weight_add(v.w ,dist[u])
    #                     pred[v.v] = u
    #                     #check if the node is not already discoverd the add it to the discoverd queue
    #                     if (not d.DPQ_contains(v.v)):
    #                         d.DPQ_insert(v.v, dist[v.v])
    #                     else:
    #                         d.DPQ_decrease_key(v.v, dist[v.v])


    #     return sssp_result_t(N, src, dst, False, pred, dist, stat_edges_explored)





if (__name__ == "__main__"):
    agent = NaiveAgent()
    agent.run()
