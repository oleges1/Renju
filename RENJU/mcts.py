import abc
import numpy as np
import numpy
import subprocess
import util
import renju
from agent import Agent
from time import time

class Node():
    def __init__(self, board, color, black_model, white_model):
        self._color = color
        self._black_model = black_model
        self._white_model = white_model
        
        if color == 'black': 
            with black_model[1].as_default():
                self._P = black_model[0].predict(board.reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        else:
            with white_model[1].as_default():
                self._P = white_model[0].predict(board.reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        self._visited = 0
        self._N = np.zeros(225, dtype = np.float32)
        self._R = np.zeros(225, dtype = np.float32)
        self._children = [None for i in range(225)]
    
    def travail(self, board, move):

        #if (self._visited):
        #    return False

        parsed_move = np.unravel_index(move, (15, 15))
        #if (board[parsed_move] != 0):
        #    return False
        
        board[parsed_move] = 1
        next_color = 'white'
        if (self._color == 'white'):
            board[parsed_move] = -1
            next_color = 'black'

        child = Node(board, next_color, self._black_model, self._white_model)
        self._children[move] = child
        #self._visited = 1
        return True


class MCTS(Agent):
    def __init__(self, black_model, white_model, black_rollout, white_rollout, name, color = None, high = 10, \
                 gamma = 1.0, samples = None, timeout = None, verbose = 0, min_prob = 0.8, param1 = 0.25, param2 = 0.85):
        self._name = name
        self._node_model_black = black_model
        self._node_model_white = white_model
        self._black_model = black_rollout[0]
        self._black_graph = black_rollout[1]
        self._white_model = white_rollout[0]
        self._white_graph = white_rollout[1]
        if (timeout == None):
            self._timeout = 666
        else:
            self._timeout = timeout
        if samples == None:
            self._samples = 100000
        else:
            self._samples = samples
        self._high = high
        self._color = color
        self._verbose = verbose
        self._gamma = gamma
        self._iters = 0.0
        self._root = None
        self._board = None
        self._start_time = None
        self._min_prob = min_prob
        self._param1 = param1
        self._param2 = param2
    
    def name(self):
        return self._name


    def rollout(self, board, temp_high, color):
        gamma = self._gamma
        
        if (len(util.list_positions(board, renju.Player.NONE)) == 0):
            return 0
        
        temp_predictions = 0
        if color == 'black': 
            with self._black_graph.as_default():
                temp_predictions = self._black_model.predict(board.reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        else:
            with self._white_graph.as_default():
                temp_predictions = self._white_model.predict(board.reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
            
        temp_pos = numpy.random.choice(225, p = temp_predictions)
        temp_parsed_pos = numpy.unravel_index(temp_pos, (15, 15))

            
        if (board[temp_parsed_pos] != 0):
            if (temp_high % 2):
                return gamma ** (temp_high - 1)
            else:
                return -gamma ** (temp_high - 1)
            
        board[temp_parsed_pos] = 1
        next_color = 'white'
        if color == 'white':
            next_color = 'black'
            board[temp_parsed_pos] = -1

        if (util.check(board, temp_parsed_pos)):
            if (temp_high % 2):
                return -gamma ** (temp_high - 1)
            else:
                return gamma ** (temp_high - 1)

        if (temp_high < self._high):
            return self.rollout(board, temp_high + 1, next_color)
        return 0
    
    def list_visiting(self, ucb_eps = 0.01):
        temp_high = -1
        reward = 0
        path = []
        temp_root = self._root
        gamma = self._gamma
        board = np.copy(self._board)

        if (self._root == None):
            raise MAMA_MIA('root is None')

        while temp_high < self._high:
            temp_high += 1
            if (temp_root._visited):
                #values = (temp_root._R) / (1 + temp_root._N)
                #values -= values.mean()
                #if values.std():
                #    values /= values.std()
                ucb = ucb_eps * np.sqrt(2 * np.log(temp_root._N + 1) / (1 + self._iters + temp_root._N.sum()))
                #ucb -= ucb.mean()
                #if ucb.std():
                #    ucb /= ucb.std()

                #temp_pos = numpy.argmax(values + ucb + temp_root._P)
                values = (temp_root._R + 10 * temp_root._P) / (1 + temp_root._N) + (temp_root._R > self._param1 * temp_root._N) + (temp_root._R > self._param2 * temp_root._N)
                temp_pos = numpy.argmax(values + ucb)
                temp_parsed_pos = numpy.unravel_index(temp_pos, (15, 15))

                path.append(temp_pos)
                
                if (board[temp_parsed_pos] != 0):
                    if (temp_high % 2):
                        return path, gamma ** (temp_high - 1)
                    else:
                        return path, -gamma ** (temp_high - 1)
                
                if (temp_root._color == 'black'):
                    board[temp_parsed_pos] = 1
                else:
                    board[temp_parsed_pos] = -1

                if (util.check(board, temp_parsed_pos)):
                    if (temp_high % 2):
                        return path, -gamma ** (temp_high - 1)
                    else:
                        return path, gamma ** (temp_high - 1)
            
                if not temp_root._children[temp_pos]:
                    temp_root.travail(board, temp_pos)

                temp_root = temp_root._children[temp_pos]
            else:
                temp_root._visited = 1
                
                temp_pos = numpy.random.choice(225, p = temp_root._P)
                temp_parsed_pos = numpy.unravel_index(temp_pos, (15, 15))
                path.append(temp_pos)
                
                if (board[temp_parsed_pos] != 0):
                    if (temp_high % 2):
                        return path, gamma ** (temp_high - 1)
                    else:
                        return path, -gamma ** (temp_high - 1)

                board[temp_parsed_pos] = 1
                next_color = 'white'
                if temp_root._color == 'white':
                    next_color = 'black'
                    board[temp_parsed_pos] = -1

                if (util.check(board, temp_parsed_pos)):
                    if (temp_high % 2):
                        return path, -gamma ** (temp_high - 1)
                    else:
                        return path, gamma ** (temp_high - 1)

                if not temp_root._children[temp_pos]:
                    temp_root.travail(board, temp_pos)
                
                if (temp_high < self._high):
                    return path, self.rollout(board, temp_high + 1, next_color)
                return path, 0

        return path, reward
        
    def update(self, path, reward):
        current = self._root
        for action in path:
            if current == None:
                raise MAMA_MIA('current is None')

            if current._color == self._color:
                current._R[action] += reward
            else:
                current._R[action] -= reward
            current._N[action] += 1
            current = current._children[action]

        del path
    
    def tree_search(self):
        self._iters = 0
        while (time() - self._start_time < self._timeout and self._iters < self._samples):
            self._iters += 1
            path, reward = self.list_visiting()
            self.update(path, reward)
        return self._iters
    
    def make_move(self, move):
        if (self._root == None):
            raise MAMA_MIA('root is None')
        
        if not self._root._children[move]:
            return False
        
        self._root = self._root._children[move]
        return True
        
    
    def policy(self, game):
        self._start_time = time()
        if not self._color:
            if ((225 - len(util.list_positions(game.board(), renju.Player.NONE))) % 2 == 1):
                self._color = 'white'
            else:
                self._color = 'black'
        if (len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[112] = 1
            return res.reshape((1, 225))
        
        done = True
        if (self._root and len(util.list_positions(game.board(), renju.Player.NONE)) < 224):
            to_do_moves = game._positions[-2:]
            for move in to_do_moves:
                if not self.make_move(move[0] * 15 + move[1]):
                    done = False
        else:
            done = False
    
        if self._verbose:
            print(done)

        checker = np.zeros(225)
        available = numpy.zeros(225)
        self._board = np.copy(-game.board())
        for parsed_pos in util.list_positions(self._board, renju.Player.NONE):
            pos = parsed_pos[0] * 15 + parsed_pos[1]
            parsed_pos = tuple(parsed_pos)
            
            self._board[parsed_pos] = 1
            if (util.check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = -1
            if (util.check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = 0
            available[pos] = 1
        
        if not done:
            self._root = Node(self._board, self._color, self._node_model_black, self._node_model_white)
        
        self.tree_search()
        #norm_values = (self._root._R) / (1 + self._root._N)
        #norm_values -= norm_values.mean()
        #if norm_values.std():
        #    norm_values /= norm_values.std()
        #
        #values = (norm_values + self._root._P * 3) * available * (1 + checker)
        values = (self._root._N > self._iters / 5) * self._root._R / (1 + self._root._N) * available * (1 + 10 * checker)
        
        if np.max(values) > self._min_prob:
            code_move = np.argmax(values)
        else:
            code_move = np.argmax((self._root._N) * available * (1 + 10 * checker))
        
        res = np.zeros(225)
        res[code_move] = 1
        
        if (self._verbose):
            for elem in np.where(self._root._R != 0)[0]:
                try:
                    print(util.to_move(numpy.unravel_index(int(elem), (15, 15))), 'R:', self._root._R[int(elem)], 'V:', values[int(elem)])
                except:
                    print(np.where(self._root._R != 0))
                    continue
            
            # code_move = numpy.argmax(values.reshape(1, 225))
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]), \
                  'working time:', time() - self._start_time, 'iterations:', self._iters)

        #values -= values.mean()
        #values /= values.std()
        return res.reshape(1, 225)

"""
    def policy_test(self, board):
        self._start_time = time()
        if not self._color:
            if ((225 - len(util.list_positions(game.board(), renju.Player.NONE))) % 2 == 1):
                self._color = 'white'
            else:
                self._color = 'black'
        if (len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[112] = 1
            return res.reshape((1, 225))
        
        checker = np.zeros(225)
        available = numpy.zeros(225)
        self._board = np.copy(-game.board())
        for parsed_pos in util.list_positions(self._board, renju.Player.NONE):
            pos = parsed_pos[0] * 15 + parsed_pos[1]
            parsed_pos = tuple(parsed_pos)
            
            self._board[parsed_pos] = 1
            if (util.check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = -1
            if (util.check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = 0
            available[pos] = 1
        
        self._root = Node(self._board, self._color, self._node_model_black, self._node_model_white)
        
        self.tree_search()
        #norm_values = (self._root._R) / (1 + self._root._N)
        #norm_values -= norm_values.mean()
        #if norm_values.std():
        #    norm_values /= norm_values.std()
        #
        #values = (norm_values + self._root._P * 3) * available * (1 + checker)
        values = (self._root._N > self._iters / 5) * self._root._R / (1 + self._root._N) * available * (1 + 10 * checker)
        
        if np.max(values) > self._min_prob:
            code_move = np.argmax(values)
        else:
            code_move = np.argmax((self._root._N) * available * (1 + 10 * checker))
        
        
        if (self._verbose):
            for elem in np.where(self._root._R != 0)[0]:
                try:
                    print(util.to_move(numpy.unravel_index(int(elem), (15, 15))), 'R:', self._root._R[int(elem)], 'V:', values[int(elem)])
                except:
                    print(np.where(self._root._R != 0))
                    continue
            
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]), \
                  'working time:', time() - self._start_time, 'iterations:', self._iters)

        return numpy.unravel_index(code_move, (15, 15))"""