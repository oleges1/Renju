#!/usr/bin/python3.4

# independent file, contains all needed functions
# for competition

import numpy as np
import tensorflow as tf
import itertools

import keras
from keras.models import load_model
import abc
import numpy as np
import numpy
import subprocess
from time import time
import logging
import os
import sys

POS_TO_LETTER = 'abcdefghjklmnop'
LETTER_TO_POS = {letter: pos for pos, letter in enumerate(POS_TO_LETTER)}

def to_move(pos):
    return POS_TO_LETTER[pos[1]] + str(pos[0] + 1)

def to_pos(move):
    return int(move[1:]) - 1, LETTER_TO_POS[move[0]]

def list_positions(board, player):
    return numpy.vstack(numpy.nonzero(board == player)).T

def sequence_length(board, I, J, value):
    length = 0

    for i, j in zip(I, J):
        if board[i, j] != value:
            break
        length += 1

    return length


def check_horizontal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j + 1, min(j + 5, 15)),
        player
    )

    length += sequence_length(
        board,
        itertools.repeat(i),
        range(j - 1, max(j - 5, -1), -1),
        player
    )

    return length >= 5

def check_vertical(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + 5, 15)),
        itertools.repeat(j),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - 5, -1), -1),
        itertools.repeat(j),
        player
    )

    return length >= 5

def check_main_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i + 1, min(i + 5, 15)),
        range(j + 1, min(j + 5, 15)),
        player
    )

    length += sequence_length(
        board,
        range(i - 1, max(i - 5, -1), -1),
        range(j - 1, max(j - 5, -1), -1),
        player
    )

    return length >= 5

def check_side_diagonal(board, pos):
    player = board[pos]
    if not player:
        return False

    i, j = pos
    length = 1

    length += sequence_length(
        board,
        range(i - 1, max(i - 5, -1), -1),
        range(j + 1, min(j + 5, 15)),
        player
    )

    length += sequence_length(
        board,
        range(i + 1, min(i + 5, 15)),
        range(j - 1, max(j - 5, -1), -1),
        player
    )

    return length >= 5

def check(board, pos):
    if not board[pos]:
        return False

    return check_vertical(board, pos) \
        or check_horizontal(board, pos) \
        or check_main_diagonal(board, pos) \
        or check_side_diagonal(board, pos)

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
        parsed_move = np.unravel_index(move, (15, 15))
        
        board[parsed_move] = 1
        next_color = 'white'
        if (self._color == 'white'):
            board[parsed_move] = -1
            next_color = 'black'

        child = Node(board, next_color, self._black_model, self._white_model)
        self._children[move] = child


class MCTS():
    def __init__(self, name, black_model, white_model, black_rollout, white_rollout, color = None, high = 10, \
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
        
        if (len(list_positions(board, 0)) == 0):
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

        if (check(board, temp_parsed_pos)):
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
                # values = (temp_root._R) / (1 + temp_root._N)
                # values -= values.mean()
                # if values.std():
                #    values /= values.std()
                ucb = ucb_eps * np.sqrt(2 * np.log(temp_root._N + 1) / (1 + self._iters + temp_root._N.sum()))
                # ucb = ucb_eps * np.sqrt(2 * np.log(temp_root._N + 1) / (1 + self._iters))
                # ucb -= ucb.mean()
                # if ucb.std():
                #    ucb /= ucb.std()

                # temp_pos = numpy.argmax(values + ucb + temp_root._P)
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

                if (check(board, temp_parsed_pos)):
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

                if (check(board, temp_parsed_pos)):
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
        logging.debug('tree_search_finished')
        return self._iters

    def make_move(self, move):
        if (self._root == None):
            raise MAMA_MIA('root is None')
        
        if not self._root._children[move]:
            return False
        
        self._root = self._root._children[move]
        logging.debug('root_changed:' + to_move(numpy.unravel_index(move, (15, 15))))
        return True

    def policy_test(self, board, positions):
        self._start_time = time()
        if not self._color:
            if ((225 - len(list_positions(board, 0))) % 2 == 1):
                self._color = 'white'
            else:
                self._color = 'black'

        if (self._verbose):
            logging.debug(self._color)

        if (len(list_positions(board, 0)) == 225):
            # first move in center of board
            # 119 - for move in center of right column
            return numpy.unravel_index(112, (15, 15))

        done = True
        if (self._root and len(list_positions(board, 0)) < 224):
            # to_do_moves = game._positions[-2:]
            for move in positions[-2:]:
                pos = to_pos(move)
                if not self.make_move(pos[0] * 15 + pos[1]):
                    done = False
        else:
            done = False

        if self._verbose:
            logging.debug('Node visited? - ' + str(done))
        
        checker = np.zeros(225)
        available = numpy.zeros(225)
        self._board = np.copy(board)
        for parsed_pos in list_positions(self._board, 0):
            pos = parsed_pos[0] * 15 + parsed_pos[1]
            parsed_pos = tuple(parsed_pos)
            
            self._board[parsed_pos] = 1
            if (check(self._board, parsed_pos)):
                checker[pos] += 1
            self._board[parsed_pos] = -1
            if (check(self._board, parsed_pos)):
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
        
        
        #if (self._verbose):
        #    for elem in np.where(self._root._R != 0)[0]:
        #        try:
        #            logging.debug(to_move(numpy.unravel_index(int(elem), (15, 15))), 'R:', self._root._R[int(elem)], 'V:', values[int(elem)])
        #        except:
        #            logging.debug(np.where(self._root._R != 0))
        #            continue
        #    
        logging.debug(" ".join([self._name + ':', str(to_move(numpy.unravel_index(code_move, (15, 15)))), \
                  'working time:', str(time() - self._start_time), 'iterations:', str(self._iters)]))

        return numpy.unravel_index(code_move, (15, 15))

if __name__ == "__main__":
    pid = os.getpid()
    LOG_FORMAT = '%(levelname)s:%(asctime)s: KOPATYCH_KOPAET-{0}: %(message)s'.format(pid)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

    papa_black = load_model('kopatych_papa_black4096.h5')
    papa_black_graph = tf.get_default_graph()
    papa_white = load_model('kopatych_papa_white4096.h5')
    papa_white_graph = tf.get_default_graph()

    logging.debug('models loaded')

    mcts = MCTS(name = 'KOPATYCH_KOPAET',
            black_model = (papa_black, papa_black_graph), white_model = (papa_white, papa_white_graph),
                    black_rollout = (papa_black, papa_black_graph), white_rollout = (papa_white, papa_white_graph), 
                           timeout = 14.85, high = 14, gamma = 0.99, verbose = 1, min_prob = 0.8, param1 = 0.2, param2 = 0.65)

    logging.debug('mcts created')

    board = np.zeros((15, 15))

    while True:
        game = sys.stdin.readline()
        if game:
            point = 1
            if not game == "\n":
                for move in game.split(' '):
                    # logging.debug(move)
                    board[to_pos(move)] = point
                    point *= -1

            logging.debug('board parsed')

            res = mcts.policy_test(board, game.split(' '))
            logging.debug('result move:' + to_move(res))
            if sys.stdout.closed:
                logging.debug('sys closed')
            sys.stdout.write(to_move(res) + '\n')
            sys.stdout.flush()
        else:
            logging.debug('finishing...')
            break