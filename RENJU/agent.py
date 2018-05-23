import abc
import numpy
import subprocess
import util
import renju
from time import time

class Agent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def policy(game):
        '''Return probabilty matrix of possible actions'''

    @abc.abstractmethod
    def name():
        '''return name of agent'''

class HumanAgent(Agent):
    def __init__(self, name='Human'):
        self._name = name

    def name(self):
        return self._name

    def policy(self, game):
        move = input()
        pos = util.to_pos(move)

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs

class CnnAgent(Agent):
    def __init__(self, color, name, model, verbose = 0):
        self._name = name
        self._color = color
        self._verbose = verbose
        self._model = model[0]
        self._graph = model[1]
        # self.model = load_model(color + '.h5')

    def name(self):
        return self._name

    def policy(self, game):
        from time import time
        if (self._color == 'black' and len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[112] = 1
            return res.reshape((1, 225))
        
        board = numpy.copy(-game.board())
        available = numpy.zeros(225)
        checker = numpy.zeros(225)
        
        for parsed_pos in util.list_positions(game.board(), renju.Player.NONE):
            pos = parsed_pos[0] * 15 + parsed_pos[1]
            parsed_pos = tuple(parsed_pos)
            
            board[parsed_pos] = 1
            if (util.check(board, parsed_pos)):
                checker[pos] += 1
            board[parsed_pos] = -1
            if (util.check(board, parsed_pos)):
                checker[pos] += 1

            board[parsed_pos] = 0
            available[pos] = 1
        
        start = time()
        with self._graph.as_default():
            predictions = self._model.predict(board.reshape(1, 15, 15, 1))[0]

        arr = (predictions * available) * (1 + checker)

        code_move = numpy.argmax(arr)
        if (self._verbose):
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]), time() - start)
        return arr.reshape(1, 225)

# sums policy agents, better to use obvious policy agent
class ComplexCnnAgent(Agent):
    def __init__(self, color, name, models, verbose = 0):
        self._name = name
        self._color = color
        self._models = models
        self._verbose = verbose

    def name(self):
        return self._name

    def policy(self, game):
        if (self._color == 'black' and len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[142] = 1
            return res.reshape((1, 225))
        predictions = numpy.zeros((1, 225))
        for model in self._models:
            with model[1].as_default():
                predictions += model[0].predict(-game.board().reshape(1, 15, 15, 1))
        available = numpy.zeros((225, 1))
        positions = util.list_positions(game.board(), renju.Player.NONE)
        for pos in positions:
            available[pos[0] * 15 + pos[1]] = 1
        arr = predictions.T + available
        code_move = numpy.argmax(arr)
        if (self._verbose):
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]))
        return arr

class RandomBeamSearchAgent(Agent):
    def __init__(self, black_model, white_model, name, color, high = 10, 
                 gamma = 0.96, fine = 1, bonus = 1, samples = None, timeout = None, verbose = 0):
        self._name = name
        self._black_model = black_model[0]
        self._black_graph = black_model[1]
        self._white_model = white_model[0]
        self._white_graph = white_model[1]
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
        self._gamma = gamma
        self._fine = fine
        self._bonus = bonus
        self._verbose = verbose

    def name(self):
        return self._name

    def policy(self, game):
        start = time()
        if (self._color == 'black' and len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[112] = 1
            return res.reshape((1, 225))
        predictions = 0
        temp_high = 0
        
        checker = numpy.zeros(225)
        available = numpy.zeros(225)
        board = numpy.copy(-game.board())
        for parsed_pos in util.list_positions(game.board(), renju.Player.NONE):
            pos = parsed_pos[0] * 15 + parsed_pos[1]
            parsed_pos = tuple(parsed_pos)
            
            board[parsed_pos] = 1
            if (util.check(board, parsed_pos)):
                checker[pos] += 1
            board[parsed_pos] = -1
            if (util.check(board, parsed_pos)):
                checker[pos] += 1
            board[parsed_pos] = 0
            available[pos] = 1
        
        if self._color == 'black': 
            with self._black_graph.as_default():
                predictions = self._black_model.predict(board.reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        else:
             with self._white_graph.as_default():
                predictions = self._white_model.predict(board.reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]

        logs = numpy.zeros(225, dtype = numpy.float32)
        res = numpy.zeros(225, dtype = numpy.float32)
        n = numpy.zeros(225, dtype = numpy.float32)
        
        def rollout(board, temp_high, color, checking_pos, sum_logs, max_high = self._high, gamma = self._gamma,
                       fine = self._fine, bonus = self._bonus):
            
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
                if (temp_high % 2 == 0):
                    res[checking_pos] += fine * gamma ** (temp_high - 1)
                else:
                    res[checking_pos] -= bonus * gamma ** (temp_high - 1)
                return
            
            board[temp_parsed_pos] = 1
            next_color = 'white'
            if color == 'white':
                next_color = 'black'
                board[temp_parsed_pos] = -1
            
            if (util.check(board, temp_parsed_pos)):
                if (temp_high % 2 == 0):
                    res[checking_pos] -= fine * gamma ** (temp_high - 1)
                else:
                    res[checking_pos] += bonus * gamma ** (temp_high - 1)
                return

            if (temp_high < max_high):
                rollout(board, temp_high + 1, next_color, checking_pos, sum_logs + numpy.log(temp_predictions[temp_pos]))
            else:
                if logs[checking_pos]:
                    logs[checking_pos] = max(logs[checking_pos], sum_logs + numpy.log(temp_predictions[temp_pos]))
                else:
                    logs[checking_pos] = sum_logs + numpy.log(temp_predictions[temp_pos])
            return
        
        i = 0
        while (time() - start < self._timeout and i < self._samples):
            i += 1
            pos = numpy.random.choice(225, p = predictions)
            parsed_pos = numpy.unravel_index(pos, (15, 15))
            n[pos] += 1
            
            if (board[parsed_pos] != 0):
                res[pos] -= 1
                continue
            
            board[parsed_pos] = 1
            color = 'white'
            if self._color == 'white':
                color = 'black'
                board[parsed_pos] = -1
            
            if (util.check(board, parsed_pos)):
                res[pos] += res.max() + 1000
                break
            
            rollout(numpy.copy(board), 1, color, pos, numpy.log(predictions[pos]))
            board[parsed_pos] = 0

        for indx in numpy.where(logs == 0.0):
            logs[indx] = logs.min() - 100.0
        
        value = (logs) * available *  (1 + checker)
        
        #print(value)
        value /= value.sum()
        
        if numpy.max(value) > 0.8:
            code_move = numpy.argmax(value)
        else:
            if (self._verbose):
                print('----!!!----')
            code_move = numpy.argmax((n + res) * available * (1 + checker))
        
        ans = numpy.zeros(225)
        ans[code_move] = 1

        if (self._verbose):
            for elem in numpy.where(res != 0)[0]:
                print(util.to_move(numpy.unravel_index(int(elem), (15, 15))), 'R:', res[int(elem)], 'V:', value[int(elem)], 'N:', n[int(elem)])
            
            # code_move = numpy.argmax(values.reshape(1, 225))
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]), \
                  'working time:', time() - start, 'iterations:', i)

        return ans.reshape(1, 225)

class BackendAgent(Agent):
    def __init__(self, backend, name='BackendAgent', **kvargs):
        self._name = name
        self._backend = subprocess.Popen(
            backend.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            **kvargs
        )

    def name(self):
        return self._name

    def send_game_to_backend(self, game):
        data = game.dumps().encode()
        self._backend.stdin.write(data + '\n')
        self._backend.stdin.flush()

    def wait_for_backend_move(self):
        data = self._backend.stdout.readline().rstrip()
        return data.decode()

    def policy(self, game):
        self.send_game_to_backend(game)
        pos = util.to_pos(self.wait_for_backend_move())

        probs = numpy.zeros(game.shape)
        probs[pos] = 1.0

        return probs

# deprecated agent
class BeamSearchAgent(Agent):
    def __init__(self, black_model, white_model, name, width, color, high, verbose = 0):
        self._name = name
        self._black_model = black_model[0]
        self._black_graph = black_model[1]
        self._white_model = white_model[0]
        self._white_graph = white_model[1]
        self._width = width
        self._high = high
        self._color = color
        self._verbose = verbose

    def name(self):
        return self._name

    def policy(self, game):
        if (self._color == 'black' and len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[142] = 1
            return res.reshape((1, 225))
        predictions = 0
        temp_high = 0
        
        if self._color == 'black':
            #positions = util.list_positions(game.board(), renju.Player.NONE)
            #if (len(positions) == 225):
                
            with self._black_graph.as_default():
                predictions = self._black_model.predict(-game.board().reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        else:
             with self._white_graph.as_default():
                predictions = self._white_model.predict(-game.board().reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        
        max_sum_log = -100000000000000
        best_move = -1
        results = dict()
        danger = set()

        def check_state_white(board, pos, checking_pos, temp_prob, temp_high, results):
            # print(board, pos)
            temp_high += 1
            parsed_pos = numpy.unravel_index(pos, (15, 15))
            #if (util.check(-board, parsed_pos)):
                #if self._color == 'white':
                    #checking_pos = -1
                    #return
                #results[0] = checking_pos
                #results[1] = 0
                #return
            
            if (temp_high > self._high):
                if (checking_pos not in danger):
                    if (results[checking_pos] < temp_prob):
                        results[checking_pos] = temp_prob
                return
            
            board[parsed_pos] = 1
            white_predict = 0
            with self._white_graph.as_default():
                white_predict = self._white_model.predict(board.reshape(1, 15, 15, 1))[0]
            top_args_white = numpy.argsort(white_predict)[::-1][:self._width]
            
            for pos in top_args_white:
                if (util.check(-board, parsed_pos)):
                    if (self._color == 'black'):
                        danger.add(checking_pos)
                        continue
                if checking_pos not in danger:
                    check_state_black(numpy.copy(board), pos, checking_pos, temp_prob + numpy.log(white_predict[pos]), 
                                 temp_high, results)

        def check_state_black(board, pos, checking_pos, temp_prob, temp_high, results):
            # print(board, pos)
            # global max_sum_log, best_move
            temp_high += 1
            parsed_pos = numpy.unravel_index(pos, (15, 15))
            #if (util.check(-board, parsed_pos)):
            #    if self._color == 'black':
                    #checking_pos = -1
                    #return
                #results[0] = checking_pos
                #results[1] = 0
                #return
            
            if (temp_high > self._high):
                if (checking_pos not in danger):
                    if (results[checking_pos] < temp_prob):
                        results[checking_pos] = temp_prob
                return
            
            board[parsed_pos] = -1
            black_predict = 0
            with self._black_graph.as_default():
                black_predict = self._black_model.predict(board.reshape(1, 15, 15, 1))[0]
            top_args_black = numpy.argsort(black_predict)[::-1][:self._width]
            
            for pos in top_args_black:
                if (util.check(-board, parsed_pos)):
                    if (self._color == 'white'):
                        danger.add(checking_pos)
                        continue
                if checking_pos not in danger:
                    check_state_white(numpy.copy(board), pos, checking_pos, temp_prob + numpy.log(black_predict[pos]), 
                                 temp_high, results)
        
        top_args = numpy.argsort(predictions)[::-1][:self._width]
        # print(top_args)
        for pos in top_args:
            results[pos] = -100000000000
            if (self._verbose):
                print(numpy.unravel_index(pos, (15, 15)), end = ' - ')
            if (self._color == 'black'):
                check_state_white(numpy.copy(-game.board()), pos, pos, numpy.log(predictions[pos]), 
                                 temp_high, results)
            else:
                check_state_black(numpy.copy(-game.board()), pos, pos, numpy.log(predictions[pos]),
                                 temp_high, results)
        max_sum_log, best_move = -100000000, -1
        for move in results:
            if move not in danger:
                if max_sum_log < results[move]:
                    max_sum_log = results[move]
                    best_move = move
        res = numpy.zeros((225, 1))
        res[best_move] = 1
        code_move = numpy.argmax(res.reshape(1, 225))
        if (self._verbose):
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]))
        return res.reshape(1, 225)

# deprecated agent
class RBSAcombined(Agent):
    def __init__(self, black_model, white_model, name, color, high = 20, 
                 gamma = 0.9, samples = None, timeout = None, verbose = 0):
        self._name = name
        self._black_model = black_model[0]
        self._black_graph = black_model[1]
        self._white_model = white_model[0]
        self._white_graph = white_model[1]
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

    def name(self):
        return self._name

    def policy(self, game):
        start = time()
        if (self._color == 'black' and len(util.list_positions(game.board(), renju.Player.NONE)) == 225):
            res = numpy.zeros((225, 1))
            res[112] = 1
            return res.reshape((1, 225))
        predictions = 0
        temp_high = 0
        
        if self._color == 'black': 
            with self._black_graph.as_default():
                predictions = self._black_model.predict(-game.board().reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]
        else:
             with self._white_graph.as_default():
                predictions = self._white_model.predict(-game.board().reshape(1, 15, 15, 1), batch_size = 1, verbose=0)[0]

        res = numpy.zeros(225)
        n = numpy.zeros(225)
        
        def rollout(board, temp_high, color, checking_pos, max_high = self._high, gamma = self._gamma):
            
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
                    res[checking_pos] += gamma ** (temp_high - 1)
                else:
                    res[checking_pos] -= gamma ** (temp_high - 1)
                return
            
            if (util.check(board, temp_parsed_pos)):
                if (temp_high % 2):
                    res[checking_pos] -= gamma ** (temp_high - 1)
                else:
                    res[checking_pos] += gamma ** (temp_high - 1)
                return
            
            board[temp_parsed_pos] = 1
            next_color = 'white'
            if color == 'white':
                next_color = 'black'
                board[temp_parsed_pos] = -1

            if (temp_high < max_high):
                rollout(board, temp_high + 1, next_color, checking_pos)
            return
        
        i = 0
        while (time() - start < self._timeout and i < self._samples):
            i += 1
            pos = numpy.random.choice(225, p = predictions)
            parsed_pos = numpy.unravel_index(pos, (15, 15))
            
            board = numpy.copy(-game.board())
            n[pos] += 1
            
            if (board[parsed_pos] != 0):
                res[pos] -= 1
                continue
            
            if (util.check(board, parsed_pos)):
                res[pos] = res.max() + 1
                break
            
            board[parsed_pos] = 1
            color = 'white'
            if self._color == 'white':
                color = 'black'
                board[parsed_pos] = -1
            
            rollout(board, 1, color, pos)
        
        values = (res + 10 * predictions) / (1 + n) + (res > 0.5 * n)
        
        if (self._verbose):
            code_move = numpy.argmax(values.reshape(1, 225))
            print(self._name + ':', util.to_move([code_move // 15, code_move % 15]), \
                  'working time:', time() - start, 'iterations:', i)
        
        #values -= values.mean()
        #values /= values.std()
        return values.reshape(1, 225)