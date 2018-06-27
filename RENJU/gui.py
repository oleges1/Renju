#-*- coding: utf-8 -*-

import tkinter
import math
import time

from renju import run, run_test, Player, Game
from agent import BackendAgent, HumanAgent, Agent, CnnAgent, BeamSearchAgent, ComplexCnnAgent, RandomBeamSearchAgent, RBSAcombined
from keras.models import load_model
import tensorflow as tf
from mcts import MCTS
import numpy as np
import util


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.pixel_x = 45 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y


class BoardCanvas(tkinter.Canvas):
    def __init__(self, master=None, height=0, width=0, board_size=15):
        self.board_size = board_size    
        tkinter.Canvas.__init__(self, master, height=height, width=width)
        self.init_board_points()    
        self.init_board_canvas()

    def init_board_points(self):
        self.board_points = [[None for i in range(self.board_size + 1)] for j in range(self.board_size + 1)]
        for i in range(self.board_size + 1):
            for j in range(self.board_size + 1):
                self.board_points[i][j] = Point(i, j)

    def init_board_canvas(self):
        column_letter = 'abcdefghjklmnop'

        for i in range(self.board_size + 1):
            p1 = self.board_points[i][0]
            p2 = self.board_points[i][self.board_size]
            self.create_line(p1.pixel_x - 15, p1.pixel_y - 15, p2.pixel_x - 15, p2.pixel_y - 15)
            if (i != self.board_size):
                self.create_text(p2.pixel_x, p2.pixel_y, text=column_letter[i], tag=column_letter[i])

        for j in range(self.board_size + 1):  
            p1 = self.board_points[0][j]
            p2 = self.board_points[self.board_size][j]
            self.create_line(p1.pixel_x - 15, p1.pixel_y - 15, p2.pixel_x - 15, p2.pixel_y - 15)
            if (j != self.board_size):
                self.create_text(p1.pixel_x - 30, p1.pixel_y, text=str(self.board_size - j), tag=str(self.board_size - j))

        for i in range(self.board_size):  
            for j in range(self.board_size):
                r = 1
                p = self.board_points[i][j]
                self.create_oval(p.pixel_x-r, p.pixel_y-r, p.pixel_x+r, p.pixel_y+r)

    def place_move(self, move, color):
        """Draw circle on the board"""
        p = self.board_points[move[1]][14 - move[0]]
        self.create_oval(p.pixel_x-10, p.pixel_y-10, p.pixel_x+10, p.pixel_y+10, fill=color)

    def find_move_coordinates(self, event):
        """Find line crossing closest to the click position"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                p = self.board_points[i][j]
                square_distance = math.pow((event.x - p.pixel_x), 2) + math.pow((event.y - p.pixel_y), 2)
                if (square_distance <= 300): 
                    return (14 - j, i)

    def print_message(self, text):
        """Display text message below board"""
        self.delete("text_tag")
        self.create_text(240, 550, text=text, tag="text_tag")


class BoardFrame(tkinter.Frame):
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.board_label_frame = tkinter.LabelFrame(self, text="Gomoku", padx=5, pady=5)
        self.board_label_frame.pack()
        

class Runner(object):
    """class for comfortable gaming one against another"""
    def __init__(self, black, white):
        self.game = Game()
        self.papa_black = load_model('papa_black4096.h5')
        self.papa_black_graph = tf.get_default_graph()
        self.papa_white = load_model('papa_white4096.h5')
        self.papa_white_graph = tf.get_default_graph()

        # parse args and create models
        if (black == 'MCTS'):
            self.black = MCTS(name = 'MCTSSLOW', 
                   black_model = (self.papa_black, self.papa_black_graph), 
                   white_model = (self.papa_white, self.papa_white_graph),
                   black_rollout = (self.papa_black, self.papa_black_graph), 
                   white_rollout = (self.papa_white, self.papa_white_graph), 
                   timeout = 14.75, high = 14, gamma = 0.99, verbose = 0, min_prob = 0.8, param1 = 0.2, param2 = 0.65)
        elif (black == 'Policy'):
            self.black = CnnAgent(color = 'black', name = 'black_cnn', model = (self.papa_black, self.papa_black_graph), verbose = 0)
        elif (black == 'BeamSearch'):
            self.black = RandomBeamSearchAgent(color = 'black', name = 'TreeBlack', 
                   black_model = (self.papa_black, self.papa_black_graph), 
                   white_model = (self.papa_white, self.papa_white_graph), 
                   timeout = 14.75, high = 16, gamma = 0.99, fine = 1.0, bonus = 1.0, verbose = 0)
        else:
           self.black = 'Human'

        if (white == 'MCTS'):
            self.white = MCTS(name = 'MCTSSLOW', 
                   black_model = (self.papa_black, self.papa_black_graph), 
                   white_model = (self.papa_white, self.papa_white_graph),
                   black_rollout = (self.papa_black, self.papa_black_graph), 
                   white_rollout = (self.papa_white, self.papa_white_graph), 
                   timeout = 14.75, high = 14, gamma = 0.99, verbose = 0, min_prob = 0.8, param1 = 0.2, param2 = 0.65)
        elif (white == 'Policy'):
            self.white = CnnAgent(color = 'white', name = 'white_cnn', model = (self.papa_white, self.papa_white_graph), verbose = 0)
        elif (white == 'BeamSearch'):
            self.white = RandomBeamSearchAgent(color = 'white', name = 'TreeWlack', 
                   black_model = (self.papa_black, self.papa_black_graph), 
                   white_model = (self.papa_white, self.papa_white_graph), 
                   timeout = 14.75, high = 16, gamma = 0.99, fine = 1.0, bonus = 1.0, verbose = 0)
        else:
            self.white = 'Human'

        # init gui
        window = tkinter.Tk()
        self.board_frame = BoardFrame(window)
        self.board_canvas = BoardCanvas(self.board_frame.board_label_frame, height=600, width=500)

        # bind left mouse button click event
        self.board_canvas.bind('<Button-1>', self.click_event)  

        self.board_frame.pack()
        self.board_canvas.pack()

        window.mainloop()

    def finish_game(self, message):
        self.board_canvas.print_message(message)
        self.board_canvas.unbind('<Button-1>')

    def process_move(self, move):
        # check and record move, if not finished

        if not self.game.is_posible_move(move):
            self.board_canvas.print_message("Move is invalid!")
            return False

        self.board_canvas.place_move(move, self.game._player.__repr__())
        self.board_canvas.print_message(self.game._player.__repr__() + " move is: " + util.to_move(move))

        if not self.game.move(move): 
            self.finish_game("Player " + self.game._player.__repr__() + " wins")
            return False

        return True
    
    def click_event(self, event):
        move = self.board_canvas.find_move_coordinates(event)
        if (self.game._player == Player.BLACK):
            if (self.black == 'Human'):
                if not self.process_move(move):
                    return
            else:
                move = np.argmax(self.black.policy(self.game))
                parsed_move = np.unravel_index(move, (15, 15))
                if not self.process_move(parsed_move):
                    return
        else:
            if (self.white == 'Human'):
                if not self.process_move(move):
                    return
            else:
                move = np.argmax(self.white.policy(self.game))
                parsed_move = np.unravel_index(move, (15, 15))
                if not self.process_move(parsed_move):
                    return