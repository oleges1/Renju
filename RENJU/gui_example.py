#!/usr/bin/python3
from gui import Runner

'''
Available partners:
 * Human
 * Policy - just a policy network
 * BeamSearch - random beam search based on this network
 * MCTS - monte-carlo tree search
'''

print('Available partners:\n\
 * Human\n\
 * Policy - just a policy network\n\
 * BeamSearch - random beam search based on this network\n\
 * MCTS - monte-carlo tree search')

print(
'To run game just type\n\
    <agent>Vs<agent>\n\
For example:\n\
    HumanVsHuman\n\
Or\n\
    MCTSVsHuman')

print(
'Attention! \n\
While playing you need to click one more time to submit the chosen move.')

# just read line from input and split it on 'Vs'
play = input().split('Vs')

# first arg - black player, second arg - white player
# Attention! While playing you need to click one more time to submit the chosen move.
Runner(play[0], play[1])
