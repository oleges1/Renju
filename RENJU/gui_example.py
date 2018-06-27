from gui import Runner

'''
Available partners:
 * Human
 * Policy - just a policy network
 * BeamSearch - random beam search based on this network
 * MCTS - monte-carlo tree search
'''

# just read line from input and split it on 'Vs'
play = input().split('Vs')

# first arg - black player, second arg - white player
# Attention! While playing you need to click one more time to submit the chosen move.
Runner(play[0], play[1])
