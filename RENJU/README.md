# RENJU PLAYER
## Installation guide:

You need:
* Ubuntu 14 - 16 or anaconda
* TensorFlow 1.4-1.8 and keras 2.1.6
* Python 3
* jupyter notebook

On other backend correct working is not guaranteed.

You need to copy all files from this folder, except cnn_learning.ipynb and runner_models.ipynb.

Also you need to load my models from google drive: [papa_black], [papa_white].

Then run example.ipynb in jupyter and have fun!

## File guide:

* agent.py - collection of diffrents agents, ex. Policy and Beamsearch agents
* mcts.py - implementation of monte-carlo tree search agent
* renju.py - class game and basic gui
* runner_models.py - example of using this models
* util.py - all needed functions
* cnn_learning.ipynb - preparing networks in google colaboratory
* example.ipynb - simple example
* competition.py - agent for competition of agents in gomoku, based on mcts agent, and named 'kopatych' in honor of the popular russian cartoon hero.

[papa_black]:https://drive.google.com/open?id=1GQ2Bs3z84mpJbshxeinWZE84XlrNujmM
[papa_white]:https://drive.google.com/open?id=1BsEL7tphJtLTwHEEGXUzxh61x120cOq4
