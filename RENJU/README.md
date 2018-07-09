# RENJU PLAYER
## Installation guide:

You need:
* Ubuntu 14 - 16 or anaconda
* TensorFlow 1.4-1.8 and keras 2.1.6
* Python 3
* Tkinter 4 or jupyter notebook (as you wish)

On other backend correct working is not guaranteed.

You need to copy all files from this repository, maybe except cnn_learning.ipynb and runner_models.ipynb.

To do it just use command:
`git clone https://github.com/oleges1/Renju/RENJU`

Also you need to load my models from google drive: [papa_black], [papa_white].

### Playing in Jupyter notebook:
Run `example.ipynb` in jupyter and have fun!

### Playing in separate window by clicking:
Run `gui_example.py` by `python3 gui_example.py` and just follow the instructions, also you can just type `./gui_example.py` in terminal.
If you have some problems with loading python3, check command `which python3` and, if necessary, change the path at the very begging of `gui_example.py`.

This method offers more comfortable way of playing Gomoku.

## File guide:

* `agent.py` - collection of diffrents agents, ex. Policy and Beamsearch agents
* `mcts.py` - implementation of monte-carlo tree search agent
* `renju.py` - class game and basic gui
* `runner_models.py` - example of using this models
* `util.py` - all needed functions
* `cnn_learning.ipynb` - preparing networks in google colaboratory
* `example.ipynb` - simple example of playing in jupyter notebook
* `gui.py` - gui for playing, based on tkinter
* `gui_example.py` - simple example of gui usage 
* `competition.py` - agent for competition of agents in gomoku, based on mcts, named `Kopatych` in honor of the popular russian cartoon hero and all in all `Kopatych` took the first place in this competition.

[papa_black]:https://drive.google.com/open?id=1GQ2Bs3z84mpJbshxeinWZE84XlrNujmM
[papa_white]:https://drive.google.com/open?id=18edjuILw_t84A8NcAHZCpDnn2TOkTfYo
