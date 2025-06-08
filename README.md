# Reinforcement Learning (Pathfinding and Exploration) Inspired from Slime Mold
This repository contains code and some example outputs (image and video) from the programs.
"QL_ex" is the exploration algorithm which plots a graphic of learned Q-values for reaching a certain state (averaged).

"Pathfinder_single" is the single-agent-single-target pathfinding algorithm.
"Conquer_test" is the extended version of the pathfinder algorithm in that it is the single-agent-multiple-target pathfinding algorithm.

All programs are written in Python and utilize Pytorch and Gymnasium in conjunction with some other standard libraries.
Please note that as the programs randomly generate most of the targets, the example results may be different from what the code generates.

### Exploration Algorithm
The exploration algorithm is a standalone file which will generate a target at the southeast corner and other targets randomly throughout the grid. The start node is located at the center of the top of the grid.
The example output for this program is included as "Figure_1." The figure with lines is an edited version of this output figure to illustrate possible connection pathways between the target nodes.
The nominal values of the Q-values appear to be incorrect, but the relative values seem to be correct.

The code for this algorithm is a modification of the tutorial/sample code from here: https://www.geeksforgeeks.org/q-learning-in-python/

### Pathfinding Algorithms
The pathfinding algorithms use the Gymnasium environments provided in the "gym_game" folder. These environments are modified versions of the custom "GridWorld" environment.
For more information about custom environments, please visit the documentation for Gymnasium: https://gymnasium.farama.org/introduction/create_custom_env/

The programs each have a `main()` and `test()` function to train and test the artificial intelligence models, respectively.
A pretrained model is already included in this repository, so the command to call the `main()` function can be commented out.
The output of the programs should generate a video in the current working directory. Output video samples for the programs have been included in the repository for reference.
**Please be aware that the execution of these programs WILL overwrite the previously generated videos and those videos cannot be recovered!**
The current example videos were renamed so they will not be deleted upon execution of the code.

The code for the Deep RL agent is primarily sourced from the tutorial here: https://www.datacamp.com/tutorial/reinforcement-learning-with-gymnasium
Only minor changes were made to the reward calculations and observation tensors used for training. The `main()` function was modified to save the trained model to "Model.pt" in the working directory so testing of the algorithm does not require retraining.
Testing can be done with just the `test()` function. Video recording may require the MoviePy library as a prerequisite.
