# Peer-Review-Bidding
Tanner Fiez, Nihar Shah, Lillian Ratliff. "A SUPER* Algorithm to Optimize Paper Bidding in Peer Review" In ICML Workshop on Real-World Sequential Decision Making, 2019.

The folder SUPER_ICML_RWSD_CODE contains code. The code has been implemented python 2.7. The primary dependencies are numpy, scipy, and the lap package available at https://github.com/gatagat/lap for solving linear assignment problems efficiently. 

The code folder contains the following files:

SUPER_Algorithm.ipynb: SUPER* algorithm as a standalone function to solve for the ordering of papers to present a reviewer given the gain and bidding functions, the similarity scores and heuristic, and the number of bids each paper has. The notebook shows an example usage of the algorithm using the linear programming solution and the sorting method. We demonstrate a timing comparison between the methods as the number of papers. 

SUPER_Algorithm.py: SUPER* algorithm as a standalone function and example usage in a python file. 

algorithms.py: This file contains the problem environment and the algorithm implementations for SUPER* and the baseline methods. It is the primary tool to run the experiments in the paper.  

Simulations.ipynb: Runs the experiments from the paper and provides a wrapper to parallelize the experiments.

utils.py: functions for visualizing results.
