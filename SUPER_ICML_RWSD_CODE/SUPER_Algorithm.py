import numpy as np
import lap
import time
import matplotlib.pyplot as plt

def super_algorithm(g_p, g_r, f, s, bids, h, trade_param, g_r_similarity=None):
    
    """Solve for a paper ordering for a reviewer using SUPER* procedure. 
    
    This procedure requires numpy and the lap package https://github.com/gatagat/lap 
    to solve the linear assignment problem. 
    
    :param g_p (function): Paper-side gain function mapping bid counts to a score. 
    The score function should be non-decreasing in the number of bids. 
    The function should handle the bid count input as an array containing 
    the number of bids for each paper ordered by the paper index or the 
    bid count input as a number for a fixed paper. 
    
    Ex/ g_p = lambda bids: np.sqrt(bids)
    
    :param g_r (function): Reviewer-side gain function mapping similarity score and paper position to a score.
    The score function should be non-decreasing in the similarity score and and non-increasing in the paper position.
    The function should handle the similarity score input and the paper position input as arrays containing the
    similarity scores and paper positions for each paper ordered by the paper index or the similarity score and 
    paper position for a fixed paper.
    
    Ex/ g_r = lambda s, pi: (2**s - 1)/np.log2(pi + 1)
    
    :param f (function): Bidding function mapping paper position to a score. The score function should be 
    non-increasing in the paper position. The function should handle the paper position input as an array 
    containing the paper positions for each paper ordered by the paper index or the paper position for a fixed paper.
    
    Ex/ f = lambda pi: 1/np.log2(pi + 1)
    
    :param s (array): Similarity scores for each paper ordered by paper index.
    
    :param bids (array): Number of bids for each paper ordered by the paper index prior to the arrival of the reviewer.
    
    :param h (array): Heuristic values estimating the number of bids for each paper in the future ordered by the paper index. 
    
    :param trade_param (float): Parameter dictating the weight given to the reviewer-side gain function. 
    
    :param g_r_similarity (function): Optional function that allows for computationally effiicent solution. 
    If the reviewer-side gain function is multiplicatively separable into the form g_r(pi, s) = g_r_similarity(s)f(pi) 
    where g_r_similarity is a non-decreasing function of the similarity score and f is the non-increasing bidding function
    of the position a paper is shown, then a simple sorting routine can be used instead of the linear program. To run the sorting
    procedure, the function g_r_similarity should be passed in and it should handle the similarity score input as an array 
    containing the similarity scores for each paper ordered by paper index or the similarity score for a given paper. 
    
    Ex/ If g_r(s, pi) = (2**s - 1)/np.log2(pi + 1), then g_r_similarity(s) = (2**s - 1) and f(pi) = np.log2(pi + 1), so 
    the function g_r_similarity = lambda s: 2**s - 1 can be passed in so the efficent sorting procedure is deployed.
    
    return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example, 
    pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1. 
    """

    d = len(s)

    if g_r_similarity is None:
        # Solve linear assignment problem to get ordering to present.
        w_p = lambda j,k: s[j]*f(k)*(g_p(bids[j] + h[j] + 1) - g_p(bids[j]+h[j])) 
        w_r = lambda j,k: trade_param*g_r(s[j], k)
        w = np.array([w_p(j, np.arange(1, d+1)) + w_r(j, np.arange(1, d+1)) for j in xrange(d)])
        pi_t = lap.lapjv(-w)[1]
        pi_t += 1
    else:
        # Rank papers from maximum to minimum for alpha breaking ties by the similarity score.
        alpha = s*(g_p(bids + h + 1) - g_p(bids + h)) + trade_param*g_r_similarity(s)
        alpha_pairs = np.array(zip(s, alpha, np.arange(1, d+1)), dtype=[('sim', float), ('alpha', float), ('index', float)])        
        pi_t = np.argsort(np.lexsort((-alpha_pairs['sim'], -alpha_pairs['alpha'])))+1  

    return pi_t


"""EXAMPLE USAGE

# Configuration
num_papers = 100
g_p = lambda x: np.sqrt(x)
g_r = lambda s, pi: (2**(s)-1)/np.log2(pi + 1)
f = lambda pi: 1./np.log2(pi + 1)
s = np.random.rand(num_papers)
bids = np.random.randint(0, 10, num_papers) 
h = np.random.rand(num_papers)
trade_param = 1
g_r_similarity = lambda s: (2**(s)-1)


# Linear Program Method to Find Paper Ordering
pi_lp = super_algorithm(g_p, g_r, f, s, bids, h, trade_param, g_r_similarity=None)

# Sorting Method to Find Paper Ordering
pi_sort = super_algorithm(g_p, g_r, f, s, bids, h, trade_param, g_r_similarity)
"""

