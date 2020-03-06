from __future__ import division
from __future__ import print_function
import numpy as np
import itertools
from scipy.stats import rankdata
# https://github.com/gatagat/lap
import lap 


class Problem(object):
    """Class providing environment, algorithms, and simulation protocol for the bidding problem."""

    def __init__(self, s, g_p, g_r, f, f_tilde, noise, hyper, special, stop, poisson, subset):
        """Initialize the environment and problem class.
        
        :param s (list): List of similarity matrix to simulate problem and algorithms on. 
    
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

        :param f (function): Model bidding function mapping paper position to a score. The score function should be 
        non-increasing in the paper position. The function should handle the paper position input as an array 
        containing the paper positions for each paper ordered by the paper index or the paper position for a fixed paper.

        Ex/ f = lambda pi: 1/np.log2(pi + 1)
        
        :param f_tilde (function): This paramter provides the capability to provide a bidding function that generates
        the bids that may not agree with the model. If the model is legitimate, we can take f = f_tilde. If not, 
        the actual bid generating f_tilde can be provided in this argument following an equivalent format as f. 

        :param noise (float): Variance for normal noise to include in the similarity scores. 
        In general, the default should be zero noise.

        :param hyper (float): Parameter dictating the weight given to the reviewer-side gain function. 

        :param special (bool): If the reviewer-side gain function is multiplicatively separable into the form g_r(pi, s) = g_r_similarity(s)f(pi) 
        where g_r_similarity is a non-decreasing function of the similarity score and f is the non-increasing bidding function
        of the position a paper is shown, then a simple sorting routine can be used instead of the linear program. To run the sorting
        procedure, the argument special should be passed in as True. In general, the default should be special=False.

        Ex/ If g_r(s, pi) = (2**s - 1)/np.log2(pi + 1), then g_r_similarity(s) = (2**s - 1) and f(pi) = np.log2(pi + 1), so 
        the argument special=True can be passed in so the efficent sorting procedure is deployed.
        
        :param stop (function): Function taking as input the number of reviewers, and it should return when the reviewers should 
        stop arriving. The default should be stop = lambda x: x, so that each reviewer arrives. This argument is intended to allow
        the ability to stop early to simulate if not all reviewers arrive.
        
        :param poisson (bool): This parameter allows for poisson arrivals where more than 1 reviewer arrives at once. In general,
        the default should be False so that reviewers arrive in a sequential order. If True, poisson(1) reviewers will arrive
        simultaneously and need to be presented papers simultaneously.
        
        :param subset (bool): This parameter allows for the option that a reviewer will only bid on a subset of the papers with 
        non-zero probability. In general, the default should be False. If True, the reviewer only considers sqrt(num_papers) papers
        randomly selected and the ordering the algorithm selects is adjusted to rank the papers among this subset.       
        """
        

        # List of similarity matrix.
        self.s_list = s

        # Number of reviewers. 
        self.T = s[0].shape[0]  

        # Number of papers.
        self.n = s[0].shape[1]
        
        # Gain function for paper-side gain.                     
        self.g_p = g_p

        # Gain function for reviewer-side gain.                     
        self.g_r = g_r

        # Model bidding function f.
        self.f = f

        # Real bidding function f.
        self.f_tilde = f_tilde

        # Noise variance from normal distribution in the similarity scores.
        self.noise = noise

        # Hyperparamter for reviewer-side gain function.
        self.hyper = hyper

        # Sorting method or linear program.
        self.special = special
        
        # When to stop reviewers from arriving.
        self.end = stop(self.T)
        
        # Indicator for poisson arrivals.
        self.poisson = poisson
        
        # Indicator if the reviewer will only bid on a subset of papers.
        self.subset = subset


    def simulate(self, algorithm, seed=0):

        # Tracking the gain each time the algorithm is ran. 
        self.p_gain_history = []
        self.r_gain_history = []
        self.gain_history = []
        
        # Compute ahead to speed up code.
        self.paper_index = np.arange(1, self.n+1)
        self.mean_value = self.f(self.paper_index).mean()
        self.zeros = np.zeros(self.n)
        self.ones = np.ones(self.n)

        np.random.seed(seed)
        count = len(self.s_list)

        bids = []

        # Run the algorithm on each similarity matrix.
        for self.s in self.s_list:
            
            # Compute sum of similarity scores for each paper ahead to speed up code.
            self.s_sum = self.s.sum(axis=0)
            
            # Compute cumulative sum of similarity scores for each paper from reviewer t on ahead to speed up code.
            self.s_cumsum = self.s[1:][::-1].cumsum(axis=0)[::-1]
                
            # Initialize reviewer-side gain.
            r_gain = 0.
    
            # Tracking number of bids.
            self.d = np.zeros(self.n)
            
            # Reviewer index.
            self.t = 0

            # Run algorithm.
            while self.t < self.end:
                
                if not self.poisson:

                    # Select ordering to present to reviewers.
                    pi_t = algorithm()
                                 
                    # If reviewer will only bid on subset of papers selected randomly.
                    if self.subset:
                        subset = np.random.choice(self.n, size=int(np.sqrt(self.n)), replace=False)
                        pi_t[subset] = rankdata(pi_t[subset], 'ordinal')
                        non_subset = np.array(list(set(np.arange(self.n))-set(subset)))

                        # Simulate the bidding process and update the number of bids on each paper.
                        noisy_similarity = np.clip(self.s[self.t]+np.random.normal(0, self.noise, self.n), 0, 1)
                        noisy_similarity[non_subset] = 0

                    else:
                        # Simulate the bidding process and update the number of bids on each paper.
                        noisy_similarity = np.clip(self.s[self.t]+np.random.normal(0, self.noise, self.n), 0, 1)
                        
                    # Compute bid probability and obtain realizations to update bids.
                    bid_probs = noisy_similarity*self.f_tilde(pi_t)
                    self.d += np.random.binomial(1, p=bid_probs)
                    
                    if self.subset:
                        # Update the cumulative reviewer-side gain.
                        r_gain += self.g_r(self.s[self.t][subset], pi_t[subset]).sum()

                    else:
                        # Update the cumulative reviewer-side gain.
                        r_gain += self.g_r(self.s[self.t], pi_t).sum()

                    self.t += 1 
                    
                else:
                    
                    num_arrivals = np.random.poisson(1)
                    arrival_count = 0
                    interval_bids = np.zeros(self.n)
                    interval_r_gain = 0
                    
                    while arrival_count < num_arrivals and self.t < self.end:
                    
                        # Select ordering to present to reviewers.
                        pi_t = algorithm()

                        # Simulate the bidding process and update the number of bids on each paper.
                        noisy_similarity = np.clip(self.s[self.t]+np.random.normal(0, self.noise, self.n), 0, 1)
                        bid_probs = noisy_similarity*self.f_tilde(pi_t)
                        interval_bids += np.random.binomial(1, p=bid_probs)

                        # Update the cumulative reviewer-side gain.
                        interval_r_gain += self.g_r(self.s[self.t], pi_t).sum()
                        
                        arrival_count += 1
                        self.t += 1
                        
                    self.d += interval_bids
                    r_gain += interval_r_gain

            # Compute the cumulative gain for the simulation.
            self.p_gain_history.append(self.g_p(self.d).sum())
            self.r_gain_history.append(self.hyper*r_gain)
            self.gain_history.append(self.g_p(self.d).sum() + self.hyper*r_gain)

            # Track the bids.
            bids.append(self.d)
        
        # Tracking data.
        self.bid_mean = np.array(self.d).mean(axis=0)
        self.bid_se = np.array(self.d).std(axis=0)/np.sqrt(count)

        self.p_gain_history = np.array(self.p_gain_history)
        self.p_gain_mean = self.p_gain_history.mean()
        self.p_gain_se = self.p_gain_history.std()/np.sqrt(count)

        self.r_gain_history = np.array(self.r_gain_history)
        self.r_gain_mean = self.r_gain_history.mean()
        self.r_gain_se = self.r_gain_history.std()/np.sqrt(count)

        self.gain_history = np.array(self.gain_history)
        self.gain_mean = self.gain_history.mean()
        self.gain_se = self.gain_history.std()/np.sqrt(count)


    def super_zero_heuristic_policy(self):
        """Algorithm computing ordering from SUPER with zero heuristic
    
        return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example, 
        pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1. 
        """

        if not self.special:
            # Solve linear assignment problem to get ordering to present.
            w_p = lambda j,k: self.s[self.t, j]*self.f(k)*(self.g_p(self.d[j]+1) - self.g_p(self.d[j])) 
            w_r = lambda j,k: self.hyper*self.g_r(self.s[self.t, j], k)
            w = np.array([w_p(j, self.paper_index) + w_r(j, self.paper_index) for j in xrange(self.n)])
            pi_t = lap.lapjv(-w)[1]
            pi_t += 1

        if self.special:
            # Rank papers from maximum to minimum for alpha breaking ties by the similarity score and show in that order.
            alpha = self.s[self.t]*(self.g_p(self.d + 1) - self.g_p(self.d)) + self.hyper*self.g_r(self.s[self.t], self.ones)
            alpha_pairs = np.array(zip(self.s[self.t], alpha, self.paper_index), dtype=[('sim', float), ('alpha', float), ('index', float)])        
            pi_t = np.argsort(np.lexsort((-alpha_pairs['sim'], -alpha_pairs['alpha'])))+1 

        return pi_t


    def super_mean_heuristic_policy(self):
        """Algorithm computing ordering from SUPER with mean heuristic
        
        return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example, 
        pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1. 
        """
        
        # Compute heuristic.
        if self.t == self.T - 1:
            proxy = self.zeros
        else:
            # Compute the estimate of future bids for each paper using mean heuristic.
            proxy = self.mean_value * self.s_cumsum[self.t]

        if not self.special:
            # Solve linear assignment problem to get ordering to present.
            w_p = lambda j,k: self.s[self.t, j]*self.f(k)*(self.g_p(self.d[j]+proxy[j]+1) - self.g_p(self.d[j]+proxy[j])) 
            w_r = lambda j,k: self.hyper*self.g_r(self.s[self.t, j], k)
            w = np.array([w_p(j,np.arange(1, self.n+1)) + w_r(j,np.arange(1, self.n+1)) for j in xrange(self.n)])
            pi_t = lap.lapjv(-w)[1]
            pi_t += 1

        if self.special:
            # Rank papers from maximum to minimum for alpha breaking ties by the similarity score.
            alpha = self.s[self.t]*(self.g_p(self.d + proxy + 1) - self.g_p(self.d + proxy)) + self.hyper*self.g_r(self.s[self.t], self.ones)
            alpha_pairs = np.array(zip(self.s[self.t], alpha, self.paper_index), dtype=[('sim', float), ('alpha', float), ('index', float)])        
            pi_t = np.argsort(np.lexsort((-alpha_pairs['sim'], -alpha_pairs['alpha'])))+1  
            
        return pi_t


    def bid_policy(self):
        """BID policy ordering papers in decreasing order of the number of bids.
        
        return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example, 
        pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1. 
        """
        
        # Rank papers from minimum to maximum number of bids received breaking ties in favor of paper with higher similarity score.        
        bid_pairs = np.array(zip(self.s[self.t], self.d, self.paper_index), dtype=[('sim', float), ('bid', float), ('index', float)])        
        pi_t = np.argsort(np.lexsort((-bid_pairs['sim'], bid_pairs['bid'])))+1
                
        return pi_t


    def sim_policy(self):
        """SIM policy ordering papers in decreasing order of the similarity scores.
        
        return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example, 
        pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1. 
        """

        # Rank papers from maximum to minimum similarity score breaking ties in favor of paper with fewer bids.        
        similarity_pairs = np.array(zip(self.s[self.t], self.d, self.paper_index), dtype=[('sim', float), ('bid', float), ('index', float)])        
        pi_t = np.argsort(np.lexsort((similarity_pairs['bid'], -similarity_pairs['sim'])))+1
        
        return pi_t


    def random_policy(self):
        """RAND policy ordering papers in a random order.
        
        return pi_t (array): Array containing the position each paper is to be presented ordered by paper index. For example, 
        pi_t = [2, 1] means paper 1 is presented in position 2, and paper 2 is presented in position 1. 
        """

        # Select a random paper ordering to present to reviewers.
        pi_t = np.random.permutation(range(1, self.n+1))

        return pi_t

