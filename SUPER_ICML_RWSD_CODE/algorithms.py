from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.stats import rankdata
import itertools
import lap
import copy
import time


class Problem(object):

    def __init__(self, s, g_p, g_r, f, f_tilde, noise, hyper, special, stop, poisson, subset):

        # Similarity matrix.
        self.s_list = s

        # Number of reviewers. 
        self.T = s[0].shape[0]  

        # Number of papers.
        self.n = s[0].shape[1]
        
        # Gain function for paper-side loss.                     
        self.g_p = g_p

        # Gain function for reviewer-side loss.                     
        self.g_r = g_r

        # Model bidding function f.
        self.f = f

        # Real bidding function f.
        self.f_tilde = f_tilde

        # Noise in the similarity scores.
        self.noise = noise

        # Hyperparamter for loss function.
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

        # Tracking the loss each time the algorithm is ran. 
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

        # Run the algorithm many times on the similarity matrix.
        for self.s in self.s_list:
            
            # Compute ahead to speed up code.
            self.s_sum = self.s.sum(axis=0)
            self.s_cumsum = self.s[1:][::-1].cumsum(axis=0)[::-1]
                
            # Initialize reviewer-side gain.
            r_gain = 0.
    
            # Tracking number of bids.
            self.d = np.zeros(self.n)
            
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


    def super_random_heuristic_policy(self, special=False):
  
        # Compute heuristic.
        if self.t == self.T - 1:
            proxy = self.zeros
        else:
            # Compute the estimate of future bids for each paper.
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
        
        # Rank papers from minimum to maximum number of bids received breaking ties in favor of paper with higher similarity score.        
        bid_pairs = np.array(zip(self.s[self.t], self.d, self.paper_index), dtype=[('sim', float), ('bid', float), ('index', float)])        
        pi_t = np.argsort(np.lexsort((-bid_pairs['sim'], bid_pairs['bid'])))+1
                
        return pi_t


    def sim_policy(self):

        # Rank papers from maximum to minimum similarity score breaking ties in favor of paper with fewer bids.        
        similarity_pairs = np.array(zip(self.s[self.t], self.d, self.paper_index), dtype=[('sim', float), ('bid', float), ('index', float)])        
        pi_t = np.argsort(np.lexsort((similarity_pairs['bid'], -similarity_pairs['sim'])))+1
        
        return pi_t


    def random_policy(self):

        # Select a random paper ordering to present to reviewers.
        pi_t = np.random.permutation(range(1, self.n+1))

        return pi_t

