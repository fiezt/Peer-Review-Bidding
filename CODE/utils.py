from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from seaborn import xkcd_rgb as xkcd
import seaborn as sns
sns.set()


sns.set(rc={'text.usetex' : True}, font_scale=1.0)
sns.set_style("whitegrid", 
             {'font.family':['serif'], 'font.serif':['Times New Roman'], 
              'grid.color':'.9'})

lw = 2
fs1 = 22
fs2 = 24
fs3 = 26

ms = 0
ms_big = 20
msew = 2
cs = 8


def sweep_comparison_plot(algs, axis, xlabel=None, fi=None, show_fig=True, save_fig=False):


    fig = plt.figure(1, figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)
    fig.canvas.draw()

    xkcd_colors = [xkcd['black'], xkcd['tomato red'], xkcd['yellow orange'], xkcd['teal'], xkcd['magenta']]
    
    count = range(len(algs))
    num_items = range(len(axis))

    base = np.array([algs[i][-1][0] for i in xrange(len(algs))])
    algs = [[algs[i][j] for i in xrange(len(algs))] for j in xrange(5)]
    labels = [r'$\texttt{SUPER}^{\ast}$', r'$\texttt{SUPER}^{0}$', r'$\texttt{SIM}$', r'$\texttt{BID}$', r'$\texttt{RAND}$']

    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))] 
    markers = ['o', 's', '^', 'X', 'D']

    for i, alg in enumerate(algs):
        x = axis
        mean = np.array([item[0] for item in alg])-base
        std = np.array([item[1] for item in alg])

        ax.errorbar(num_items, mean, std, fmt='ok', lw=lw, capsize=8, markersize=0, 
                    markeredgewidth=2, ecolor=xkcd_colors[i])
        ax.plot(num_items, mean, linestyle=linestyles[i], marker=markers[i], color=xkcd_colors[i], lw=lw, ms=10, label=labels[i])

    ax.set_xlabel(xlabel, fontsize=fs2)
    ax.set_ylabel(r'Relative gain', fontsize=fs3, rotation='vertical')

    ax.set_xticks(num_items)
    ax.set_xticklabels(axis)
    ax.tick_params(labelsize=fs2)
    
    if save_fig:
        plt.savefig(fi, bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()
        
        
def sweep_comparison_plot_sim_base(algs, axis, xlabel=None, fi=None, show_fig=True, save_fig=False):


    fig = plt.figure(1, figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)
    fig.canvas.draw()

    xkcd_colors = [xkcd['black'], xkcd['tomato red'], xkcd['yellow orange'], xkcd['teal'], xkcd['magenta']]
    
    count = range(len(algs))
    num_items = range(len(axis))

    base = np.array([algs[i][2][0] for i in xrange(len(algs))])
    algs = [[algs[i][j] for i in xrange(len(algs))] for j in xrange(3)]
    labels = [r'$\texttt{SUPER}^{\ast}$', r'$\texttt{SUPER}^{0}$', r'$\texttt{SIM}$']

    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))] 
    markers = ['o', 's', '^', 'X', 'D']

    for i, alg in enumerate(algs):
        x = axis
        mean = np.array([item[0] for item in alg])-base
        std = np.array([item[1] for item in alg])

        ax.errorbar(num_items, mean, std, fmt='ok', lw=lw, capsize=8, markersize=0, 
                    markeredgewidth=2, ecolor=xkcd_colors[i])
        ax.plot(num_items, mean, linestyle=linestyles[i], marker=markers[i], color=xkcd_colors[i], lw=lw, ms=10, label=labels[i])

    ax.set_xlabel(xlabel, fontsize=fs2)
    ax.set_ylabel(r'Relative gain', fontsize=fs3, rotation='vertical')

    ax.set_xticks(num_items)
    ax.set_xticklabels(axis)
    ax.tick_params(labelsize=fs2)
    
    if save_fig:
        plt.savefig(fi, bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()
        
        
        
def plot_bid_count_data(bid_count_data, hyper_index, intervals, alg_set, endpoint, index=None, fi=None, show_fig=True, save_fig=False):
    

    fig = plt.figure(1, figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)
    fig.canvas.draw()

    xkcd_colors = [xkcd['black'], xkcd['tomato red'], xkcd['yellow orange'], xkcd['teal'], xkcd['magenta']]

    if endpoint:
        xlabels = ['\{'+','.join([str(num) for num in range(item[0], item[1]+1)])+'\}' for item in intervals]
    else:
        xlabels = ['\{'+','.join([str(num) for num in range(item[0], item[1]+1)])+'\}' for item in intervals]
        xlabels[-1] ='\{'+str(intervals[-1][0])+'$+$'+'\}'

    if index is None:
        bid_distributions = np.vstack([get_counts(bid_count_data[hyper_index][i][-1], intervals, endpoint)[1] for i in alg_set]).T 
    else:
        bid_distributions = np.vstack([get_counts(bid_count_data[hyper_index][i][-1], intervals, endpoint, index)[1] for i in alg_set]).T 
    print(np.round(bid_distributions))
    pd.DataFrame(bid_distributions).plot(kind='bar', ax=ax, rot=0, color=xkcd_colors[:len(alg_set)], legend=False)
    
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(r'Number of bids', fontsize=fs2)
    ax.set_ylabel(r'Number of papers', fontsize=fs3, rotation='vertical')
    ax.tick_params(labelsize=fs2)
    
    
    if save_fig:
        plt.savefig(fi, bbox_inches='tight', dpi=300)

    if show_fig:
        plt.show()
        
def get_counts(bids, intervals, endpoint=True, index=None):

    arr = []
    arr_total = []
        
    for run_ in bids:
        if index is None:
            run = run_
        else:
            run = run_[index:]
        temp = []
        for inter in intervals:
            if inter == intervals[-1] and not endpoint:
                temp.append(len(run[(run>=inter[0])]))            
            else:
                temp.append(len(run[(run>=inter[0]) & (run<=inter[1])]))
        arr.append(temp)
        arr_total.append(np.sum(run))
        
    return np.mean(arr_total), np.mean(arr, axis=0)



    fig = plt.figure(1, figsize=(6.5,6))
    fig.clf()
    ax = fig.add_subplot(111)
    fig.canvas.draw()

    xkcd_colors = [xkcd['black'], xkcd['tomato red'], xkcd['yellow orange'], xkcd['teal'], xkcd['magenta']]