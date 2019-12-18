from __future__ import division
import numpy as np
import os
import matplotlib.pyplot as plt
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