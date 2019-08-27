#%load code/plot_xyw.py
# Author: Youngsup Kim, idebtor@gmail.com
# 2018.03.01 - creation
# 2018.04.08 - handles an array of weights W, X0 options
# 2018.04.18 - works with plot_decision_region

import matplotlib.pyplot as plt
import numpy as np

def plot_xyw(x, y, W=None, X0=False, title='Perceptron',
             classes=['class1', 'class2'], annotate=False, savefig=None):
    """ plots data x and its class label y as well as the the linear decision
        boundary and and the value W[-1] or w. 

        x(m, 2): m training samples with two features, x1 and x2 only.
                 Its shape is (m, 2); X0 must be set to False.
        x(m, 3): m training samples with two features x0=1, x1, x2
                  its shape is (m, 3); X0 must be set to True.
        y(m): m number of class labels, each value may be either 1 or -1,
              also it may be either 1 or 0

        w(3,): only one boundary to display
               If you have an array of w's, but want to plot the last one, pass W[-1].
        W(epochs, 3): epochs number of decision boundaries or weights
              If there is one set of weights, its shape can be either (3, ) or (1, 3)

        X0: X has x_0 = 1 term in all samples or not; if True, removed before plotting
        annotate: add a sequence number at each sample if True
        savefig: save the plot in a file if a filename is given
    """
    if X0 == True:      # remove the first column; change shape(6x3) into shape(6x2)
        x = x[ : , 1:]     # check a column?: np.all(X == X[0,:], axis = 0)[0] == True and X[0,0] == 1.0

    # setting min max range of data - 10% of margin allowed in four sides
    rmin, rmax = np.array(np.min(x)), np.array(np.max(x))
    rmin -= (rmax - rmin) * 0.1
    rmax += (rmax - rmin) * 0.1

    nums = ['  {}'.format(i+1) for i in range(len(y))]    # numbering dots

    for num, ix, iy in zip(nums, x, y):
        if annotate == True:
            plt.annotate(num, xy=ix)
        ## can be replaced using plt.scatter
        ##if iy == 1:
        ##    c1, = plt.plot(ix[0], ix[1], label='class 1', marker='s', color='blue')
        ##else:
        ##    c2, = plt.plot(ix[0], ix[1], label='class 2', marker='o', color='orange')

    # This handles class 1 and -1, class 1 and 0 as well.
    plt.scatter(x[y==1, 0], x[y==1, 1], label=classes[0], marker='s', s=9)
    plt.scatter(x[y!=1, 0], x[y!=1, 1], label=classes[1], marker='o', s=9)

    if W is not None:
        if W.ndim == 1:                             # one boundary in1-d array shape(3,)
            x1 = np.arange(rmin, rmax, .1)
            x2 = -W[0]/W[2] - W[1]/W[2]*x1
            plt.plot(x1, x2)
            title += ':w{}'.format(np.round(W, 2))          #display the weights at title
        else:
            for w in W:                                     # for every decision boundary
                x1 = np.arange(rmin, rmax, .1)
                x2 = -w[0]/w[2] - w[1]/w[2]*x1
                #display all decision boundaries and legend-weights
                plt.plot(x1, x2, label='w:{}'.format(np.round(w, 2)))
            title += ':w{}'.format(np.round(W[-1], 2))     #display the last weights at title

    plt.axhline(0, linewidth=1, linestyle='dotted')
    plt.axvline(0, linewidth=1, linestyle='dotted')
    plt.xlim([rmin, rmax])
    plt.ylim([rmin, rmax])
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel('$x_1$', fontsize=16)
    plt.ylabel('$x_2$', fontsize=16)
    if W is not None and W.ndim != 1:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(loc='best')
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight', dpi=150)
