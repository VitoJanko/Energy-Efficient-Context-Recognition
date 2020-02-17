import numpy as np
import matplotlib.pyplot as plt


def normalize_matrix(matrix, rows=False):
    mtrx = np.array(matrix).astype(float)
    if rows:
        for row in range(mtrx.shape[0]):
            if np.sum(mtrx[row]) > 0:
                mtrx[row] /= np.sum(mtrx[row])
            else:
                mtrx[row][row] = 1
        return mtrx
    return mtrx/np.sum(mtrx)


def reverse_first(lst):
    return [(1-l[0],) + tuple(l[1:]) for l in lst]


def accuracy_matrix(matrix):
    return sum([matrix[i][i] for i in range(len(matrix))])/float(np.real(sum([sum(m) for m in matrix])))


def get_transitions(stream, activities, count_self=True):
    transitions_real = np.zeros((len(activities), len(activities)))
    for i in range(len(activities)):
        for j in range(len(activities)):
            if i != j or count_self:
                transitions_real[i][j] = len(stream[(stream.shift() == activities[i]) & (stream == activities[j])])
    sums = transitions_real.sum(axis=1)
    sums[sums == 0] = 1
    transitions_real = (transitions_real.T/sums).T
    return transitions_real


def get_steady_state_trans(m):
    s = m.T-np.identity(len(m))
    e = np.linalg.eig(s)
    sols = list(map(abs,np.round(e[0], 10)))
    p = e[1][:,sols.index(min(sols))]
    steady = p/sum(p)
    steady = np.array([np.real(s) for s in steady])
    return steady


def get_steady_state(p):
    if type(p) == list:
        p = np.array(p)
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q, ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ, bQT)


# TODO: fix conditional
def average_length(matrix):
    return [1.0/(1-matrix[i][i]) if matrix[i][i] < 1 else 3 for i in range(len(matrix))]


def pareto_dominance(lst):
    lst = list(lst)
    lst.sort(key=lambda x: x[0])
    i = 1
    while i < len(lst):
        if i == 0:
            continue
        if lst[i][1] >= lst[i-1][1]:
            del lst[i]
            i -= 1
        elif lst[i][0] == lst[i-1][0] and lst[i][1] < lst[i-1][1]:
            del lst[i-1]
            i -= 1
        i += 1
    return lst


def f1_single(conf,i):
    TP = conf[i][i]
    FN = sum(conf[i])-TP
    FP = sum(conf[j][i] for j in range(len(conf)))-TP
    if TP == 0:
        return 0
    precision = TP/float(TP+FP)
    recall = TP/float(TP+FN)
    return 2*precision*recall/(precision+recall)


def f1_from_conf(conf):
    return sum(f1_single(conf, i) for i in range(len(conf)))/len(conf)


def pareto_solutions(configurations, tradeoffs):
    solutions = [(1-a, e, h) for (a,e),h in zip(tradeoffs, configurations)]
    pareto_points = pareto_dominance(solutions)
    conf_pareto = [h for a, e, h in pareto_points]
    trade_pareto = [(1-a, e) for a, e, h in pareto_points]
    return conf_pareto, trade_pareto


def draw_tradeoffs(plots, labels, xlim = None, ylim=None, name=None, reverse=True, pareto=False,
               short=False, points=None, folder="artificial", percentage=True, percentage_energy=False,
               scatter_indices=None, color_indices=None, text_factor=50, ylabel="Energy", dotted_indices=None,
               thick_indices=None, xlabel="Classification error"):
    """
    A tool for quick visualization of different trade-offs.

    Essentially a wrapper around matplotlib.pyplot that makes drawing and comparing different sets
    of trade-offs easier. As an input it expects a list of trade-offs and can plot them in the
    way that is standard for Pareto fronts: step-wise with the quality axis reversed so that the
    ideal point
    lies in the lower-left corner. It streamlines some other aspects, for example labeling, saving
    and making sure only Pareto-optimal points are drawn.

    While some utility parameters are presents for modifying the look of the graph, it is recommended
    to use  matplotlib.pyplot library directly for any complex drawing task.

    :param plots: a list of one or more trade-off sets. Each trade-off set is for example a result
                  of a different energy-optimization function. They can be represented in two
                  different ways. Either as list of tuples (each tuples representing quality, energy)
                  or a tuple of two lists (first list containing quality, second energy). All outputs
                  returned from energy-optimization functions already fit this criteria.
    :param labels: a list of labels in the same order as the trade-offs in the ``plots`` parameter
    :param xlim: a tuple with min and max for x-axis
    :param ylim: a tuple with min and max for y-axis
    :param name: None or name of a file where the figure is saved
    :param reverse: reverses the x-axis
    :param pareto: only Pareto-optimal points are drawn
    :param short: if true, the graph will be drawn as square instead of a rectangle
    :param points: a list of points of to be drawn [(x1,y1,s1), (x2,y2,s2)]. s1, s2, etc. are
                   optional and provide a way to annotate points
    :param folder: None or name of a folder where the figure is saved
    :param percentage: multiplies the x-axis by 100 (in order to transform the accuracy of 0.45 into
                       45%)
    :param percentage_energy: multiplies the x-axis by 100
    :param scatter_indices: indices of trade-offs sets that should be drawn un-connected
    :param color_indices: if not None, each element of this list determines the index of a color
                          (trade-offs with the same index will be drawn with the same color)
    :param text_factor: changing this parameter moves the label from/away the annotated ``points``
    :param ylabel: label for the y-axis
    :param dotted_indices: indices of trade-offs sets that should be drawn with dots
    :param thick_indices: indices of trade-offs sets that should be drawn thicker than the rest
    :param xlabel: label for the x-axis
    """
    cmap = plt.get_cmap("tab10")
    if short:
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=(20,10))
    for i, (p, n) in enumerate(zip(plots, labels)):
        if len(p) == 2:
            x = p[0]
            y = p[1]
        else:
            x = list(zip(*p))[0]
            y = list(zip(*p))[1]
        if reverse:
            x = [1-v for v in x]
        if pareto:
            x, y = zip(*pareto_dominance(zip(x, y)))
        if percentage:
            x = [100*v for v in x]
        if percentage_energy:
            y = [100*v for v in y]
        color = cmap(i)
        linestyle = "-"
        linewidth = None
        if dotted_indices is not None and i in dotted_indices:
            linestyle = ":"
        if thick_indices is not None and i in thick_indices:
            linewidth = 4
        if color_indices is not None:
            color = cmap(color_indices[i])
        if scatter_indices is None or (i not in scatter_indices):
            plt.step(x, y, label=n, where="post", color=color, linestyle=linestyle, linewidth=linewidth)
        else:
            plt.scatter(x, y, label=n, color=color)
    if points is not None:
        for i, point in enumerate(points):
            color = cmap(0)
            if percentage:
                point = list(point)
                point[0] = point[0]*100
            if percentage_energy:
                point = list(point)
                point[1] = point[1]*100
            plt.plot(point[0], point[1], 'bo', markersize=15, color=color)
            if len(point) > 2:
                plt.text(point[0]+0.01*text_factor, point[1]-0.005*text_factor, point[2], fontsize=18)
    plt.xlabel(xlabel, fontsize=26)
    if percentage:
        plt.xlabel(xlabel+" [%]", fontsize=26)
    plt.ylabel(ylabel, fontsize=26)
    plt.tick_params(labelsize=18)
    if len(plots) > 0:
        plt.legend(prop={'size': 24})
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if name is not None:
        plt.savefig('./'+folder+'/'+name+'.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()
