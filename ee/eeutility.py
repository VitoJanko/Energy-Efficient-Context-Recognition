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


def average_length(matrix):
    return [1.0/(1-matrix[i][i]) for i in range(len(matrix))]


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


def draw_graph(plots, labels, xlim = None, ylim=None, name=None, reverse=False, pareto=False,
               short=False, points=None, folder="artificial", percentage=True, percentage_energy=False,
               scatter_indices=None, color_indices=None, text_factor=50, ylabel="Energy", dotted_indices=None,
               thick_indices=None, xlabel="Classification error"):
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
