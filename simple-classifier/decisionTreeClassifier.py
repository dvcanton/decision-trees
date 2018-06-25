import matplotlib.pyplot as plt
import pydotplus
import collections
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
from sklearn import tree

# Visualizing the decision tree classifier results
def visualize(model, x, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    ax.scatter(X[:,0], X[:,1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))

    print(xlim)
    print('XX type: ')
    print(type(xx))
    print('XX shape: ')
    print(xx.shape)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)


    ax.set(xlim=xlim, ylim=ylim)
    plt.show()

# Data: Generic isoteric Gausian blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)


# Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)
visualize(model, X, y)

# Visualize data
dot_data = tree.export_graphviz(model,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
