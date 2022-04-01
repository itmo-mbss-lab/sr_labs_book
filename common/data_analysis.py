import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import numpy as np
import torch

from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tsne(data):
    """

    :param data:
    :return:
    """
    x = np.array([])
    y = np.array([])
    lang_id = 0
    for k, v in data.items():
        lang_id += 1

        embds = [v[k][0] for k in v.keys()]
        embds = torch.stack(embds)

        if x.size == 0:
            x = np.array(embds)
        else:
            x = np.concatenate((x, embds))

        if len(y) == 0:
            y = [k] * len(embds)
        else:
            y.extend([k] * len(embds))

    get_tsne_plot(x, y, title=None)


def get_tsne_plot(data_array, labels_array, title=None):
    """

    :param data_array:
    :param labels_array:
    :param output_file_name:
    :param title:
    :return:
    """
    custom_colors = ['red', 'cyan', 'dimgray',
                     'green', 'blue', 'darkorange', 'pink',
                     'teal', 'lime', 'yellow', 'royalblue',
                     'goldenrod']

    uniq_labels = set(labels_array)
    data_array = np.stack(data_array)
    labels_array = np.array(labels_array)

    tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000)
    vis_data = tsne.fit_transform(data_array)

    matplotlib.rcParams['figure.figsize'] = [20, 10]
    fig, ax = plt.subplots()

    use_custom_colors = False
    if len(uniq_labels) <= len(custom_colors):
        use_custom_colors = True

    for ind, uniq_lab in enumerate(uniq_labels):
        vis_data_ = vis_data[labels_array == uniq_lab]
        vis_x = vis_data_[:, 0]
        vis_y = vis_data_[:, 1]
        if use_custom_colors:
            ax.scatter(vis_x, vis_y, label=uniq_lab,
                       alpha=0.4, edgecolors='none', c=custom_colors[ind])
        else:
            ax.scatter(vis_x, vis_y, label=uniq_lab,
                       alpha=0.4, edgecolors='none')

    ax.legend()
    ax.grid(True)

    if title is not None:
        plt.title(title)
    show()
    return
