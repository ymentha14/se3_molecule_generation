import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def plot_r1_exp(points, target_points, pred_points, axes):
    ax = axes[0]
    ax.scatter(*points[:-1], color='#348ABD', label='input')
    ax.scatter(*target_points[:-1], color='orange', label='target')
    ax.scatter(target_points[1], target_points[0],
               color='g', label='desired prediction')
    ax.legend()

    ax = axes[1]
    ax.scatter(*points[:-1], color='#348ABD', label='input')
    ax.scatter(*target_points[:-1], color='orange', label='target')

    ax.scatter(*pred_points[:-1], color='#E24A33',
               label='effective prediction')
    ax.legend()
    stdize_plots(axes,
                 lambda ax: ax.get_ylim(),
                 lambda ax, minval, maxval: ax.set_ylim(minval, maxval))
    stdize_plots(axes,
                 lambda ax: ax.get_xlim(),
                 lambda ax, minval, maxval: ax.set_xlim(minval, maxval))


def plot_coordinates(target_points, predicted_points, coor_str='z'):
    coor_2_ax = {'x': 0, 'y': 1, 'z': 2}
    coor = coor_2_ax[coor_str]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes = axes.reshape(-1)
    ax = axes[0]
    ax.plot(target_points[coor])
    ax.set_xlabel('Point index in the spiral')
    ax.set_ylabel(f'{coor_str.upper()} coordinate')
    ax.set_title('Target')
    ax = axes[1]
    ax.plot(predicted_points[coor])
    ax.set_xlabel('Point index in the spiral')
    ax.set_ylabel(f'{coor_str.upper()} coordinate')
    ax.set_title('Model predictions')


def stdize_plots(axes, get_lim, set_lim):
    lows, highs = zip(*[get_lim(ax) for ax in axes])
    max_val = max(highs)
    min_val = min(lows)
    for ax in axes:
        set_lim(ax, min_val, max_val)


def viz_point_cloud(*args):
    """
    Visualize points clouds in 3D

    Args:
        [list of tensors]
    """

    color_map = {'src': '#348ABD', 'trgt': '#E24A33',
                 'pred': '#988ED5', 'clean_trgt': '#8EBA42'}
    # fig = plt.figure(dpi=100)
    fig = plt.figure()

    N = len(args)
    axes = []
    for i, pointss in enumerate(args):
        ax = fig.add_subplot(1, N, i+1, projection='3d')
        axes.append(ax)
        if len(pointss[0]) == 2:
            for points, label in pointss:
                color = color_map.get(label, np.random.rand(3,))
                ax.scatter3D(*points.transpose(), label=label,  color=color)
                ax.legend()
        else:
            for points in pointss:
                ax.scatter3D(*points.transpose())
    stdize_plots(axes,
                 lambda ax: ax.get_zlim(),
                 lambda ax, minval, maxval: ax.set_zlim(minval, maxval))
    stdize_plots(axes,
                 lambda ax: ax.get_ylim(),
                 lambda ax, minval, maxval: ax.set_ylim(minval, maxval))
    stdize_plots(axes,
                 lambda ax: ax.get_xlim(),
                 lambda ax, minval, maxval: ax.set_xlim(minval, maxval))
    plt.close()
    return fig


def viz_point_cloud_2d(*args):
    """
    Visualize points clouds in 2D

    Args:
        [list of tensors]
    """

    # fig = plt.figure(dpi=100)
    fig = plt.figure()

    N = len(args)
    axes = []
    for i, pointss in enumerate(args):
        ax = fig.add_subplot(1, N, i+1)
        axes.append(ax)
        if len(pointss[0]) == 2:
            for points, label in pointss:
                ax.scatter(*points.transpose(), label=label)
                ax.legend()
        else:
            for points in pointss:
                ax.scatter(*points.transpose())
    stdize_plots(axes,
                 lambda ax: ax.get_ylim(),
                 lambda ax, minval, maxval: ax.set_ylim(minval, maxval))
    stdize_plots(axes,
                 lambda ax: ax.get_xlim(),
                 lambda ax, minval, maxval: ax.set_xlim(minval, maxval))
    plt.close()
    return fig
