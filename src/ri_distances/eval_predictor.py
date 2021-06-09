"""
Methods to evaluate a single predictor whose parameters are passed as arguments to the script
The only parameter that changes is the number of point cloud, in order to assess how the predictor
in question scales in terms of time and performance (SGW or MSE)
"""

import argparse
import pdb
import pickle as pk
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from pdb import post_mortem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from src.ri_distances.eval_data_param import DataParam, execute_run
from src.ri_distances.icp.icp import IcpPredictor
from src.ri_distances.pnt_cloud_generation import (get_gaussian_point_cloud,
                                                   get_spiral)
from src.ri_distances.rotation_predictor import MSE
from src.ri_distances.SGW.risgw import RisgwPredictor
from src.ri_distances.SGW.sgw_pytorch import sgw_gpu, sgw_gpu_np
from src.se3.visualization import viz_point_cloud
from tqdm import tqdm, trange

plt.style.use('ggplot')

torch.set_default_dtype(torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_predictor(predictor, params):
    """
    Evaluate many data parameters for a given predictor. In practice, the params should
    all be the same except for the number of points in the point cloud

    Args:
        predictor (Predictor): predictor to assess performances for
        params (list of DataParam): parameters to test

    Return:
        [list of dict]: list of results as returned by execute_run
    """
    results = []
    for p in tqdm(params, desc='Parameters iteration', leave=False):
        # target point cloud is the point cloud we aim to obtain
        src_pnt_cloud,clean_trgt,trgt_pnt_cloud= p.generate_target()
        result = execute_run(predictor, p, src_pnt_cloud, trgt_pnt_cloud,
                    clean_trgt)
        results.append(result)
    return results


def incertitude_plot(x_ticks, ts, ax):
    """
    Plot an incertitue plot in terms of quartile

    Args:
        x_ticks (np.array): values to plot on the x axis
        ts (np.array): values to plot and extract the incertitued for
        ax (plt.axis): ax to plot on
    """
    q75 = np.quantile(ts, 0.75, axis=1)
    q25 = np.quantile(ts, 0.25, axis=1)
    q50 = np.median(ts, axis=1)
    ax.plot(x_ticks, q50)
    ax.fill_between(x_ticks, q25, q75, alpha=0.2)


def plot_time(n_points, times_ts, ax):
    """
    generate and incertitude plot for the time data passed in parameter
    """
    incertitude_plot(n_points, times_ts, ax)
    ax.set_xlabel('Number of points in the point cloud')
    ax.set_ylabel('Time taken (second)')


def plot_metric(n_points, metric_ts, metric_name,ax):
    """
    generate an incertitude plot for the metric data passed in parameter
    """
    incertitude_plot(n_points, metric_ts, ax)
    ax.set_xlabel('Number of points in the point cloud')
    ax.set_ylabel(f'{metric_name} difference')


def display_predictor_metrics_vs_pnt_cloud_size(results):
    """
    Display a given predictor metrics, that is, its SGW or MSE and time
    vs number of point cloud

    Args:
        results (list of dict): list of dict as outputed by
    """

    # TODO: ensure there is only one predictor in the dataframe
    results_df = pd.DataFrame(results)
    data = results_df.groupby('data_param').agg(list).reset_index()
    N_runs = results_df.groupby('data_param').count().iloc[0, 0]
    p = data['data_param'].iloc[0]
    predictor = data['predictor'].iloc[0][0]
    fname = p.data_func.__name__
    n_points = data['data_param'].apply(lambda x: x.N_pts)
    times_ts = np.array(data['time'].tolist())
    MSE_ts = np.array(data['metric'].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Func={fname},N_runs={N_runs},permute={p.permute},noise={p.noise_factor}\n{predictor}",
        fontsize=15)
    plot_time(n_points, times_ts, axes[1])
    plot_metric(n_points, MSE_ts, metric_name=p.metric_name,ax=axes[0])
    return fig


def main():
    run_name = time.strftime("%m_%d_%Y_%H_%M_%S")
    run_path = Path("models").joinpath(run_name).with_suffix('.pk')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '-d', '--dataset', help='dataset type(spiral="s",gaussian="g")', default='s')
    parser.add_argument(
        '-m', '--metric_func', help='metric function type(MSE="mse",sgw="sgw")', default='mse')
    parser.add_argument('-N', '--N_runs', help='Number of experiments',
                        type=int, default=1)
    parser.add_argument('-p', '--permute', default=True, action='store_true')
    parser.add_argument('-s', '--step_size', default=10, type=int)
    parser.add_argument('-f', '--noise_factor', default=0.0, type=float)
    parser.add_argument('-q', '--quick', default=False, action='store_true')
    args = parser.parse_args()

    N_runs = args.N_runs
    N_rots = 5 if args.quick else 50
    predictor = IcpPredictor(max_iter=100, N_rots=N_rots)

    if args.dataset == 's':
        data_func = get_spiral
    elif args.dataset == 'g':
        data_func = get_gaussian_point_cloud
    else:
        raise ValueError(
            f"Option {args.dataset} not recognized for the dataset.")

    # Type of metric function
    if args.metric_func == 'mse':
        metric_func = MSE
        metric_name = "MSE"
    elif args.metric_func == 'sgw':
        metric_func = sgw_gpu_np
        metric_name = "SGW"
    else:
        raise ValueError(
            f"Option {args.metric_func} not recognized for the metric function.")

    data_params = []
    for n_points in range(10, 100, args.step_size):
        data_param = DataParam(N_pts=n_points,
                               data_func=data_func,
                               metric_func=metric_func,
                               permute=args.permute,
                               noise_factor=args.noise_factor,
                               metric_name=metric_name)
        data_params.append(data_param)

    results = []
    for _ in trange(N_runs, desc="Run number:", leave=True):
        results += evaluate_predictor(predictor, data_params)
        pk.dump(results, run_path.open('wb'))


if __name__ == '__main__':
    main()
