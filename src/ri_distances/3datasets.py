"""
Functions to regenerate the 3 datasets plot
"""
from src.bunny.bunny import get_bunny_data, plot_bunny
from src.ri_distances.eval_predictor import *
from src.ri_distances.rotation_predictor import *
from src.se3.visualization import (viz_point_cloud, viz_point_cloud_2d,
                                   viz_src_trgt)


def test_rotation_inv(predictor, N_runs, thetas, data_func, noise_factor, N_pts):
    """Test  the rotation invariance of the predictor passed in parameter
    """
    params = []
    for theta in thetas:
        for _ in range(N_runs):
            p = DataParam(data_func, theta=theta, N_pts=N_pts,
                          noise_factor=noise_factor, permute=True, metric_name="SGW")
            params.append(p)
    results = evaluate_predictor(predictor, params)
    return results


def get_dummy_icp_ts(thetas, N_runs, data_func, noise_factor, N_pts, predictor):
    """
    Return the metrics for a dummy predictor and hte icp
    """
    res_icp = test_rotation_inv(
        predictor, N_runs, thetas, data_func, noise_factor, N_pts)

    dummy_predictor = DummyPredictor()
    res_dummy = test_rotation_inv(
        dummy_predictor, N_runs, thetas, data_func, noise_factor, N_pts)
    metric_icp = extract_theta_metrics(res_icp)
    metric_dummy = extract_theta_metrics(res_dummy)
    return metric_dummy, metric_icp


def extract_theta_metrics(results):
    """Generate the metric array N_runs x thetas in order to draw incertitude plots
    Returns:
        [np.array]: the metric array

    Args:
        results ([type]): [description]
    """
    results_df = pd.DataFrame(results)
    data = results_df.groupby('data_param').agg(list).reset_index()
    N_runs = results_df.groupby('data_param').count().iloc[0, 0]
    p = data['data_param'].iloc[0]
    predictor = data['predictor'].iloc[0][0]
    fname = p.data_func.__name__
    metric_ts = np.array(data['metric'].tolist())

    return metric_ts

def display_dummy_icp_inceritude_plots(thetas,dummy,icp,ax,title=""):
    ax.set_title(f"SGW loss vs angle for {title}")
    ax.set_xlabel('Angle in radians')
    ax.set_ylabel(f'SGW difference')
    incertitude_plot(thetas,icp,ax,label='ICP',color='#8EBA42')
    incertitude_plot(thetas,dummy,ax,label='No alignment',color='#988ED5')

def visualize_repr(data_func, N_pts, noise_factor, ax):
    repr_p = DataParam(data_func, theta=0, N_pts=N_pts,
                       noise_factor=noise_factor, permute=True, metric_name="SGW")
    src, clean, trgt = repr_p.generate_target()
    viz_src_trgt([(src, 'src'), (trgt, 'trgt')], ax=ax)


def main():

    # General parameters
    N_thetas = 8
    thetas = np.linspace(0, np.pi, N_thetas)
    predictor = IcpPredictor(max_iter=30, N_rots=5)
    N_runs = 8
    fig = plt.figure(figsize=(15, 8))

    # Spiral
    data_func = get_spiral
    noise_factor = 0.12
    N_pts = 40

    spiral_dummy, spiral_icp = get_dummy_icp_ts(
        thetas, N_runs, data_func, noise_factor, N_pts, predictor)
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    visualize_repr(data_func, N_pts, noise_factor, ax)
    ax = fig.add_subplot(2, 3, 4)
    display_dummy_icp_inceritude_plots(
        thetas, spiral_dummy, spiral_icp, ax, title="spiral")

    # Gaussian
    data_func = get_gaussian_point_cloud
    noise_factor = 0.06
    N_pts = 40

    gaussian_dummy, gaussian_icp = get_dummy_icp_ts(
        thetas, N_runs, data_func, noise_factor, N_pts, predictor)
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    visualize_repr(data_func, N_pts, noise_factor, ax)
    ax = fig.add_subplot(2, 3, 5)
    display_dummy_icp_inceritude_plots(
        thetas, gaussian_dummy, gaussian_icp, ax, title="gaussian")

    # Bunny
    data_func = get_bunny_data
    noise_factor = 0.04
    N_pts = 800

    bunny_dummy, bunny_icp = get_dummy_icp_ts(
        thetas, N_runs, data_func, noise_factor, N_pts, predictor)
    ax = fig.add_subplot(2, 3, 3, projection='3d')
    repr_p = DataParam(data_func, theta=0, N_pts=N_pts,
                       noise_factor=noise_factor)
    src, clean, trgt = repr_p.generate_target()
    plot_bunny(src, ax, color='#348ABD', label='src')
    plot_bunny(trgt, ax, color='#E24A33', label='trgt')
    ax = fig.add_subplot(2, 3, 6)
    fig.savefig("results/3dataset.png", bbox_inches='tight')


if __name__ == '__main__':
    main()
