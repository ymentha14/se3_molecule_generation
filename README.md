
<!-- PROJECT LOGO -->
<br />
  <h1 align="center">Towards an efficient loss function for SE(3) equivariant molecule generation</h1>
<p align="center">
  <!-- <a href="https://github.com/ymentha14/Emoji Dataset"> -->
   <p align="center">
    An exploration of metrics for equivariant NN based molecule generation
  </p>
    <img src="images/logo.png" alt="Logo" width="" height="">
  </a>

</p>

## Table of Content

- [About the Project](#about-the-project)
- [Dataset Specifications](#dataset-specifications)
- [Getting Started](#getting-started)
  * [Notebooks](#notebooks)
  * [Dataset Generation](#dataset-generation)
    + [Dataset ( _`RESULTS_DIR/data/dataset`_ )](#dataset-----results-dir-data-dataset----)
    + [Demographic Information](#demographic-information)
  * [Figures and embeddings generation](#figures-and-embeddings-generation)
  * [Notebooks](#notebooks-1)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)






## About the Project
TODO: copy abstract
Equivariant neural networks such as the [SE(3) transformer](https://arxiv.org/abs/2006.10503) could help in task

## Getting Started
### Prerequisites
* `docker`: In order to guarantee results reproducibility, this project runs on docker: after having pulled the latest version of the project, run in the root of the repo:

Run:
```
make build_image
```
this will generate the docker image the project runs on: `se3_equiv` (it might take a few minutes).

Once this is done, you can now simply start a container by running:
```
make run_container
```
This will start a bash shell running in the container: from there you'll be able to run several commands to reproduce the figures present in the report. </br>
### NB
* The port 8888 is forwarded for jupyter notebooks and visualizations purpose
* The `results` directory is mounted as a volume: every modification in this directory will indeed reflect in the host machine. In particular, this is where the figures of the report are generated.

## Reproducibility

### Notebooks
```
make start_jupy
```
This command opens a jupyter lab environment running on the container and display it on the host port 8888. The notebooks display several commmented visualizations/ data processing for both loss functions exploration (`3d_pts_alignment.ipynb`) or SE(3) transformer overfit (`se3.ipynb`)

### Report Figures

#### Point cloud aligner algorithm comparison
In order to reproduce the figure `Loss vs time` as follows:
<img src="images/loss_vs_time.png" alt="Logo" width="" height="">
run

```
make loss_vs_time
```
Alternatively, you can modify several parameters like the type of dataset (spiral, gaussian etc) the number of runs per point aligner or the noise amplitude by running:

```
python src/ri_distances/eval_data_param.py --help
```

#### ICP scaling
Similarly, for the ICP scaling figures
<img src="images/icp_metrics_2.png" alt="Logo" width="" height="">
run

```
make icp_metrics
```

for help


```
python src/ri_distances/eval_predictor --help
```


#### SE(3) Appendix Figures
Finally, for every figure of SE(3) experiment such as:
<img src="images/3.png" alt="Logo" width="" height="">

run
```
make se3_expes
```
As the training of an SE(3) transformer can be relatively costly even on such small datasets, the program will prompt you to confirm you want to run this experience on a GPU-free device. In this case, the training might take > 30min.



## Project Structure
------------
    .

--------



## License
Distributed under the MIT License. See `LICENSE` for more information.



## Contact
[Project Link](https://github.com/ymentha14/se3_project)

yann.mentha@gmail.com



