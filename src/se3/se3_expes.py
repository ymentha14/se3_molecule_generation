"""
SE(3) transformer overfit experiments as presented in the appendix of the paper
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from src.ri_distances.pnt_cloud_generation import SpiralGenerator
from src.se3.torch_funcs import (get_model, get_predictions, start_training,
                                 visualize_prediction)
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.style.use('ggplot')
torch.set_default_dtype(torch.float32) # works best in float64


def run(i,src_kwargs,trgt_kwargs,quick=False,use_wandb=False):
    """Execute a run of SE(3) transformer overfit given the parameters passed for the
    source and target point clouds

    Args:
        i (int): experiment number
        src_kwargs (dict): kwargs for the source SpiralGenerator
        trgt_kwargs (dict): kwargs for the target SpiralGenerator
        quick (bool): debugging option
    """
    if quick:
        epochs = 2
        batch_size = 1
    else:
        epochs = 100
        batch_size = 4

    transformer = get_model()
    criterion = torch.nn.MSELoss()
    lr = 0.01
    optimizer = torch.optim.Adam(transformer.parameters(),lr=lr)
    # optimizer = torch.optim.SGD(transformer.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        patience=100,
                                                        factor=0.4,
                                                        threshold=0.001,
                                                        verbose=True)
    center_output = False

    src_gen = SpiralGenerator(**src_kwargs)
    trgt_gen = SpiralGenerator(**trgt_kwargs)
    points, target_points = src_gen.generate(),trgt_gen.generate()

    # Training phase
    start_training(transformer,lr,optimizer,epochs,criterion,batch_size,scheduler,device,src_gen,trgt_gen,center_output,use_wandb)

    # Visualization of the prediction
    points, target_points, predicted_points = get_predictions(transformer, src_gen,trgt_gen, center_output)
    fig = visualize_prediction(points, target_points, predicted_points)
    if use_wandb:
        wandb.log({"chart": wandb.Image(fig)})
    fig.savefig(f"results/{i}.png",bbox_inches='tight')

def main():
    """
    Reproduce the SE(3) transformer experiments present in the appendix
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-q', '--quick', help='Quick option for debugging',action='store_true')
    parser.add_argument('-w', '--wandb', help='Whether to log metrics in wandb',action='store_true')
    args = parser.parse_args()

    if device.type == 'cpu':
        user_choice = input("The detected device is a CPU: do you still want to execute training? (y/n)")
        if user_choice != 'y':
            return 0
    expes_kwargs = [({},{"shift":0.3}),
                    ({"shift":2.0,},{"scaling":2.0,"shift":2.0}),
                    ({"centering":True},{"asym":True,"centering":True,"width_factor":1.5}),
                    ({"centering":True,"asym":True},{"asym":True,"centering":True,"width_factor":1.5}),
                    ({"asym":True,"shift":0.3},{"asym":True,"shift":1.0})]

    for i,(src_kwargs,trgt_kwargs) in enumerate(tqdm(expes_kwargs,desc="SE(3) overfits experiments",leave=False)):
        run(i,src_kwargs,trgt_kwargs,args.quick,args.wandb)
        sys.stdout.flush()



if __name__ == '__main__':
    main()
