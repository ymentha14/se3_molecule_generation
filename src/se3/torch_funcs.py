import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from numpy.linalg import norm
from se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from src.ri_distances.pnt_cloud_generation import (center, center_batch,
                                                   to_numpy_array,
                                                   to_torch_tensor)
from src.se3.visualization import viz_point_cloud
from tqdm import trange

plt.style.use("ggplot")

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float32)

# Type conversion functions


def predict(model, input_tens):
    """
    Infer the prediction of the model passed in parameter for the input tensor
    passed inparameter with constant features set to 1

    Args:
        model (torch.model): transformer
        input_tens (torch.Tensor): point cloud
    """
    feats = torch.ones(input_tens.shape[0], input_tens.shape[1], 1).to(device)
    predicted_deltas_tens = model(feats, input_tens.to(device), return_type=1)
    predicted_deltas_tens = predicted_deltas_tens.cpu().reshape(input_tens.shape)
    return predicted_deltas_tens


def get_batch(src_gen, trgt_gen, batch_size=10):
    """
    Return a batch of batch_size of results as in get_rotated_src_target_spirals

    Args:
      batch_size (int): number of samples in the batch
      factor (float): scaling factor for the spiral

    Return:
      [torch.tensor,torch.tensor]: src and target batches

    """
    batch_points = [src_gen.generate() for _ in range(batch_size)]
    batch_targets = [trgt_gen.generate() for _ in range(batch_size)]
    batch_points = [to_torch_tensor(i) for i in batch_points]
    batch_targets = [to_torch_tensor(i) for i in batch_targets]
    return torch.cat(batch_points), torch.cat(batch_targets)


def get_model(model_path=None):
    """
    Return fresh model if model_path is not provided
    """
    transformer = SE3Transformer(
        dim=1,  # feature dimension
        depth=3,  # number of attention block SE3
        input_degrees=1,
        num_degrees=2,
        output_degrees=2,
        reduce_dim_out=False,
    )

    if model_path is not None:
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path}")
        transformer.load_state_dict(torch.load(model_path))
    _ = transformer.to(device)
    transformer = transformer.to(torch.float32)
    return transformer


def get_r1_src_target(k_in=2, k_out=1):
    """
    Returns the r1 experiment src and targets tensors
    """
    # torch.set_default_dtype(torch.float64) # works best in float64
    points_tens = torch.tensor(
        [[-k_in, 0, 0], [k_in, 0, 0]]).unsqueeze(0).float()
    target_points_tens = (
        torch.tensor([[0, -k_out, 0], [0, k_out, 0]]).unsqueeze(0).float()
    )
    return points_tens, target_points_tens


def train_one_epoch(
    model,
    optimizer,
    epoch,
    criterion,
    batch_size,
    scheduler,
    device,
    src_gen,
    trgt_gen,
    use_wandb=True,
    tb_writer=None,
    center_output=False,
    asym_features=False
):
    """
    Train the model passed in parameter for one batch (epoch)
    """
    model.train()
    # generate batch
    batch_points, batch_target_points = get_batch(
        src_gen=src_gen, trgt_gen=trgt_gen, batch_size=batch_size)
    batch_points = batch_points.to(device)
    batch_target_points = batch_target_points.to(device)

    # constant features
    batch_size = batch_points.shape[0]
    N_points = batch_points.shape[1]
    if asym_features:
        feats = torch.ones(batch_size, N_points, 1)
    else:
        feats = torch.arange(N_points).unsqueeze(1).repeat(batch_size, 1, 1)
    feats = feats.to(device)
    predicted_deltas = model(feats, batch_points, return_type=1)
    if center_output:
        predicted_deltas = center_batch(predicted_deltas)
    predicted_deltas = predicted_deltas.reshape(batch_target_points.shape)
    predicted_points = batch_points + predicted_deltas

    loss = criterion(predicted_points, batch_target_points)

    # Tensorboard logger
    if tb_writer is not None:
        tb_writer.add_scalar("Loss/train", loss.item(), epoch)

    # Wandb logger
    if use_wandb:
        wandb.log({"loss": loss.item()})
    # writer.add_scalar('Loss/dist', dist.item(), epoch)

    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    optimizer.zero_grad()
    # del predicted_deltas, predicted_points, loss, dist, uncentered_batch_points, uncentered_batch_targets_points, feats, batch_target_points, batch_points
    # torch.cuda.empty_cache()
    return loss


def start_training(model, lr, optimizer, epochs, criterion, batch_size, scheduler, device, src_gen, trgt_gen, center_output, asym_features=False, use_wandb=False):
    if use_wandb:
        wrun = wandb.init("se3_runs")
        config = wandb.config
        config.criterion = criterion
        config.optimizer = optimizer
        config.lr = lr
        config.center_input = src_gen.centering
        config.center_target = trgt_gen.centering
        config.center_output = center_output
        config.src_gen = src_gen
        config.trgt_gen = trgt_gen

    t_bar = trange(epochs, leave=False, desc="Epochs progression")
    for epoch in t_bar:
        loss = train_one_epoch(model=model,
                               use_wandb=use_wandb,
                               optimizer=optimizer,
                               epoch=epoch,
                               criterion=criterion,
                               batch_size=batch_size,
                               scheduler=scheduler,
                               device=device,
                               src_gen=src_gen,
                               trgt_gen=trgt_gen,
                               center_output=center_output,
                               asym_features=asym_features)
        t_bar.set_description(
            f'Epoch progression: loss: {loss:.8f}')


def get_predictions(transformer, src_gen, trgt_gen, center_output):

    points_raw = src_gen.generate()
    target_points_raw = trgt_gen.generate()
    points_tens, target_points_tens = to_torch_tensor(
        points_raw
    ), to_torch_tensor(target_points_raw)
    # points_tens_raw,target_points_tens_raw = rotate(points_tens_raw,target_points_tens_raw)
    feats = (
        torch.ones(points_tens.shape[0],
                   points_tens.shape[1], 1).double().to(device)
    )

    predicted_deltas_tens = predict(transformer, points_tens)

    if center_output:
        predicted_deltas_tens = center_batch(predicted_deltas_tens)
    predicted_points_tens = points_tens + predicted_deltas_tens

    points = to_numpy_array(points_tens)
    target_points = to_numpy_array(target_points_tens)
    predicted_points = to_numpy_array(predicted_points_tens)
    return points, target_points, predicted_points


def visualize_prediction(points, target_points, predicted_points):
    return viz_point_cloud(
        [(points, "src"), (target_points, "trgt")],
        [(points, "src"), (predicted_points, "pred")],
    )


class MachineScaleChecker:
    """
    Class to check whether the equivariances at are machine scale or not
    """

    def __init__(self, transform, transformer, N=25):
        self.transformer = transformer
        self.transform = transform
        self.N = N
        self.shift = 500
        self.scale = 1000

    def get_machine_scale_error(
        self,
    ):
        """
        Check if the rotation equivariance is at machine scale

        Args:
          transform (function): rotation or translation of a point in 3D
          N (int): number of point in the point cloud

        Return:
          [float]: loss between the pre and post rotated outputs
        """

        # input point cloud
        points_tens = torch.rand(1, self.N, 3) * \
            self.scale  # we scale the noise
        # and shift the point cloud
        points_tens += torch.tensor([self.shift, self.shift, self.shift])

        transf_points_tens = self.transform(points_tens)

        # pre-rotated
        pre_output_tens = predict(self.transformer, transf_points_tens)

        # post-rotated
        output_tens = predict(self.transformer, points_tens)
        post_output_tens = self.transform(output_tens)

        loss = torch.nn.MSELoss()(post_output_tens, pre_output_tens)
        return loss.item()
