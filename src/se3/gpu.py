
#GPU-related functions
import os
from copy import copy

import GPUtil as GPU
import humanize
import psutil
import torchg

  GPUs = GPU.getGPUs()
  # XXX: only one GPU on Colab and isn’t guaranteed
  gpu = GPUs[0]
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

def get_free_gpu():
  gpu = GPUs[0]
  return gpu.memoryFree

def del_all_tens():
  varo = copy(vars())
  for key,value in varo.items():
    if isinstance(value,torch.Tensor):
      mem_before = get_free_gpu()
      del vars()[key]
      torch.cuda.empty_cache()
      mem_after = get_free_gpu()
      mem_diff = mem_after - mem_before
      print(f"deleting {key} released {mem_diff:.2f} MB")
