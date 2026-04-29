import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import ndimage
import sys

# ─────────────────────────────────────────────
# 1. DATA LOADING & VISUALISATION
# ─────────────────────────────────────────────

def load_example_data(path: str, example_num: int):

    mat_contents = scio.loadmat(path)
    
    amp = f'amplitudes{example_num}'
    dist = f'distances{example_num}'
    clo = f'cloud{example_num}'
    
    amplitudes = mat_contents.get(amp)
    distances = mat_contents.get(dist)
    cloud = mat_contents.get(clo)
    
    if amplitudes is None or distances is None or cloud is None:
        raise KeyError(f"Example {example_num} data not found in {path}")
    
    return amplitudes, distances, cloud

def main():
    
    # Trigger : python ex_01.py example1kinect.mat 1
    filepath = sys.argv[1] if len(sys.argv) > 1 else "./data/example1kinect.mat"
    number = sys.argv[2] if len(sys.argv) > 1 else 1

    print(f"Loading: {filepath} number  {number}")
   

    Amp, Dist, PC = load_example_data(filepath, number)
    print(f"  Amp shape:  {Amp.shape}")
    print(f"  Dist shape:  {Dist.shape}")
    print(f"  PC shape: {PC.shape}")

   


if __name__ == "__main__":
    main()
