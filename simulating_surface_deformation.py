from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import h5py

from deformation import okada

import os
from PIL import Image
import matplotlib.pyplot as plt


# TODO: utils.record_params_as_yaml
# TODO: downsample as data augmentation
def plot_data_cust(data, output, vm=None, cmap="RdBu"):
    vm = vm or np.max(np.abs(data))
    fig = plt.figure(figsize=(9, 6))
    axim = plt.imshow(data, cmap=cmap, vmin=-vm, vmax=vm)
    fig.colorbar(axim)
    #plt.text(1600, 0, 'cm')
    plt.title(output)
    plt.savefig(output)

def generate_stacks(
    dsets=["defo"],
    num_days=2,
    num_defos=100,
    defo_shape=(512, 512),
    add_day1_turbulence=False,
    turbulence_kwargs={},  # p0=1e-2,
    stratified_kwargs={},
    deformation_kwargs={},
):
    if not os.path.exists('simulated defo'):
        os.makedirs('simulated defo')
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                okada.random_displacement,
                shape=defo_shape,
                **deformation_kwargs,
            )
            for _ in range(num_defos)
        ]
        for i, future in enumerate(as_completed(futures)):
            defo, _, _ = future.result()
            defo *= 100
            los = _get_random_los()
            defo = okada.project_to_los(defo, los)
            Image.fromarray(defo).save('./test-on-synthetic1024/%s.tif'%i)
            data = defo*0.056/3.1415926/4*100
            plot_data_cust(-data, './%s.jpg'%i)

def _get_random_los():
    """Picks a random LOS, similar to Sentinel-1 range"""
    north = 0.1
    up = -1 * np.random.uniform(0.5, 0.85)
    east = np.sqrt(1 - up ** 2 - north ** 2)
    # sometimes ascending pointing east, sometimes descending pointing west
    east *= np.random.choice([-1, 1])
    return [east, north, up]

if __name__ == '__main__':
    generate_stacks()
