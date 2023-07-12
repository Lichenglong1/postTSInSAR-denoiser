# A deep-learning-based denoiser to automatically recover clean time series deformation following an earthquake directly from input noisy TSInSAR, which is applicable if a time series InSAR including more than 16 SAR acquisitions is available.

# Installation
## Basic
- `Python` >= 3.8
## Modules

- `h5py`
- `numpy`
- `matplotlib` <= 3.5.1
- `rasterio`
- `tensorflow-gpu` >= 2.5.0

```shell
pip install -r requirements.txt
```


# Prediction
```shell
python postTSInSAR-denoiser.py
```
