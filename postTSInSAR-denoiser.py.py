# Scott's data simulation repo
from simulated_stack_new16 import data_loader, load_all_data, plot_data, save_predictions_as_tifs

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import re
import argparse
import rasterio as rio
import cv2

tf.random.set_seed(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

parser = argparse.ArgumentParser(description='Sentinel Processing Wrapper')
parser.add_argument('-s', type=int, dest='start', default=None,
  help='Specify the start step in the config file. eg: -s Function-4')
parser.add_argument('-e', type=int, dest='end', default=None,
  help='Specify the end step in the config file. eg: -e Function-8')
args = parser.parse_args()
data_dir = 'stack_defo_realnoise16/'
model_out = 'denoiser/donised_CNN_InSAR'
#model_weights_out = 'denoiser/donised_CNN_InSAR.h5'
model_weights_out = 'train_out/model.150.h5'
train_out = 'train_out/'

initialize_model = True ### Setting True when from number=00*; else False
nums = list()
ssimss = list()
mssimss = list()
nrmses = list()
snrs = list()
# X, y = load_all_data(data_dir + "simulated_stack0*.h5", data_dir + "elevation_looked.dem", normalize=False)
for i in range(1):
    sim_file_list = list()
    dem_file_list = list()
    for number in range(0, 1): ###
        for j in range(args.start, args.end):
            sim_file_list.append(data_dir + "simulated_stack_%01i.h5"%j)
            dem_file_list.append(data_dir + "maduocut_%01i.tif"%j)
    X, y, snr = load_all_data(sim_file_list=sim_file_list, dem_file_list=dem_file_list, normalize=False)

    print(X.shape, y.shape)
    y = np.moveaxis(y, 1, -1)
    X = np.expand_dims(X, -1)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(X, y, snr, test_size=0.8, random_state=42) #0.9
    snrs.extend(snr_test)

    print(np.max(np.abs(y_train)))
    print(np.max(np.abs(y_test)))

    class MeanMaxScaler:
        def __init__(self):
            pass

        def fit(self, X):
            self.mean_ = np.average(X)
            self.scale_ = np.max(np.abs(X - self.mean_))
            return self
        
        def transform(self, X):
            return (X - self.mean_)/self.scale_
    
        def inverse_transform(self, X):
            return X * self.scale_ + self.mean_


    class MeanPctScaler:
        def __init__(self, pctile=95):
            self.percentile = pctile
            pass

        def fit(self, X):
            self.mean_ = np.average(X)
            self.scale_ = np.percentile(np.abs(X - self.mean_), self.percentile)
            return self
        
        def transform(self, X):
            return (X - self.mean_)/self.scale_
    
        def inverse_transform(self, X):
            return X * self.scale_ + self.mean_


    class LogMeanScaler:
        def __init__(self):
            pass

        def fit(self, X):
            self.shift_ = (0 if np.min(X) >= 0 else -np.min(X)) + .0001 # log10(0) is undefined so slight shift
            self.mean_ = np.average(np.log(X + self.shift_))
            return self
        
        def transform(self, X):
            return np.log(X + self.shift_) - self.mean_
    
        def inverse_transform(self, X):
            return np.exp(X + self.mean_) - self.shift_
    
    def unstack_dem_timeseries(X):
        dems = X[..., :1, :, :, :]
        timeseries = X[..., 1:, :, :, :]
        return dems, timeseries

    def stack_dem_timeseries(dems, timeseries):
        return np.concatenate((dems, timeseries), axis=1)

    train_dems, train_timeseries = unstack_dem_timeseries(X_train)
    test_dems, test_timeseries = unstack_dem_timeseries(X_test)

    dem_scaler = MeanPctScaler().fit(train_dems)
    scaled_train_dems = dem_scaler.transform(train_dems)
    scaled_test_dems = dem_scaler.transform(test_dems)
    print(scaled_train_dems.shape)

    print(train_dems.min(), train_dems.max())
    print(test_dems.min(), test_dems.max())

    print(scaled_train_dems.min(), scaled_train_dems.max())
    print(scaled_test_dems.min(), scaled_test_dems.max())

    timeseries_scaler = MeanPctScaler().fit(train_timeseries)
    scaled_train_timeseries = timeseries_scaler.transform(train_timeseries)
    scaled_test_timeseries = timeseries_scaler.transform(test_timeseries)
    print(scaled_train_timeseries.shape)

    print(train_timeseries.min(), train_timeseries.max())
    print(test_timeseries.min(), test_timeseries.max())

    print(scaled_train_timeseries.min(), scaled_train_timeseries.max())
    print(scaled_test_timeseries.min(), scaled_test_timeseries.max())

    # normalize targets
    scaled_y_train = timeseries_scaler.transform(y_train)
    scaled_y_test = timeseries_scaler.transform(y_test)
    print(scaled_y_train.shape)

    print(y_train.max(), y_train.min())
    print(y_test.max(), y_test.min())
    print(scaled_y_train.max(), scaled_y_train.min())
    print(scaled_y_test.max(), scaled_y_test.min())

    scaled_X_train = stack_dem_timeseries(scaled_train_dems, scaled_train_timeseries)
    scaled_X_test = stack_dem_timeseries(scaled_test_dems, scaled_test_timeseries)
    print(scaled_X_train.shape)
    if initialize_model:
        num_filters = 64
        conv_kernel = (4, 3, 3)
        pooling_kernel = (4, 1, 1) ###
        activation = layers.LeakyReLU(alpha=0.1)
        padding = "same"

        # Input
        px = None  # can hardcode as 48
        model_input = layers.Input(shape=(17, px, px, 1)) ###
        timeseries_input = model_input[..., 1:, :, :, :]
        dem_input = model_input[..., :1, :, :, :]
        timeseries_input.shape, dem_input.shape
        # encoder
        # following example of https://keras.io/examples/vision/autoencoder/

        # 4 convolution layers
        conv_1 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(timeseries_input)
        conv_2 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_1)
        conv_3 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_2)
        conv_4 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_3)
        #conv_4 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_4)
        
        # 1 pooling layer
        # shape goes from (36, 48, 48) to (6, 48, 48) after this pooling
        pool_5 = layers.MaxPooling3D(pooling_kernel, padding=padding)(conv_4)

        # 1 convolution layer
        conv_6 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(pool_5)
        conv_6 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_6)
        #conv_6 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_6)
        
        # 1 pooling layer
        # shape goes from (6, 48, 48) to (1, 48, 48) after this pooling
        # so time dimension is fully pooled
        pool_7 = layers.MaxPooling3D(pooling_kernel, padding=padding)(conv_6)
        

        # 1 pooling layer
        # shape goes from (1, 48, 48) to (6, 48, 48) after this pooling
        pool_9 = layers.UpSampling3D(size=pooling_kernel)(pool_7)

        # 1 convolution layer
        conv_10 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(pool_9)
        conv_10 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_10)
        #conv_10 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_10)

        # 1 pooling layer
        # shape goes from (6, 48, 48) to (36, 48, 48) after this pooling
        pool_11 = layers.UpSampling3D(size=pooling_kernel)(conv_10)
        dem_plus_pool = layers.Concatenate(axis=-1)([pool_11, tf.tile(dem_input, [1, 16, 1, 1, 1])])
        # 4 convolution layers
        conv_12 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(dem_plus_pool)
        conv_13 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_12)
        conv_14 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_13)
        #conv_15 = layers.Conv3D(num_filters, conv_kernel, activation=activation, padding=padding)(conv_14)
       
        # final output
        output = layers.Conv3D(1, conv_kernel, activation="linear", padding=padding)(conv_14)

        # Autoencoder
        # again from https://keras.io/examples/vision/autoencoder/
        autoencoder = Model(model_input, output)
    else:
        autoencoder = tf.keras.models.load_model(model_out, compile=False)

    def Loss(y_true, y_pred):
        y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 1])
        ssim = tf.image.ssim(y_true, y_pred, max_val=5)
        rmse = tf.sqrt(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(y_true, y_pred))
        nrmse = tf.reduce_mean((rmse - tf.reduce_mean(rmse)) / tf.math.reduce_std(rmse))
        return 1 - ssim + nrmse
    loss = Loss
    # loss = "mae" # -- what the original paper uses, according to their supplementary information.
    # For us it resulted in odd grid artifacts in the corners.

    autoencoder.compile(optimizer="adam", loss=loss)
    autoencoder.summary()

    autoencoder.load_weights(model_weights_out)


    predictions_scaled = autoencoder.predict(scaled_X_test)
    predictions_vis = timeseries_scaler.inverse_transform(predictions_scaled)
    predictions = tf.transpose(tf.squeeze(predictions_vis, -1), (0, 2, 3, 1))
    print(np.max(np.abs(predictions_scaled)))
    print(np.max(np.abs(predictions)))

    def plot_data_cust(X, y, output, name, n=9, vm=None, cmap='RdBu'):
        vm = vm or np.max(np.abs(X))/1.5
        fig = plt.figure(constrained_layout=True, figsize=(16, 16), dpi=300)
        num = 4
        grid = plt.GridSpec(9, 3 * num)
        ax11 = []
        for i in range(num):
            for j in range(num):
                ax11.append(fig.add_subplot(grid[i, j]))
        ax12 = []
        for i in range(num):
            for j in range(num, 2 * num):
                ax12.append(fig.add_subplot(grid[i, j]))
        ax13 = []
        for i in range(num):
            for j in range(2 * num, 3 * num):
                ax13.append(fig.add_subplot(grid[i, j]))
        ax21 = fig.add_subplot(grid[num: 9, : num])
        ax22 = fig.add_subplot(grid[num: 9, num: 2 * num])
        ax23 = fig.add_subplot(grid[num: 9, 2 * num: 3 * num])
        
        for a in range(num):
            for b in range(num):
                i = num * a + b
                ax11[i].imshow(X[i], cmap=cmap, vmin=-vm, vmax=vm)
        for a in range(num):
            for b in range(num):
                i = num * a + b
                axim = ax13[i].imshow(output[i], cmap=cmap, vmin=-vm, vmax=vm)
        if y is not None:
            for a in range(num):
                for b in range(num):
                    i = num * a + b
                    ax12[i].imshow(y[i], cmap=cmap, vmin=-vm, vmax=vm)
            index = np.unravel_index(np.argmax(y), y.shape)
            xs = []
            outputs = []
            ys = []
            for i in range(16):
                xs.append(X[i, index[1], index[2]][0])
                outputs.append(float(output[i, index[1], index[2]]))
                ys.append(float(y[i, index[1], index[2]]))
            ax21.scatter(list(range(1, 17)), xs)
            ax22.scatter(list(range(1, 17)), ys)
            ax23.scatter(list(range(1, 17)), outputs)
            pd.DataFrame({'t': list(range(1, 17)), 'xs': xs, 'ys': ys, 'outputs': outputs}).to_csv('../log-fitting/postseismic/x_y_output_%s.txt'%name, index=False, sep=' ')
            fig.savefig(train_out + 'prediction_%02d_%s.pdf'%(number, name))
        else:
            fig.savefig(train_out + 'prediction_%s.pdf'%name)

    def avg_ssim(pred_img, target_img):
        """Take the SSIM for each prediction - target pair in the test set, and return the average
        across the entire test set"""
        return float(tf.image.ssim(pred_img, target_img, 2, filter_size=8))

    # the paper appears to be using MSSIM not SSIM, but they are not taking the average anywhere
    # that's something I'm doing to get a nice single metric to use to evaluate the different
    # model runs I'm doing. 

    def avg_mssim(pred_img, target_img):
        """For each prediction-target pair in the set, take 8x8 windows, calculate SSIM, average
        across the image, then average across the entire test set"""
        # make patches
        filter_dims = (1,8,8,1)
        rates = (1,1,1,1)
        
        pred_img = tf.reshape(pred_img, (pred_img.shape[0], 96, 96, 16))
        preds_patches = tf.image.extract_patches(pred_img,
                                                 sizes=filter_dims,
                                                 strides=filter_dims,
                                                 rates=rates,
                                                 padding='VALID')
        preds_patches = np.array(preds_patches).reshape(preds_patches.shape[0] * 16, 72, 128)
        preds_patches = preds_patches.reshape(-1, 144, 8, 8)
        
        target_img = tf.reshape(target_img, (target_img.shape[0], 96, 96, 16))
        targs_patches = tf.image.extract_patches(target_img,
                                                 sizes=filter_dims,
                                                 strides=filter_dims,
                                                 rates=rates,
                                                 padding='VALID')
    
        # reshape for needed sizes for SSIM
        targs_patches = np.array(targs_patches).reshape(targs_patches.shape[0] * 16, 72, 128)
        targs_patches = targs_patches.reshape(-1, 144, 8, 8)
    
        # run SSIM for patches
        local_ssims = tf.image.ssim(preds_patches, targs_patches, max_val=2, filter_size=3)
        return np.average(local_ssims)

    for i in range(scaled_X_test.shape[0]): # change this seed and rerun this cell to see different examples
        x_img = X_test[i][1:]   # X_test is unscaled
        targ_img = tf.transpose(y_test[i], (2, 0, 1))   # y_test is unscaled
        y_img = tf.squeeze(predictions_vis[i], -1)   # predictions has been de-scaled
        print("Max target deformation: ", np.max(targ_img))

        plot_data_cust(x_img, targ_img, y_img, i)
        ssim = avg_ssim(y_img, targ_img)
        ssimss.append(ssim)
        print('ssim', ssim)
        mssim = avg_mssim(tf.expand_dims(y_img, axis=0), tf.expand_dims(targ_img, axis=0))
        mssimss.append(mssim)
        print('mssim', mssim)
        rmse = tf.sqrt(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(targ_img, y_img))
        nrmse = tf.reduce_mean(rmse / tf.math.reduce_std(targ_img))
        nrmses.append(float(nrmse))
        print('nrmse', float(nrmse))
        del x_img, targ_img, y_img
print('mean ssim', np.mean(ssimss))
print('mean mssim', np.mean(mssimss))
print('mean nrmse', np.mean(nrmses))
pd.DataFrame({'Number': list(range(len(ssimss))), 'ssim': ssimss, 'mssim': mssimss, 'snr': snrs, 'nrmse': nrmses}).to_csv(train_out + 'prediction_ssim_snr_%s.txt'%args.start, index=False, header=None, sep=' ')


import h5py
import numpy as np
testH5_path = '/home/n2208015e/CNN-InSAR-postseismic/troposim/real-noise.h5'
f = h5py.File(testH5_path, 'r')
X = f['timeseries'][:] * 100
f.close()
X[np.isnan(X)] = 0
print('X', X.shape)

watermask='/home/n2208015e/CNN-InSAR-postseismic/postseismic-figure/filter/watermask-final.tif'
with rio.open(watermask) as src:
    dem = src.read()
    dem = dem.astype(np.float32)
print(dem.shape)
print(np.max(dem))
print(np.min(dem))
X=np.where(dem == 0.0, 0.0, X)

def plot_data_cust(X, y, output, name, n=9, vm=None, cmap='RdBu'):
    vm = 3
    fig = plt.figure(constrained_layout=True, figsize=(32, 16), dpi=900)
    num = 4
    grid = plt.GridSpec(9, 2 * num)
    ax11 = []
    for i in range(num):
        for j in range(num):
            ax11.append(fig.add_subplot(grid[i, j]))
    ax12 = []
    for i in range(num):
        for j in range(num, 2 * num):
            ax12.append(fig.add_subplot(grid[i, j]))
    ax21 = fig.add_subplot(grid[num: 9, : num])
    ax22 = fig.add_subplot(grid[num: 9, num: 2 * num])
    
    #output = np.where(output[-1] < -0.01, 0, np.squeeze(X))
    for a in range(num):
        for b in range(num):
            i = num * a + b
            ax11[i].imshow(X[i], cmap=cmap, vmin=-vm, vmax=vm)
    for a in range(num):
        for b in range(num):
            i = num * a + b
            axim = ax12[i].imshow(output[i], cmap=cmap, vmin=-vm, vmax=vm)
    index = np.unravel_index(np.argmax(output[-1]), X.shape)
    #index = [0, 1250, 4500]
    print(index)
    xs = []
    outputs = []
    for i in range(16):
        xs.append(X[i, index[1], index[2]][0])
        outputs.append(float(output[i, index[1], index[2]]))
    ax21.scatter(list(range(1, 17)), xs)
    ax22.scatter(list(range(1, 17)), outputs)
    pd.DataFrame({'t': list(range(1, 17)), 'xs': xs, 'outputs': outputs}).to_csv('../log-fitting/postseismic/x_output_%s.txt'%name, index=False, sep=' ')
    fig.savefig(train_out + 'prediction_%s.pdf'%name)

import random
indices = random.sample(range(1, 43), 16)
indices = sorted(indices)
print(indices)
#indices = list(range(1, 17))
texas_tests = X[indices, 1500:2000, 2000: 2520, None]

import rasterio as rio
demfile1 = 'maduodem106.tif'
with rio.open(demfile1) as src:
    dem = src.read()
    dem = dem.astype(np.float32)
dems = dem[:, :texas_tests.shape[1], 2000: 2520]
print('dem', dems.shape)

dem = dems

dem = np.expand_dims(dem, -1)
dem = np.expand_dims(dem, 0)
texas_tests = np.expand_dims(texas_tests, 0)

dem_scaled = dem_scaler.transform(dem)
timeseries_scaled = timeseries_scaler.transform(texas_tests)

print(dem_scaled.shape)
print(timeseries_scaled.shape)

test_scaled = stack_dem_timeseries(dem_scaled, timeseries_scaled)
test_scaled.shape

texas_predictions_scaled = autoencoder.predict(test_scaled)
texas_predictions = timeseries_scaler.inverse_transform(texas_predictions_scaled)

for i in range(texas_tests.shape[0]):
    plot_data_cust(texas_tests[i], None, texas_predictions[i], 'real', vm=8)
    
    
    
    
import h5py
import numpy as np
testH5_path = '/home/n2208015e/CNN-InSAR-postseismic/troposim/stack_defo_noise16/simulated_stack_18.h5'
f = h5py.File(testH5_path, 'r')
X1 = f["stratified"][:]
X2 = f["turbulence"][:]
X3 = f["decorrelation"][:]
X4 = f["defo"][:]
f.close()
X=X1+X2
X[np.isnan(X)] = 0
print('X', X.shape)
texas_tests = X[:, :, : , None]

import rasterio as rio
demfile1 = '/home/n2208015e/CNN-InSAR-postseismic/troposim/stack_defo_noise16/maduocut_18.tif'
with rio.open(demfile1) as src:
    dem = src.read()
    dem = dem.astype(np.float32)
dems = dem[:, :,: ]
print('dem', dems.shape)

dem = dems

def plot_data_cust(X, y, output, name, n=9, vm=None, cmap='RdBu'):
    vm = vm or np.max(np.abs(X))
    fig = plt.figure(constrained_layout=True, figsize=(32, 16), dpi=900)
    num = 4
    grid = plt.GridSpec(9, 2 * num)
    ax11 = []
    for i in range(num):
        for j in range(num):
            ax11.append(fig.add_subplot(grid[i, j]))
    ax12 = []
    for i in range(num):
        for j in range(num, 2 * num):
            ax12.append(fig.add_subplot(grid[i, j]))
    ax21 = fig.add_subplot(grid[num: 9, : num])
    ax22 = fig.add_subplot(grid[num: 9, num: 2 * num])
    
    #output = np.where(output[-1] < -0.01, 0, np.squeeze(X))
    for a in range(num):
        for b in range(num):
            i = num * a + b
            ax11[i].imshow(X[i], cmap=cmap, vmin=-vm, vmax=vm)
    for a in range(num):
        for b in range(num):
            i = num * a + b
            axim = ax12[i].imshow(output[i], cmap=cmap, vmin=-vm, vmax=vm)
    index = np.unravel_index(np.argmax(output[-1]), X.shape)
    #index = [0, 1250, 4500]
    print(index)
    xs = []
    outputs = []
    for i in range(16):
        xs.append(X[i, index[1], index[2]][0])
        outputs.append(float(output[i, index[1], index[2]]))
    ax21.scatter(list(range(1, 17)), xs)
    ax22.scatter(list(range(1, 17)), outputs)
    pd.DataFrame({'t': list(range(1, 17)), 'xs': xs, 'outputs': outputs}).to_csv('../log-fitting/postseismic/x_output_%s.txt'%name, index=False, sep=' ')
    fig.savefig(train_out + 'prediction_%s.pdf'%name)


dem = np.expand_dims(dem, -1)
dem = np.expand_dims(dem, 0)
texas_tests = np.expand_dims(texas_tests, 0)

dem_scaled = dem_scaler.transform(dem)
timeseries_scaled = timeseries_scaler.transform(texas_tests)

print(dem_scaled.shape)
print(timeseries_scaled.shape)

test_scaled = stack_dem_timeseries(dem_scaled, timeseries_scaled)
test_scaled.shape

texas_predictions_scaled = autoencoder.predict(test_scaled)
texas_predictions = timeseries_scaler.inverse_transform(texas_predictions_scaled)

for i in range(texas_tests.shape[0]):
    plot_data_cust(texas_tests[i], None, texas_predictions[i], 'synthetic')   
    
    
    
    
    
    
    
    
    
    
