import h5py
import math
import sklearn
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import rasterio as rio
import seaborn as sns


###simulating step slip during postseismic deformation 
num = 16
chunk_size = 96
k = 0
for m in range(0, 10):
    f1 = h5py.File('real_noise/simulated_stack_%s.h5'%m, 'r')
    total_size = f1['timeseries'][()].shape
    data = f1['timeseries'][()]
    data[np.isnan(data)] = 0
    dem = tiff.imread('maduodem106.tif')
    print(np.max(data), np.min(data))
    for i in range(total_size[1] // chunk_size):
        for j in range(total_size[2] // chunk_size):
            defo0 = tiff.imread('simulated defo/%s.tif'%(k % 10000))
            defo = np.expand_dims(defo0 * 0, axis=0)
            if i == 0 and j == 0:
                print(np.max(defo), np.min(defo))
            a = np.random.uniform(0, 3)
            b = np.random.uniform(4, 12)
            z = np.random.uniform(1, 5)
            jie = np.random.randint(4, 12)
            for n in range(1, num):
                if n >= jie:
                    defo = np.vstack((defo, np.expand_dims(defo0 * a * math.log((1 + n / b), math.e), axis=0) * z))
                else:
                    defo = np.vstack((defo, np.expand_dims(defo0 * a * math.log((1 + n / b), math.e), axis=0)))
            with h5py.File('stack_defo_realnoise16-jieyue/simulated_stack_%s.h5'%k, 'w') as f2: ###
                f2.create_dataset('decorrelation', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('defo', data=defo)
                print(defo.shape)
                f2.create_dataset('stratified', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('turbulence', data=data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size])
                print(data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size].shape)
            tiff.imwrite('stack_defo_realnoise16-jieyue/maduocut_%s.tif'%k, dem[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size]) ###
            k += 1
    f1.close()


###simulating step slip during postseismic deformation 
num = 16
chunk_size = 96
k = 0
for m in range(0, 10):
    f1 = h5py.File('real_noise/simulated_stack_%s.h5'%m, 'r')
    total_size = f1['timeseries'][()].shape
    data = f1['timeseries'][()]
    data[np.isnan(data)] = 0
    dem = tiff.imread('maduodem106.tif')
    print(np.max(data), np.min(data))
    for i in range(total_size[1] // chunk_size):
        for j in range(total_size[2] // chunk_size):
            defo0 = tiff.imread('simulated defo/%s.tif'%(k % 10000))
            defo = np.expand_dims(defo0 * 0, axis=0)
            if i == 0 and j == 0:
                print(np.max(defo), np.min(defo))
            a = np.random.uniform(0, 3)
            b = np.random.uniform(4, 12)
            z1 = np.random.uniform(1, 3)
            z2 = np.random.uniform(3, 6)
            jie1 = np.random.randint(3, 7)
            jie2 = np.random.randint(7, 14)
            for n in range(1, num):
                if jie1 < n < jie2:
                    defo = np.vstack((defo, np.expand_dims(defo0 * a * math.log((1 + n / b), math.e), axis=0) * z1))
                elif n >= jie2:
                    defo = np.vstack((defo, np.expand_dims(defo0 * a * math.log((1 + n / b), math.e), axis=0) * z2))
                else:
                    defo = np.vstack((defo, np.expand_dims(defo0 * a * math.log((1 + n / b), math.e), axis=0)))
            with h5py.File('stack_defo_realnoise16-step/simulated_stack_%s.h5'%k, 'w') as f2: ###
                f2.create_dataset('decorrelation', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('defo', data=defo)
                print(defo.shape)
                f2.create_dataset('stratified', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('turbulence', data=data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size])
                print(data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size].shape)
            tiff.imwrite('stack_defo_realnoise16-step/maduocut_%s.tif'%k, dem[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size]) ###
            k += 1
    f1.close()


###simulating sigmoid slip during postseismic deformation 
num = 16
chunk_size = 96
k = 0
for m in range(0, 10):
    f1 = h5py.File('real_noise/simulated_stack_%s.h5'%m, 'r')
    total_size = f1['timeseries'][()].shape
    data = f1['timeseries'][()]
    data[np.isnan(data)] = 0
    dem = tiff.imread('maduodem106.tif')
    print(np.max(data), np.min(data))
    for i in range(total_size[1] // chunk_size):
        for j in range(total_size[2] // chunk_size):
            defo0 = tiff.imread('simulated defo/%s.tif'%(k % 10000))
            defo = np.expand_dims(defo0 * 0, axis=0)
            a = np.random.uniform(0, 3)
            b = np.random.uniform(4, 12)
            if i == 0 and j == 0:
                print(np.max(defo), np.min(defo))
            for n in range(1, num):
                defo = np.vstack((defo, np.expand_dims(defo0 *  a * math.log((1 + n / b), math.e) * 1 / (1 + np.exp(-n+8)), axis=0)))
            with h5py.File('stack_defo_realnoise16-sigmoid/simulated_stack_%s.h5'%k, 'w') as f2: ###
                f2.create_dataset('decorrelation', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('defo', data=defo)
                print(defo.shape)
                f2.create_dataset('stratified', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('turbulence', data=data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size])
                print(data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size].shape)
            tiff.imwrite('stack_defo_realnoise16-sigmoid/maduocut_%s.tif'%k, dem[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size]) ###
            k += 1
    f1.close()
  

###simulating linear slip 
num = 16
chunk_size = 96
k = 0
for m in range(0, 10):
    f1 = h5py.File('real_noise/simulated_stack_%s.h5'%m, 'r')
    total_size = f1['timeseries'][()].shape
    data = f1['timeseries'][()]
    data[np.isnan(data)] = 0
    dem = tiff.imread('maduodem106.tif')
    print(np.max(data), np.min(data))
    for i in range(total_size[1] // chunk_size):
        for j in range(total_size[2] // chunk_size):
            defo0 = tiff.imread('simulated defo/%s.tif'%(k % 10000))
            defo = np.expand_dims(defo0 * 0, axis=0)
            if i == 0 and j == 0:
                print(np.max(defo), np.min(defo))
            a = np.random.uniform(0, 3)
            b = np.random.uniform(4, 12)
            z = np.random.uniform(1, 5)
            jie = np.random.randint(4, 12)
            for n in range(1, num):
                if n >= jie:
                    defo = np.vstack((defo, np.expand_dims(defo0 * n, axis=0)))
                else:
                    defo = np.vstack((defo, np.expand_dims(defo0 * n, axis=0)))
            with h5py.File('stack_defo_realnoise16-linear/simulated_stack_%s.h5'%k, 'w') as f2: ###
                f2.create_dataset('decorrelation', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('defo', data=defo)
                print(defo.shape)
                f2.create_dataset('stratified', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('turbulence', data=data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size])
                print(data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size].shape)
            tiff.imwrite('stack_defo_realnoise16-linear/maduocut_%s.tif'%k, dem[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size]) ###
            k += 1
    f1.close()


###simulating logarithmic-decay afterslip
num = 16
chunk_size = 96
k = 0
for m in range(0, 2):
    f1 = h5py.File('real_noise/simulated_stack_%s.h5'%m, 'r')
    total_size = f1['timeseries'][()].shape
    data = f1['timeseries'][()]
    data[np.isnan(data)] = 0
    dem = tiff.imread('maduodem106.tif')
    print(np.max(data), np.min(data))
    for i in range(total_size[1] // chunk_size):
        for j in range(total_size[2] // chunk_size):
            defo0 = tiff.imread('simulated defo/%s.tif'%(k % 10000))
            defo = np.expand_dims(defo0 * 0, axis=0)
            if i == 0 and j == 0:
                print(np.max(defo), np.min(defo))
            a = np.random.uniform(0, 3)
            b = np.random.uniform(4, 12)
            for n in range(1, num):
                defo = np.vstack((defo, np.expand_dims(defo0 * a * math.log((1 + n / b), math.e), axis=0)))
            with h5py.File('stack_defo_realnoise16/simulated_stack_%s.h5'%k, 'w') as f2: ###
                f2.create_dataset('decorrelation', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('defo', data=defo)
                print(defo.shape)
                f2.create_dataset('stratified', data=np.zeros((num, chunk_size, chunk_size)))
                print(np.zeros((num, chunk_size, chunk_size)).shape)
                f2.create_dataset('turbulence', data=data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size])
                print(data[:, i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size].shape)
            tiff.imwrite('stack_defo_realnoise16/maduocut_%s.tif'%k, dem[i * chunk_size: (i + 1) * chunk_size, j * chunk_size: (j + 1) * chunk_size]) ###
            k += 1
    f1.close()
