from cmath import exp, pi, sqrt
import math
import numpy as np
import OpenEXR
import os
import re
import time


from Imath import PixelType
from PIL import Image
from numba import njit, prange
from scipy import signal

    
def load_rgbd(impath):
  E = OpenEXR.InputFile(impath)
  dw = E.header()['dataWindow']
  width = dw.max.x - dw.min.x + 1
  height = dw.max.y - dw.min.y + 1

  rgb = np.zeros((height, width, 3), dtype=np.float32)
  for i, c in enumerate('RGB'):
    buffer = E.channel(c, PixelType(OpenEXR.FLOAT))
    rgb[:, :, i] = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)

  buffer = E.channel('Z', PixelType(OpenEXR.FLOAT))
  depth = np.frombuffer(buffer, dtype=np.float32).reshape(height, width)

  E.close()
  return rgb, depth

def save_srgb(rgb, outpath):
  srgb = np.where(rgb <= 0.0031308,
                  12.92 * rgb,
                  (1 + 0.055) * np.power(rgb, 1 / 2.4) - 0.055)
  srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)
  Image.fromarray(srgb).save(outpath)


def gaussian_kernel(size, std):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)


def kernel_db_std(db_size):
    db = []
    for i in range(db_size):
        # [0,999] db.append(gaussian_kernel(7,i))
        # [0.0,99.9] db.append(gaussian_kernel(7,i/10))
        # [0.00,9.99] db.append(gaussian_kernel(7,i/100))
        # [1.00, 10.99] db.append(gaussian_kernel(7,i/100 + 1))
        # [0.1, 100.0] db.append(gaussian_kernel(7,i/10 + 0.1))
        db.append(gaussian_kernel(17,i/10 + 0.5))
    tuple(map(tuple, db))
    return db

def kernel_db_size(db_size):
    db = []
    disp_i = 1
    for i in range(db_size):
        db.append(gaussian_kernel(disp_i,1))
        disp_i = disp_i + 2
        
    return db

def kernel_db_size_std(db_size):
    db = []
    disp_i = 1
    for i in range(db_size):
        # [0.0,99.9] db.append(gaussian_kernel(disp_i,i/10))
        # [0,999] db.append(gaussian_kernel(disp_i,i))
        db.append(gaussian_kernel(disp_i, math.e**(i/5)))
        disp_i = disp_i + 2
        
    return db
        
        
            
@njit()
def convolution(rgb, depth, krnls_db, st_pl_depth):
    rgb_new = rgb
    for i in range(len(rgb)):
        for j in range(len(rgb[0])):
            
            
            rgb_new[i][j] = (0,0,0)
            krnl_size = 15
            
            if int(depth[i][j]) < len(krnls_db):
                krnl = krnls_db[int(depth[i][j])]
            else:
                krnl = krnls_db[len(krnls_db)-1]
            
            '''                
            if int(depth[i][j]) <= st_pl_depth:
                continue
            '''
            krnl_size = len(krnl)
            
            krnl_range = int((krnl_size - 1) / 2)
            
            krnl_i = 0
            for k in range(i-krnl_range, i+krnl_range+1):
                
                krnl_j = 0
                for h in range(j-krnl_range, j+krnl_range+1):
                    
                    
                    if h < 0 or h >= len(rgb[0]):
                        krnl_j += 1
                        continue
                    
                    
                    if k < 0 or k >= len(rgb):
                        krnl_j += 1
                        continue
                    
                    
                    rgb_new[i][j] += rgb[k][h] * krnl[krnl_i][krnl_j]
                    krnl_j += 1
                krnl_i += 1
            
                    
    return rgb_new
    
    
def main_convolution():

    (rgb, depth) = load_rgbd('TestImages/Scena Davide/rgbd.exr')
    
    start_time = time.time()
    
    krnl_db = kernel_db_std(1000)
    #krnl_db = kernel_db_size(20)
    #krnl_db = kernel_db_size_std(10)
    
    end_time = time.time()
    print("Database construction time is: ", str(end_time-start_time)+"s")
    
    start_plane_depth = 2 
    rgb = convolution(rgb, depth, krnl_db, start_plane_depth)
    
    end_time = time.time()
    print("Convolution time is: ", str(end_time-start_time)+"s")
    
    save_srgb(rgb, 'ResImages/finalSTDcheck(std 0.5 to 100.4)noparallel.png')
    

if __name__=='__main__':
    main_convolution()