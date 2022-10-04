from array import array
from cmath import exp, pi, sqrt
from ctypes import sizeof
from distutils.command.upload import upload
import math
import numpy as np
import OpenEXR
import os
import re
import time


from Imath import PixelType
from PIL import Image
from numba import njit, prange
from numba.typed import List
from scipy import signal
import scipy.stats as st

    
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
  
  
def save_exr(rgb, outpath):
  r, g, b = np.split(rgb, 3, axis=-1)
  height, width = rgb.shape[:-1]
  header = OpenEXR.Header(width, height)

  exr = OpenEXR.OutputFile(outpath, header)
  exr.writePixels({'R': r.tobytes(),
                   'G': g.tobytes(),
                   'B': b.tobytes()})
  exr.close()


def gaussian_kernel(size, std):
    '''Returns a 2D Gaussian kernel array.'''
    kernel1d = signal.windows.gaussian(size, std=std)
    kernel2d = np.outer(kernel1d, kernel1d)
    return kernel2d / np.sum(kernel2d)

def kernel_db_std(db_size, k_size):
    db = []
    for i in range(db_size):
        db.append(gaussian_kernel(k_size,i/10 + 0.1))
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
        
            
def convolution(rgb, depth, krnls_db, krnl_size, focus):
    rgb_new = rgb*0
    
    krnl_range = int((krnl_size - 1) / 2)
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in range(krnl_range, image_width - krnl_range):
        for j in range(krnl_range, image_height - krnl_range):
            
            
            #Calcolo del circle of confusion
            #85mm to m  
            focal_length = 50.0 / 1000
            
            f = 1/focal_length + 1/focus
            magnification = f / (focus - f)
            Aperture = 1.8
            
            CoC = round(Aperture * magnification * (abs(depth[i-krnl_size+x][j-krnl_size+y] - focus) / depth[i][j]))
            
            
            if CoC < len(krnls_db):
                krnl = krnls_db[CoC]
            else:
                krnl = krnls_db[len(krnls_db)-1]
        
            
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i][j] += krnl[x][y] * rgb[i-krnl_size+x][j-krnl_size+y]
            
                    
    return rgb_new

def numpy_convolve(rgb, depth, krnls_db, krnl_size, focus):
    rgb_new = rgb*0
    
    krnl_range = int((krnl_size - 1) / 2)
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in range(krnl_range, image_width - krnl_range):
        for j in range(krnl_range, image_height - krnl_range):
            
            
            #Calcolo del circle of confusion
            #85mm to m  
            focal_length = 18.0 / 1000
            
            f = 1/focal_length + 1/focus
            magnification = f / (focus - f)
            Aperture = 1.8
            
            CoC = round(Aperture * magnification * (abs(depth[i-krnl_size+x][j-krnl_size+y] - focus) / depth[i][j]))
            
            
            if CoC < len(krnls_db):
                krnl = krnls_db[CoC]
            else:
                krnl = krnls_db[len(krnls_db)-1]
                    
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i][j] += krnl[x][y] * rgb[i-krnl_size+x][j-krnl_size+y]
    return rgb_new




'''
INIZIO NUOVO CODICE ------------------------------------------------------------------------------
'''
def upload_PSF(depth, xy, db, focus):
    # Caricamento dell PSF
    # Per ora restituisce un kernel gaussiano
    focal_length = 18.0 / 1000
            
    f = 1/focal_length + 1/focus
    magnification = f / (focus - f)
    Aperture = 1.8
            
    CoC = round(Aperture * magnification * (abs(depth - focus) / depth))
    krnl = db[CoC]
    
    return krnl 

   
def KernelBuilding(size, pos, depth, focus, db):
    
    kernel = []
    for i in range(size*size):
        kernel.append(0)
        
    kernelSum = 0
    
    for i in range(size):
        for j in range(size):
            
            psfij = upload_PSF(depth[i,j], pos, db,focus)
            ijvalue = psfij[i*size + j]
            kernel[i*size + j] = ijvalue
            kernelSum += ijvalue
    return kernel
   
'''
LINEARIZZARE IL KERNEL PER APPLICARE NUMBA
APPLICARE PARALLELIZZAZIONE ALLA SOTTOCONVOLUZIONE
'''
#Indice di linearizzazione (j-1)*krnl_size + i

@njit(parallel = True)
def psf_convolution(rgb, depth, krnls_db, krnl_size, focus, focal_length, aperture):
    
    rgb_new = rgb*0
    krnl_range = int((krnl_size - 1) / 2)
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in prange(krnl_range, image_width - krnl_range):
        for j in range(krnl_range, image_height - krnl_range):
            
            #krnl = KernelBuilding(krnl_size, (i,j), depth, focus, krnls_db)
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
    
            for h in range(krnl_size):
                for k in range(krnl_size):
                    
                    #psfij = upload_PSF(depth[i,j], pos, db,focus)
                    N = (focal_length)/aperture
                    dep = depth[i][j] * 1000
                    foc = focus * 1000
                    CoC = round((abs(dep - foc) / dep) * ((focal_length**2)/N*abs(foc-focal_length)) / 100000)
                        
                    if CoC < len(krnls_db):
                        psfij = krnls_db[CoC]
                    else:
                        psfij = krnls_db[len(krnls_db)-1]
                    
                    ijvalue = psfij[h][k]
                    krnl[h*krnl_size + k] = ijvalue
                    kernelSum += ijvalue
            
            for x in range(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i][j] += krnl[x*krnl_size+y] * rgb[i-krnl_size+x][j-krnl_size+y]
                      
    return rgb_new
       
'''
FINE NUOVO CODICE ------------------------------------------------------------------------------
'''
    

def main_convolution():

    '''
    #Caricamento dell'immagine exr
    #(rgb, depth) = load_rgbd('TestImages/Scena Davide/rgbd.exr')
    (rgb, depth) = load_rgbd('TestImages/pebble_scattering_2.exr')
    
    start_time = time.time()
    
    #Creazione del database di kernel
    krnl_db = kernel_db_std(10000, 7)
    #krnl_db = kernel_db_size(20)
    #krnl_db = kernel_db_size_std(10)
    
    end_time = time.time()
    print("Database construction time is: ", str(end_time-start_time)+"s")
    
    #Convoluzione
    rgb = convolution(rgb, depth, krnl_db, len(krnl_db[0]), 1)
    
    end_time = time.time()
    print("Convolution time is: ", str(end_time-start_time)+"s")
    
    #Salvataggio dell'immagine
    save_srgb(rgb, 'ResImages/pebbleCoCGAUSS(7 - 0.1_1000)[7]f[1][18mm][1.8]dopo.png')
    
    '''
    
    (rgb, depth) = load_rgbd('TestImages/flower_scattering.exr')
    
    start_time = time.time()
    
    krnl_db = kernel_db_std(10000, 15)
    
    end_time = time.time()
    print("Database construction time is: ", str(end_time-start_time)+"s")
    
    
    
    ker_size = 15
    focus = 6
    aperture = 45 #mm
    focal_length = 50 #mm
    #f-stop = focal_length / aperture 
    
    rgb = psf_convolution(rgb, depth, krnl_db, ker_size, focus, focal_length, aperture)
    
    end_time = time.time()
    print("Convolution time is: ", str(end_time-start_time)+"s")
    
    save_exr(rgb, 'ResImages/flower_f15['+str(focus)+']['+str(focal_length)+']['+str(focal_length/aperture)+'].exr')
    #save_srgb(rgb, 'ResImages/water2_f15['+str(focus)+']['+str(focal_length)+']['+str(focal_length/aperture)+'].png')
    '''
    for i in range(len(depth)):
        for j in range(len(depth[0])):
            print(depth[i][j])
    '''

    
    

if __name__=='__main__':
    main_convolution()