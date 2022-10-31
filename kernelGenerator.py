from numba import njit, prange, float64
import numpy as np
from scipy import ndimage as nim
import imagesManager as imaMan
import time
import os


def kernelFilesGenerator(krnl_size, camera_path, camera_name):
    
    krnl_range = int((krnl_size - 1) / 2)
    camera_new_path = '/home/lor3n/Documents/GitHub/PFSrendering/PSFkernels/'+camera_name+'_krnls'
    
    os.mkdir(camera_new_path, mode=0o777)
    
    for directory in os.listdir(camera_path):
        dir_clean = directory.split('-')
        dir_depth = float(dir_clean[1].split('m')[0])    

        directory_new_path = os.path.join(camera_new_path, directory)
        os.mkdir(directory_new_path, mode=0o777)
        dir_path = os.path.join(camera_path, directory)
        
        print(dir_depth)
        for file in os.listdir(dir_path):
            
            file_path = os.path.join(dir_path, file)
            big_psf = imaMan.load_psf(file_path)
            
            
            com = nim.center_of_mass(big_psf)
            
            shift_x, center_x = np.modf(com[0])
            shift_y, center_y = np.modf(com[1])
            
            center_x = int(center_x)
            center_y = int(center_y)
            
            big_psf_pad = np.pad(big_psf, ((krnl_range, krnl_range), (krnl_range, krnl_range)))
            
            ker_psf = big_psf_pad[center_x : center_x + (2*krnl_range)+1 , center_y : center_y + (2*krnl_range)+1]
            
            ker_psf = nim.shift(ker_psf, (shift_x, shift_y))
            
            file_new_path = os.path.join(directory_new_path, file)
            imaMan.save_exr_psf(ker_psf, file_new_path)


if __name__=='__main__':
    
    ker_size = 13
    camera_path = '/home/lor3n/Documents/GitHub/PFSrendering/psf/petzval/focus-5.00m/aperture-f1'
    camera_name = 'Petzval'
    kernelFilesGenerator(ker_size, camera_path, camera_name)