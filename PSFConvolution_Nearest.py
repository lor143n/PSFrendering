import math
from numba import njit, prange, float64
import numpy as np
from scipy import ndimage as nim
import imagesManager as imaMan
import time
import os
import sys


def load_psf_krnls(camera_path):
    psf_dict = []
    dict_index = 0
    for directory in os.listdir(camera_path):
        dir_clean = directory.split('-')
        dir_depth = float(dir_clean[1].split('m')[0])        
        print(dir_depth)
        psf_dict.append( (dir_depth, []) )
        
        dir_path = os.path.join(camera_path, directory)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            ker_psf = imaMan.load_psf(file_path)
            
            com0 = file.split('-')[0]
            com1 = file.split('-')[1]
            psf_dict[dict_index][1].append((float(com0), float(com1), ker_psf))
            
        dict_index += 1
            
    psf_dict.sort()    
    return psf_dict
  


@njit()
def psf_convolution(rgb, res, depth, krnls_db):
    
    rgb_new = res
    krnl_size = 13
    krnl_range = int((krnl_size - 1) / 2)
    
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in prange(krnl_range, image_width - krnl_range):
        print(i)
        for j in range(krnl_range, image_height - krnl_range):
            
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
    
            # KERNEL MAKING
    
            for h in range(krnl_size):
                for k in range(krnl_size):
                                              
                    #SELECTING DEPTH   
                    dep = float(round(depth[i-krnl_range+h][j-krnl_range+k]*10)/10)
                    chosen_dep_index = 0
                    
                    for dp_index in range(len(krnls_db)):
                        
                        if dp_index == len(krnls_db)-1:
                            chosen_dep_index = len(krnls_db)-1
                            break
                        
                        if dep >= krnls_db[dp_index][0] and dep <= krnls_db[dp_index+1][0]:
                             
                            diff1 = dep - krnls_db[dp_index][0]
                            diff2 = krnls_db[dp_index+1][0] - dep
                             
                            if diff1 <= diff2:
                                chosen_dep_index = dp_index
                            else:
                                chosen_dep_index = dp_index+1
                            
                            break
                    
                    pos_db = krnls_db[chosen_dep_index][1]
                    
                    #SELECTED DEPTH
                    #SELECTING POSITION
                    
                    min_distance = sys.maxsize
                    selected_krnl = pos_db[0][2]
                    
                    for psf in pos_db:
                        dist = math.sqrt((i-psf[0])**2 + (j-psf[1])**2)
                        if dist < min_distance:
                            min_distance = dist
                            selected_krnl = psf[2]
                    
                    
                    #POSITION SELECTED
                    
                    ijvalue = selected_krnl[h][k]
                    krnl[h*krnl_size + k] = ijvalue
                    kernelSum += ijvalue 
             
            #NORMALIZATION
            for elem in range(len(krnl)):
                krnl[elem] /= kernelSum
                
            # KERNEL MAKING ENDS
            
            #PIXEL COLVOLVE
            for x in prange(krnl_size):
                for y in range(krnl_size):
                    rgb_new[i-krnl_range][j-krnl_range] += krnl[x*krnl_size+y] * rgb[i-krnl_range+x][j-krnl_range+y]
            
                      
    return rgb_new
        
        
        
def convolution_init(ker_size, export_type, image_file, camera_path):
    print('Algorithm start')
    
    (rgb, depth) = imaMan.load_rgbd('TestImages/'+str(image_file)+'.exr')
    
    start_time = time.time()
    
    krnl_db = load_psf_krnls(camera_path)
    
    db_end_time = time.time()
    print("Database construction time is: ", str(db_end_time-start_time)+"s")
    print(len(krnl_db))
    start_time = time.time()
    
    krnl_range = int((ker_size - 1) / 2)
    rgb_res = rgb*0
    
    rgb = np.pad(rgb, ((krnl_range, krnl_range), (krnl_range, krnl_range), (0, 0)), mode='symmetric')
    depth = np.pad(depth, ((krnl_range, krnl_range), (krnl_range, krnl_range)), mode='symmetric')
    
    rgb_res = psf_convolution(rgb, rgb_res, depth, krnl_db)
    
    del krnl_db
    
    conv_end_time = time.time()
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    if export_type == '.exr':
        imaMan.save_exr(rgb_res , 'ResImages/'+str(image_file)+'['+str(ker_size)+']A[5.0m - 100mm - f1].exr')
    elif export_type == '.png':
        imaMan.save_srgb(rgb_res , 'ResImages/'+str(image_file)+'['+str(ker_size)+']A[5.0m - 100mm - f1].png')
    else:
        print("Save Error")
    
if __name__=='__main__':
    
    ker_size = 13
    export_type = '.png'
    image_file = 'tree1024_100'
    camera_path = '/home/lor3n/Documents/GitHub/PFSrendering/PSFkernels/Petzval_krnls'
    
    convolution_init(ker_size, export_type, image_file, camera_path)