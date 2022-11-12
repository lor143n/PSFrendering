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
  


@njit(parallel=True)
def psf_convolution(rgb, res, depth, krnls_db:list[tuple], interpolation_count):
    
    rgb_new = res
    krnl_size = 13
    krnl_range = int((krnl_size - 1) / 2)
    
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in prange(krnl_range, image_width - krnl_range):
        for j in range(krnl_range, image_height - krnl_range):
            
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
    
            #KERNEL GENERATION
            for h in range(krnl_size):
                for k in range(krnl_size):
                                              
                    #DEPTH SELECTION
                    '''START''' 
                      
                    dep = depth[i-krnl_range+h][j-krnl_range+k]
                    low_dep_index = 0
                    high_dep_index = 0
                    
                    for dp_index in range(len(krnls_db)):
                        
                        if dp_index == len(krnls_db)-1:
                            high_dep_index = len(krnls_db)-1
                            low_dep_index = high_dep_index-1
                            break
                        
                        if dep >= krnls_db[dp_index][0] and dep <= krnls_db[dp_index+1][0]:
                             
                            low_dep_index = dp_index
                            high_dep_index = dp_index+1
                            break
                    
                    low_pos_db = krnls_db[low_dep_index][1]
                    high_pos_db = krnls_db[high_dep_index][1]
                    depth_low = krnls_db[low_dep_index][0]
                    depth_high = krnls_db[low_dep_index][0]
                    
                    '''END'''
                    
                    #POSITION SELECTION - FOUR MINIMUM SEARCH
                    '''START'''
                    
                    low_dist_pos_db:list[tuple] = []
                    high_dist_pos_db:list[tuple] = []
                    
                    for elem in range(len(low_pos_db)):
                        dist_low = math.sqrt((i-low_pos_db[elem][0])**2 + (j-low_pos_db[elem][1])**2)
                        dist_high = math.sqrt((i-high_pos_db[elem][0])**2 + (j-high_pos_db[elem][1])**2)
                        
                        
                        if elem == 0:
                            low_dist_pos_db.append( (dist_low, low_pos_db[elem][2]) )
                            high_dist_pos_db.append( (dist_high, high_pos_db[elem][2]) )
                            
                        elif elem < interpolation_count:
                            done = False
                            for count in range(elem):
                                if dist_low < low_dist_pos_db[count][0]:
                                    low_dist_pos_db.insert(0, (dist_low, low_pos_db[elem][2]))
                                    done = True
                                    break
                            
                            if not done:
                                low_dist_pos_db.append((dist_low, low_pos_db[elem][2]))
                                
                            done = False
                            for count in range(elem):
                                if dist_high < high_dist_pos_db[count][0]:
                                    high_dist_pos_db.insert(0, (dist_high, high_pos_db[elem][2]))
                                    done = True
                                    break
                            
                            if not done:
                                high_dist_pos_db.append((dist_high, high_pos_db[elem][2]))
                                
                        else:
                            for count in range(interpolation_count):
                                if dist_low < low_dist_pos_db[count][0]:
                                    low_dist_pos_db.insert(0, (dist_low, low_pos_db[elem][2]))
                                    break
                            for count in range(interpolation_count):
                                if dist_high < high_dist_pos_db[count][0]:
                                    high_dist_pos_db.insert(0, (dist_high, high_pos_db[elem][2]))
                                    break
                                
                    '''END'''
                    
                    #POSITION INTERPOLATION - IDW 
                    '''START'''
                    
                    #1-depth depth 2D interpolation
                    
                    low_ijvalue = 0
                    inv_dist_sum = 0
                    
                    for count in range(interpolation_count):
                        psf_dist = low_dist_pos_db[count][0]
                        inv_dist_sum += 1 / psf_dist
                    
                    for count in range(interpolation_count):
                        psf_dist = low_dist_pos_db[count][0]
                        psf = low_dist_pos_db[count][1]
                        
                        w = (1/psf_dist) / inv_dist_sum
                        
                        low_ijvalue += psf[h][k] * w
                    
                    
                    #2-depth 2D interpolation
                    
                    high_ijvalue = 0
                    inv_dist_sum = 0
                    
                    for count in range(interpolation_count):
                        psf_dist = high_dist_pos_db[count][0]
                        inv_dist_sum += 1 / psf_dist
                    
                    for count in range(interpolation_count):
                        psf_dist = high_dist_pos_db[count][0]
                        psf = high_dist_pos_db[count][1]
                        
                        w = (1/psf_dist) / inv_dist_sum
                        
                        high_ijvalue += psf[h][k] * w
                        
                    #depth interpolation
                    
                    u = depth_high / (depth_high + depth_low)
                    
                    ijvalue = u * high_ijvalue  + (1-u) * low_ijvalue
                    
                    krnl[h*krnl_size + k] = ijvalue
                    kernelSum += ijvalue
                    
                    '''END'''
             
            #KERNEL NORMALIZATION
            for elem in range(len(krnl)):
                krnl[elem] /= kernelSum
            
            #PIXEL CONVOLUTION
            for x in range(krnl_size):
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
    
    interpolation_count = 4
    
    rgb_res = psf_convolution(rgb, rgb_res, depth, krnl_db, interpolation_count)
    
    del krnl_db
    
    conv_end_time = time.time()
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    if export_type == '.exr':
        imaMan.save_exr(rgb_res , 'ResImages/'+str(image_file)+'4['+str(ker_size)+'][5.0m - 100mm - f1].exr')
    elif export_type == '.png':
        imaMan.save_srgb(rgb_res , f'ResImages/{image_file}[{ker_size}comp3][5.0m - 100mm - f1][IDW - {interpolation_count}].png')
    else:
        print("Save Error")
    
if __name__=='__main__':
    
    ker_size = 13
    export_type = '.png'
    image_file = 'tree1024_100'
    camera_path = '/home/lor3n/Documents/GitHub/PFSrendering/PSFkernels/Petzval_krnls'
    
    convolution_init(ker_size, export_type, image_file, camera_path)