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
def psf_convolution(rgb, res, depth, krnls_db:list[tuple], interpolation_count):
    
    rgb_new = res
    krnl_size = 13
    krnl_range = int((krnl_size - 1) / 2)
    
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in range(krnl_range, image_width - krnl_range):
        print(i)
        for j in range(krnl_range, image_height - krnl_range):
            
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
    
            # KERNEL MAKING
    
            for h in range(krnl_size):
                for k in range(krnl_size):
                                              
                    #SELECTING DEPTH   
                    dep = float(round(depth[i-krnl_range+h][j-krnl_range+k]*10)/10)
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
                    
                    
                    low_dist_pos_db:list[tuple] = []
                    high_dist_pos_db:list[tuple] = []
                    
                    for elem in range(len(low_pos_db)):
                        dist_low = math.sqrt((i-low_pos_db[elem][0])**2 + (j-low_pos_db[elem][1])**2)
                        dist_high = math.sqrt((i-high_pos_db[elem][0])**2 + (j-high_pos_db[elem][1])**2)
                        
                        
                        if elem == 0:
                            low_dist_pos_db.append( ([low_pos_db[elem][0], low_pos_db[elem][1]], dist_low, low_pos_db[elem][2]) )
                            high_dist_pos_db.append( ([high_pos_db[elem][0], high_pos_db[elem][1]],dist_high, high_pos_db[elem][2]) )
                            
                        elif elem < interpolation_count:
                            done = False
                            for count in range(elem):
                                if dist_low < low_dist_pos_db[count][1]:
                                    low_dist_pos_db.insert(0, ([low_pos_db[elem][0], low_pos_db[elem][1]], dist_low, low_pos_db[elem][2]))
                                    done = True
                                    break
                            
                            if not done:
                                low_dist_pos_db.append(([low_pos_db[elem][0], low_pos_db[elem][1]], dist_low, low_pos_db[elem][2]))
                                
                            done = False
                            for count in range(elem):
                                if dist_high < high_dist_pos_db[count][1]:
                                    high_dist_pos_db.insert(0, ([high_pos_db[elem][0], high_pos_db[elem][1]], dist_high, high_pos_db[elem][2]))
                                    done = True
                                    break
                            
                            if not done:
                                high_dist_pos_db.append(([high_pos_db[elem][0], high_pos_db[elem][1]], dist_high, high_pos_db[elem][2]))
                                
                        else:
                            for count in range(interpolation_count):
                                if dist_low < low_dist_pos_db[count][1]:
                                    low_dist_pos_db.insert(0, ([low_pos_db[elem][0], low_pos_db[elem][1]], dist_low, low_pos_db[elem][2]))
                                    break
                            for count in range(interpolation_count):
                                if dist_high < high_dist_pos_db[count][1]:
                                    high_dist_pos_db.insert(0, ([high_pos_db[elem][0], high_pos_db[elem][1]], dist_high, high_pos_db[elem][2]))
                                    break
                    
                    #POSITION SELECTED
                    
                    #low
                    
                    low_ijvalue = 0
                    
                    p = [i,j]
                    a = low_dist_pos_db[0][0]
                    b = low_dist_pos_db[1][0]
                    c = low_dist_pos_db[2][0]
                    
                    v0 = [b[0] - a[0], b[1] - a[1]] 
                    v1 = [c[0] - a[0], c[1] - a[1]]  
                    v2 = [p[0] - a[0], p[1] - a[1]] 
                    d00 = v0[0]*v0[0] + v0[1]*v0[1]
                    d01 = v0[0]*v1[0] + v0[1]*v1[1]
                    d11 = v1[0]*v1[0] + v1[1]*v1[1]
                    d20 = v2[0]*v0[0] + v2[1]*v0[1]
                    d21 = v2[0]*v1[0] + v2[1]*v1[1]
                    denom = d00 * d11 - d01 * d01
                    v = abs((d11 * d20 - d01 * d21) / denom)
                    w = abs((d00 * d21 - d01 * d20) / denom)
                    u = abs(1.0 - v - w)
                    
                    
                    psf1 = low_dist_pos_db[0][2]
                    psf2 = low_dist_pos_db[1][2]
                    psf3 = low_dist_pos_db[2][2]
                    
                    low_ijvalue = psf1[h][k]*u + psf2[h][k]*v + psf3[h][k]*w
                  
                    #HIGH
                    
                    high_ijvalue = 0
                    
                    p = [i,j]
                    a = high_dist_pos_db[2][0]
                    b = high_dist_pos_db[1][0]
                    c = high_dist_pos_db[0][0]
                    
                    v0 = [b[0] - a[0], b[1] - a[1]] 
                    v1 = [c[0] - a[0], c[1] - a[1]]  
                    v2 = [p[0] - a[0], p[1] - a[1]] 
                    d00 = v0[0]*v0[0] + v0[1]*v0[1]
                    d01 = v0[0]*v1[0] + v0[1]*v1[1]
                    d11 = v1[0]*v1[0] + v1[1]*v1[1]
                    d20 = v2[0]*v0[0] + v2[1]*v0[1]
                    d21 = v2[0]*v1[0] + v2[1]*v1[1]
                    denom = d00 * d11 - d01 * d01
                    v = (d11 * d20 - d01 * d21) / denom
                    w = (d00 * d21 - d01 * d20) / denom
                    u = 1.0 - v - w
                    
                    psf1 = high_dist_pos_db[0][2]
                    psf2 = high_dist_pos_db[1][2]
                    psf3 = high_dist_pos_db[2][2]
                    
                    high_ijvalue = psf1[h][k]*u + psf2[h][k]*v + psf3[h][k]*w 
                    
                    
                    #interpolation depth
                    
                    u = depth_high / (depth_high + depth_low)
                    
                    ijvalue = u * high_ijvalue  + (1-u) * low_ijvalue
                    
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
    
    rgb_res = psf_convolution(rgb, rgb_res, depth, krnl_db, 3)
    
    del krnl_db
    
    conv_end_time = time.time()
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    if export_type == '.exr':
        imaMan.save_exr(rgb_res , 'ResImages/'+str(image_file)+'['+str(ker_size)+'][5.0m - 100mm - f1].exr')
    elif export_type == '.png':
        imaMan.save_srgb(rgb_res , 'ResImages/'+str(image_file)+'['+str(ker_size)+'][5.0m - 100mm - f1].png')
    else:
        print("Save Error")
    
if __name__=='__main__':
    
    ker_size = 13
    export_type = '.png'
    image_file = 'tree1024_100'
    camera_path = '/home/lor3n/Documents/GitHub/PFSrendering/PSFkernels/Petzval_krnls'
    
    convolution_init(ker_size, export_type, image_file, camera_path)