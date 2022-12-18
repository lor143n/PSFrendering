from numba import njit, prange
import imagesManager as imaMan
import numpy as np
import math
import click
import time
import os
import gc


def load_psf_krnls(camera_path):
    psf_dict = []
    dict_index = 0
    for directory in os.listdir(camera_path):
        dir_clean = directory.split('-')
        dir_depth = float(dir_clean[1].split('m')[0])       

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
  


@njit(parallel = True)
def psf_convolution(rgb, depth, krnl_size, krnls_db, interpolation_count):
    
    height = len(rgb)-krnl_size+1
    width = len(rgb[0])-krnl_size+1
    
    rgb_new = np.zeros((height, width, 3))
    krnl_range = int((krnl_size - 1) / 2)
    
    image_width = len(rgb)
    image_height = len(rgb[0])
    
    for i in prange(krnl_range, image_width - krnl_range):
        #print(i)
        for j in range(krnl_range, image_height - krnl_range):
            krnl = [0.0]*(krnl_size**2)
            kernelSum = 0
            
    
            #KERNEL GENERATION
            for h in range(krnl_size):
                for k in range(krnl_size):
                                              
                    #DEPTH SELECTION
                    '''START''' 
                    
                    low_dep_index = 0
                    high_dep_index = 0
                    dep = depth[i-krnl_range+h][j-krnl_range+k]
                    
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
                    
                    low_dist_pos_db = []
                
                    for elem in range(len(low_pos_db)):
                        dist_low = math.sqrt((i-low_pos_db[elem][0])**2 + (j-low_pos_db[elem][1])**2)
                        
                        if elem == 0:
                            low_dist_pos_db.append( (dist_low, low_pos_db[elem][2]) )
                            
                        elif elem < interpolation_count:
                            done = False
                            for count in range(elem):
                                if dist_low < low_dist_pos_db[count][0]:
                                    low_dist_pos_db.insert(0, (dist_low, low_pos_db[elem][2]))
                                    done = True
                                    break
                            
                            if not done:
                                low_dist_pos_db.append((dist_low, low_pos_db[elem][2]))
                                
                        else:
                            for count in range(interpolation_count):
                                if dist_low < low_dist_pos_db[count][0]:
                                    low_dist_pos_db.insert(0, (dist_low, low_pos_db[elem][2]))
                                    break
                    
                    high_dist_pos_db = []
                    
                    for elem in range(len(high_pos_db)):
                        dist_high = math.sqrt((i-high_pos_db[elem][0])**2 + (j-high_pos_db[elem][1])**2)
                        
                        if elem == 0:
                            high_dist_pos_db.append( (dist_high, high_pos_db[elem][2]) )
                            
                        elif elem < interpolation_count:
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
                                if dist_high < high_dist_pos_db[count][0]:
                                    high_dist_pos_db.insert(0, (dist_high, high_pos_db[elem][2]))
                                    break
                    '''END'''
                    
                    #POSITION INTERPOLATION - IDW 
                    '''START'''
                    
                    #1-depth depth 2D interpolation
                    
                    low_ijvalue = 0
                    inv_dist_sum = 0
                    
                    if len(low_dist_pos_db) < interpolation_count and len(low_dist_pos_db) != 1:
                        interpolation_count = len(low_dist_pos_db)
                    
                    if interpolation_count <= len(low_dist_pos_db) :
                        for count in range(interpolation_count):
                            psf_dist = low_dist_pos_db[count][0]
                            inv_dist_sum += 1 / psf_dist
                    
                        for count in range(interpolation_count):
                            psf_dist = low_dist_pos_db[count][0]
                            psf = low_dist_pos_db[count][1]
                        
                            w = (1/psf_dist) / inv_dist_sum
                        
                            low_ijvalue += psf[h][k] * w
                    else:
                        low_ijvalue = low_dist_pos_db[0][1][h][k]
                    
                    #2-depth 2D interpolation
                    
                    high_ijvalue = 0
                    inv_dist_sum = 0
                    
                    if len(high_dist_pos_db) < interpolation_count and len(high_dist_pos_db) != 1:
                        interpolation_count = len(high_dist_pos_db)
                    
                    if interpolation_count <= len(high_dist_pos_db):
                        for count in range(interpolation_count):
                            psf_dist = high_dist_pos_db[count][0]
                            inv_dist_sum += 1 / psf_dist
                    
                        for count in range(interpolation_count):
                            psf_dist = high_dist_pos_db[count][0]
                            psf = high_dist_pos_db[count][1]
                        
                            w = (1/psf_dist) / inv_dist_sum
                        
                            high_ijvalue += psf[h][k] * w
                    else:
                        high_ijvalue = high_dist_pos_db[0][1][h][k]
                        
                    #depth interpolation
                    
                    u = depth_high / (depth_high + depth_low)
                    
                    ijvalue = u * high_ijvalue  + (1-u) * low_ijvalue
                    
                    krnl[h*krnl_size + k] = ijvalue
                    kernelSum += ijvalue
   
             
             
            #KERNEL NORMALIZATION
            for elem in range(len(krnl)):
                krnl[elem] /= kernelSum
            
            #PIXEL CONVOLUTION
            for x in range(krnl_size):
                for y in range(krnl_size):
                           
                    rgb_value = rgb[i-krnl_range+x][j-krnl_range+y]
                    rgb_new[i-krnl_range][j-krnl_range] += krnl[x*krnl_size+y] * rgb_value

                    
    
    return rgb_new
        

@click.command()
@click.argument('image_file', default='rocks')
@click.argument('camera_type', default='canon-zoom')
@click.argument('aperture', default=1.4)
@click.argument('focus', default=5.0)
@click.argument('export_type', default='.png')
@click.argument('interpolation_steps', default=4)
@click.argument('krnl_size', default=15)
def convolution_init(image_file, camera_type, export_type, krnl_size, interpolation_steps, focus, aperture):
    
    # Loading image
    start_time = time.time()
    
    camera_path = f'PSF_kernels/{camera_type}_{krnl_size}_{focus}_{aperture}'
    
    (rgb, depth) = imaMan.load_rgbd(f'test/{image_file}_{camera_type}.exr')
    
    db_end_time = time.time()
    print("Image loading took ", str(db_end_time-start_time)+"s")
    
    #Lens PSFs loading
    start_time = time.time()
    
    krnl_db = load_psf_krnls(camera_path)

    db_end_time = time.time()
    print("Lens data loading took ", str(db_end_time-start_time)+"s")
    
    #Convolution
    start_time = time.time()
    
    krnl_range = int((krnl_size - 1) / 2)
    
    rgb = np.pad(rgb, ((krnl_range, krnl_range), (krnl_range, krnl_range), (0, 0)), mode='symmetric')
    depth = np.pad(depth, ((krnl_range, krnl_range), (krnl_range, krnl_range)), mode='symmetric')
    
    rgb_res = psf_convolution(rgb, depth, krnl_size, krnl_db, interpolation_steps)
    
    del krnl_db
    
    conv_end_time = time.time()
    print("Convolution time is: ", str(conv_end_time-start_time)+"s")
    
    
    #Exporting
    if export_type == '.exr':
        imaMan.save_exr(rgb_res , f'results/{image_file}_IDW_{camera_type}_{krnl_size}_{focus}_{aperture}_{interpolation_steps}.exr')
    elif export_type == '.png':
        imaMan.save_srgb(rgb_res , f'results/{image_file}_IDW_{camera_type}_{krnl_size}_{focus}_{aperture}_{interpolation_steps}.png')
    else:
        print("Save Error")
        
    print('Done!')
        

if __name__=='__main__':
    convolution_init()

    
