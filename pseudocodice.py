
Input: RGBD image [image_in], PSFs dataset [PSFs], kernel_dimension [k]
Output: RGB image [image_out]


def psf_convolution:
    
    Pol_PSFs = PFSs_Polishing(PSFs)
    
    image_out.size = image_in.size
    
    for i in image_in.width:
        for j in image_in.hieght:
        
            pixel = image_in(i,j)
    
            kernel = Kernel_Building(pixel, image_in, k, Pol_PSFs)
            
            #PIXEL CONVOLUTION
            for x in range(kernel.size):
                for y in range(kernel.size):
                           
                    image_out(i,j) += kernel(x,y) * image_in(i-kernel.size+x,j-kernel.size+y)
                    
                    
                    

Input: Kernel dimension [k],  Pixel coordinates [(i,j)], RGBD image [image_in], Polished PSFs dataset [Pol_PSFs]
Output: k-dimensional squared matrix [kernel]

def Kernel_Building

    kernel.size = k

    for x in range(k):
        for y in range(k):
                                              
            #DEPTHS SELECTION   
                    
            minor_depth, major_depth = Depth_Selection( image_in(i-k+x, j-k+y).Depth )               
  
            minor_depth_PSFs = Pol_PSFs(minor_depth)
            major_depth_PSFs = Pol_PSFs(major_depth)
                    
                    
                    
            #POSITIONS SELECTIONS
            #The positions selections must done for each pixel and depth because positions are different evry time
                    
            minor_depth_selected_PSFs = K-NN(minor_depth_PSFs, (i,j))
            major_depth_selected_PSFs = K-NN(major_depth_PSFs, (i,j))
                    
                    
                    
            #PSFs INTERPOLATION BY POSITIONS
                    
            minor_depth_interpolated_PSF = IDW(minor_depth_selected_PSFs)
            major_depth_interpolated_PSF = IDW(major_depth_selected_PSFs)
                     
                    
            #PSFs INTERPOLATION BY DEPTH
                    
            final_PSF = Linear_Interpolation(minor_depth_interpolated_PSF, major_depth_interpolated_PSF)
                    
            kernel(x,y) = final_PSF(x,y)
   
                  
    #KERNEL NORMALIZATION
    
    kernel /= Elements_Sum(kernel)
                    
    
    return kernel




Input: PSFs dataset [PSFs], PSFs final resolution [k]
Output: Polished PSFs dataset [Pol_PSFs]

def PSFs_ Polishing:
    
    Pol_PSFs.size = PSFs.size
    
    for PSF in PSFs:
        
        Center_of_mass = Compute_CenterOfMass(PSF)
        
        PSF.cut(Center_of_mass, k)
        
        PSF.shift()
        
        Pol_PSFs.add(PSF)
    