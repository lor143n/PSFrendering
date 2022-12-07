

cameras = ['human-eye','canon-zoom','petzval','itoh-zoom','kreitzer-tele']

fstop = ['1.4','2.8','4.0','5.6','8','11']

file = open("run.sh", "a")

images = ['rocks', 'field', 'tree']
for image in images:
    for camera in cameras:
        for f in fstop:
            
            if (f == '4.0' or f == '8' or f == '2.8') and (camera != 'kreitzer-tele' and camera != 'human-eye'):
                continue
                
            if (camera == 'human-eye') and (fstop == '1.4' or fstop == '2.8' or fstop == '5.6') :
                continue
                
            if (camera == 'kreitzer-tele') and (fstop == '5.6' or fstop == '8' or fstop == '11') :
                continue
                           
            file.write(f'python3 PSFConvolution_Trilinear_IDW.py {image} {camera} {f}\n')
file.close()
