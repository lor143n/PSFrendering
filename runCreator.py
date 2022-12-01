

cameras = ['human-eye','canon-zoom','petzval','itoh-zoom','kreitzer-tele']

fstop = ['1.4','2.8','4.0','5.6','11']

file = open("run.sh", "a")
file2 = open("kerGen.sh", "a")

images = ['rocks', 'field', 'tree']
for image in images:
    for camera in cameras:
        for f in fstop:
            
            if f == '4.0' and camera != 'itoh-zoom':
                continue
            if f == '2.8' and (camera == 'human-eye' or camera == 'canon-zoom'):
                continue
            
            if camera == 'human-eye':
                file.write(f'python3 PSFConvolution_Trilinear_IDW.py {image} {camera} {f} 1.0\n')
                file2.write(f'python3 kernelGenerator.py {camera} 13 1.0 {f}\n')
                           
            file.write(f'python3 PSFConvolution_Trilinear_IDW.py {image} {camera} {f}\n')
            file2.write(f'python3 kernelGenerator.py {camera} 13 5.0 {f}\n')
file.close()