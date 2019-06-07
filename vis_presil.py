import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
import os
import math

img_idx = 342

# Constants
rows = 1080 #image height
cols = 1920 #image width

# Default directory structure from PreSIL dataset
dataset_dir = os.path.expanduser('~') + '/GTAData/object/'
data_set = 'training'
depth_dir = os.path.join(dataset_dir, data_set) + '/depth'
stencil_dir = os.path.join(dataset_dir, data_set) + '/stencil'
file_path = depth_dir + '/{:06d}.bin'.format(img_idx)
stencil_file = stencil_dir + '/{:06d}.raw'.format(img_idx)

def ndcToDepth(ndc):
    nc_z = 0.15
    fc_z = 600
    fov_v = 59 #degrees
    nc_h = 2 * nc_z * math.tan(fov_v / 2.0)
    nc_w = 1920 / 1080.0 * nc_h

    depth = np.zeros((rows,cols))

    # Iterate through values
    # d_nc could be saved as it is identical for each computation
    # Then the rest of the calculations could be vectorized
    # TODO if need to use this frequently
    for j in range(0,rows):
        for i in range(0,cols):
            nc_x = abs(((2 * i) / (cols - 1.0)) - 1) * nc_w / 2.0
            nc_y = abs(((2 * j) / (rows - 1.0)) - 1) * nc_h / 2.0

            d_nc = math.sqrt(pow(nc_x,2) + pow(nc_y,2) + pow(nc_z,2))
            depth[j,i] = d_nc / (ndc[j,i] + (nc_z * d_nc / (2 * fc_z)))
            if ndc[j,i] == 0.0:
                depth[j,i] = fc_z

    return depth

# Load depth buffer
fd = open(file_path, 'rb')
f = np.fromfile(fd, dtype=np.float32,count=rows*cols)
im = f.reshape((rows, cols))

# Process depth buffer to linear format
depthIm = ndcToDepth(im)
depthIm = cv2.convertScaleAbs(depthIm) # Cap values at 255 for visualization
maxVal = depthIm.max()
cv2.imshow('Depth map (Press a Key to continue)', depthIm)
cv2.waitKey()
cv2.destroyAllWindows()

# Show a colorized version of the depth map
colmap = cm.jet(depthIm, bytes=True)
cv2.imshow('Depth map colorized (Press a Key to continue)', colmap)
cv2.waitKey()
cv2.destroyAllWindows()

# Read and process the stencil buffer
fd = open(stencil_file, 'rb')
f = np.fromfile(fd, dtype=np.uint8,count=rows*cols)
stencil = f.reshape((rows, cols)) #notice row, column format
stencil = 10 * stencil # Easier to visualize stencil buffer with this
stencil = cv2.convertScaleAbs(stencil) # Cap values at 255 for visualization
fd.close()

# Show the stencil buffer
cv2.imshow('Stencil Buffer (Press a Key to continue)', stencil)
cv2.waitKey()
cv2.destroyAllWindows()

# Save files to the Pictures directory
out_dir = os.path.expanduser('~') + '/Pictures/'
stencil_file = out_dir + '{}-stencil.png'.format(img_idx)
depth_color_file = out_dir + '{}-depth-color.png'.format(img_idx)
depth_file = out_dir + '{}-depth.png'.format(img_idx)

cv2.imwrite(depth_file, depthIm)
cv2.imwrite(depth_color_file, colmap)
cv2.imwrite(stencil_file, stencil)

print("Images saved to: ", out_dir)