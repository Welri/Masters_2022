import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def upscale_image(filename,vertical,horizontal,show_images = False):
    # Original image read in
    img = cv2.imread(filename,1)
    shape = img.shape
    print("Shape old: ",shape)

    # Image scale up
    img_scale_up = cv2.resize(img, (0, 0), fx=horizontal/shape[1], fy=vertical/shape[0])
    shape = img_scale_up.shape
    print("Shape new: ",shape)
    
    # Save the upscaled image
    file = Path(filename).stem
    new_filename = "Example_environments/" + file + "_upscaled.png"
    cv2.imwrite(new_filename,img_scale_up)

    # Show both images - 0 key to exit
    if show_images == True:
        cv2.imshow('Original', img)
        cv2.imshow('Upscaled Image', img_scale_up)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# # MOUNTAINOUS_LOW
# filename = "Example_environments/spitskop_sat.png"
# horizontal = 15312 # m
# vertical = 7606 # m

# MOUNTAINOUS HIGH
# filename = "Example_environments/MC_obs_black.png"
# horizontal = 11496 # m
# vertical = 5468 #

# GROUND
# filename = "Example_environments/Aberdeen_obstacles.png"
# horizontal = 15849 # m
# vertical = 7555 # m

# MARINE
filename = "Example_environments/Jbay_obs.png"
horizontal = 3347 # m
vertical = 1594 #

upscale_image(filename,vertical,horizontal,show_images=False)

# im = image.imread("test.png")
# shape = np.shape(im)
# im2 = np.zeros(shape)
# rows = len(im)
# for i in range(rows):
#     im2[i] = im[rows-i-1]
# fig,ax = plt.subplots()


# plt.imshow(im2)
# ax.invert_yaxis()
# plt.show()

# fig.figimage(im,alpha=0.5)