import cv2
from Code.utils.format_conversion import binary2edge
import os

in_images = '../COVID-SemiSeg/Dataset/TrainingSet/LungInfection-Train/Pseudo-label/GT'
out_imags = './my_img'

for img_file in os.listdir(in_images):
    #print(img_file)
    edge_tmp = binary2edge(os.path.join(in_images, img_file))
    cv2.imwrite(os.path.join(out_imags, img_file), edge_tmp)