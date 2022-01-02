import os
import cv2

path = './dataset/val/'            # dataset/val
images_path = './build/val/images/'# build/val
if not os.path.isdir(images_path): # image_path = build/val 
    os.makedirs(images_path)
labels_path = './build/val/labels' # labels_path = build/val
if not os.path.isdir(labels_path):
    os.makedirs(labels_path)
lister = os.listdir(path)          # lister = dataset/val

if __name__ == '__main__':
    for i in lister:               # i in dataset/val
        if i.endswith('.jpg'):
            name = i.split('.')[0] + '.png'
            image = cv2.imread(os.path.join(path,i),0) # path = dataset/val
            cv2.imwrite(os.path.join(images_path,name), image,[int(cv2.IMWRITE_PNG_COMPRESSION),0])
            # image_path = build/val image
        elif i.endswith('.png'):
            name = i.split('-')[0] + '.png'
            label = cv2.imread(os.path.join(path,i),0) # path = dataset/val
            cv2.imwrite(os.path.join(labels_path, name), label, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            # labels_path  label




