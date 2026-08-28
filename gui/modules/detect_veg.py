import os
import cv2
import numpy as np

def detect_veg(image,Min_veg_px):
    img_mask = create_green_mask(image,Min_veg_px)
    if img_mask is None:
        return []

    save_images_path = "img_mask.jpg"
    cv2.imwrite(save_images_path,img_mask)

    bboxes = get_bbox(img_mask)

    # save the raw image with bbox
    for bbox in bboxes:
        x,y,w,h = bbox
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imwrite("img_bbox.jpg", image)

    # sort the bboxes by lowest j0 coord
    bboxes_sorted = sorted(bboxes,reverse=True,key=lambda x : x[1]+ x[3])

    return bboxes_sorted
    
def create_green_mask(image,Min_veg_px):
    # yiq = rgb2yiq(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[..., 1]
    mask = cv2.inRange(a, 0, 120)

    if np.sum(mask == 255) > Min_veg_px:
        # otsu = Filter_Otsu(a)
        yiq = rgb2yiq(image)
        q = cv2.normalize(src=yiq[:,:,2], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask = cv2.inRange(q, 0, 120)
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    return None

def get_bbox(img_mask):
    bboxes = []
    blobs, _ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
    for blob in blobs:
        x,y,w,h = cv2.boundingRect(blob)     # (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        bboxes.append((x,y,w,h))

    return bboxes

# Converts an RGB image to YIQ color space
def rgb2yiq(im):
    rgb = im.astype(np.float32) / 255.0  # Normalize to [0, 1]
    yiq_matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.595716, -0.274453, -0.321263],
        [0.211456, -0.522591, 0.311135]
    ])

    return rgb @ yiq_matrix.T

if __name__ == '__main__':
    image = cv2.imread(r'C:\Users\james.y.kim\OneDrive - USDA\SRU\Research\Python\RPi\modules\test2.jpg')
    cv2.imshow('im',image), cv2.waitKey(0)
    Min_veg_px = 1000

    bboxes = detect_veg(image,Min_veg_px)    
    print('bboxes=',bboxes)

    