import os
import numpy as np
import cv2

def check_noise(img, threshold = 100):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    noise_level = np.mean(gradient_magnitude)

    is_noisy  = noise_level > threshold
    return is_noisy, noise_level

def check_brightness(img, dim = 10, bright_thresh = 0.9, dark_thresh = 0.4):
    img = cv2.resize(img, (dim,dim))
    L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2Lab))
    L = L/np.max(L)
    avg_brightness = np.mean(L)

    too_dark = avg_brightness < dark_thresh
    too_bright = avg_brightness > bright_thresh

    return too_dark, too_bright, avg_brightness

def check_blurring(img, threshold = 500):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_level = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    is_blur = blur_level < threshold
    return is_blur, blur_level

def check_contrast(img, threshold = 50.0):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray_img)

    is_low_contrast  = (threshold - contrast ) > 15
    is_high_contrast = contrast > threshold
    return is_low_contrast, is_high_contrast, contrast

def auto_denoising(img):
    # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.fastNlMeansDenoisingColored(img,None,3,8,7,21)

def auto_adjust_brightness(img: cv2, avg_brightness: float, state: int):
    # state = 1: too dark, 2: too bright
    if state == 1:
        target_brightness = 0.5
    elif state == 2: 
        target_brightness = 0.8
        
    ratio = avg_brightness / target_brightness
    adjusted_img = cv2.convertScaleAbs(img, alpha = 1 / ratio, beta = 0)

    return adjusted_img

def auto_sharpening(img: cv2, target_sharpness = 500):    
    kernel = np.array([[0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
    max_iter = 5
    for i in range(max_iter):
        _, sharpness = check_blurring(img)
        # print(f"Iteration {i+1}: Sharpness = {sharpness}")
        if sharpness >= target_sharpness:
            return img
        img = cv2.filter2D(img,-1,kernel)
    return img
    
def auto_adjust_contrast(img: cv2, current_contrast: float):
    target_contrast = 50.0
    if abs(current_contrast - target_contrast) >= 10:
        scale = 1.0
    elif abs(current_contrast - target_contrast) < 10:
        scale = 0.5
    
    adjustment_factor = min(1.0, max(0.0, scale * (target_contrast - current_contrast) / target_contrast))
    # print(adjustment_factor)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit= 4.0 * adjustment_factor, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    
    lab_eq = cv2.merge((l_eq, a, b))
    
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result

def auto_enhancing(img_path):
    cv2_img = cv2.imread(img_path)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    is_noise, noise_level = check_noise(cv2_img)
    if is_noise:
        cv2_img = auto_denoising(cv2_img)

    is_dark, is_bright, avg_brightness = check_brightness(cv2_img)
    if is_dark or is_bright:
        if is_dark:
            cv2_img = auto_adjust_brightness(cv2_img,avg_brightness,1)
        else:
            cv2_img = auto_adjust_brightness(cv2_img,avg_brightness,2)

    is_low_contrast, is_high_contrast,contrast_lvl = check_contrast(cv2_img)
    if is_low_contrast:
        cv2_img = auto_adjust_contrast(cv2_img, contrast_lvl)    

    
    is_blur, blurring_lvl = check_blurring(cv2_img)
    if is_blur:
        cv2_img = auto_sharpening(cv2_img)

    
    return cv2_img

    