# src/utils.py
import numpy as np
import cv2
import os
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import List

def load_image(image_path, target_size=None):
    """LoadImage"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法ReadImage: {image_path}")
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32) / 255.0
    
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    return img

def save_image(image, save_path):
    """SaveImage"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if image.max() <= 1.0:
        image_to_save = (image * 255).astype(np.uint8)
    else:
        image_to_save = image.astype(np.uint8)
    
    if len(image_to_save.shape) == 3 and image_to_save.shape[2] == 3:
        image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, image_to_save)

def calculate_psnr(clean_img, denoised_img):
    """CalculatePSNR"""
    if clean_img.max() > 1.0:
        clean_img = clean_img / 255.0
    if denoised_img.max() > 1.0:
        denoised_img = denoised_img / 255.0
    
    return peak_signal_noise_ratio(clean_img, denoised_img, data_range=1.0)

def calculate_ssim(clean_img, denoised_img):
    """CalculateSSIM"""
    if clean_img.max() > 1.0:
        clean_img = clean_img / 255.0
    if denoised_img.max() > 1.0:
        denoised_img = denoised_img / 255.0
    
    if len(clean_img.shape) == 3:
        return structural_similarity(
            clean_img, denoised_img, 
            channel_axis=2, data_range=1.0
        )
    else:
        return structural_similarity(
            clean_img, denoised_img, data_range=1.0
        )

def add_gaussian_noise(image, sigma=0.1):
    """添加高斯Noise"""
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + noise
    return np.clip(noisy, 0, 1)

def add_salt_pepper_noise(image, prob=0.05):
    """添加椒盐Noise"""
    noisy = image.copy()
    
    salt_mask = np.random.rand(*image.shape) < prob/2
    pepper_mask = np.random.rand(*image.shape) < prob/2
    
    noisy[salt_mask] = 1
    noisy[pepper_mask] = 0
    
    return noisy

def get_image_files(folder_path, extensions=['.png', '.jpg', '.jpeg', '.bmp']):
    """获取File夹中的所有ImageFile"""
    image_files = []
    for file in os.listdir(folder_path):
        file_lower = file.lower()
        if any(file_lower.endswith(ext) for ext in extensions):
            image_files.append(os.path.join(folder_path, file))
    
    return sorted(image_files)

def timer(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        return result, elapsed_time
    return wrapper
