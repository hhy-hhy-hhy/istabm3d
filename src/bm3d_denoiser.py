# src/bm3d_denoiser.py
import numpy as np

try:
    import bm3d
    BM3D_AVAILABLE = True
except ImportError:
    BM3D_AVAILABLE = False
    print("⚠️ BM3D库未安装，使用OpenCV替代方案")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def bm3d_denoise(noisy_image, sigma, color_mode='gray'):
    """BM3DImageDenoising"""
    if BM3D_AVAILABLE:
        # 使用官方BM3D库
        if len(noisy_image.shape) == 2 or color_mode == 'gray':
            return bm3d.bm3d(noisy_image, sigma_psd=sigma)
        else:
            return bm3d.bm3d(noisy_image, sigma_psd=sigma, color_denoise=True)
    
    elif OPENCV_AVAILABLE:
        # 使用OpenCV作为替代
        if noisy_image.max() <= 1.0:
            noisy_uint8 = (noisy_image * 255).astype(np.uint8)
        else:
            noisy_uint8 = noisy_image.astype(np.uint8)
        
        if len(noisy_uint8.shape) == 2:
            denoised = cv2.fastNlMeansDenoising(
                noisy_uint8, h=sigma*20, 
                templateWindowSize=7, searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoisingColored(
                noisy_uint8, h=sigma*20, hColor=sigma*10,
                templateWindowSize=7, searchWindowSize=21
            )
        
        return denoised.astype(np.float32) / 255.0
    
    else:
        raise ImportError("既没有安装bm3d也没有安装opencv-python")

if __name__ == "__main__":
    # Testing
    np.random.seed(42)
    img = np.random.rand(64, 64)
    result = bm3d_denoise(img, sigma=0.1)
    print(f"BM3DTesting完成，OutputShape: {result.shape}")
