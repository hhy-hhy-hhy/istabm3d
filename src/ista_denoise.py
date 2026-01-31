# src/ista_denoise.py
"""
ISTA（迭代收缩阈值算法）图像去噪实现
"""

import numpy as np
import pywt
from .utils import soft_threshold

class ISTADenoise:
    """ISTA 图像去噪类"""
    
    def __init__(self, wavelet='haar', level=3, lam=0.1, max_iter=100, tol=1e-6):
        """
        初始化 ISTA 去噪器
        
        参数:
            wavelet: 小波基类型
            level: 小波分解层数
            lam: 正则化参数
            max_iter: 最大迭代次数
            tol: 收敛容忍度
        """
        self.wavelet = wavelet
        self.level = level
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        
    def denoise(self, noisy_image):
        """
        使用 ISTA 进行图像去噪
        
        参数:
            noisy_image: 含噪声图像
            
        返回:
            denoised_image: 去噪后的图像
        """
        # 小波变换
        coeffs = pywt.wavedec2(noisy_image, self.wavelet, level=self.level)
        
        # 对每一层细节系数进行 ISTA 去噪
        denoised_coeffs = []
        denoised_coeffs.append(coeffs[0])  # 近似系数保持不变
        
        for i in range(1, len(coeffs)):
            # 当前层的细节系数
            detail_coeffs = coeffs[i]
            
            # 对每个方向的系数进行 ISTA
            denoised_detail = []
            for coeff in detail_coeffs:
                denoised_coeff = self._ista_single_channel(coeff)
                denoised_detail.append(denoised_coeff)
            
            denoised_coeffs.append(tuple(denoised_detail))
        
        # 逆小波变换
        denoised_image = pywt.waverec2(denoised_coeffs, self.wavelet)
        
        # 确保图像范围在 [0, 1]
        denoised_image = np.clip(denoised_image, 0, 1)
        
        return denoised_image
    
    def _ista_single_channel(self, coeff):
        """对单通道系数进行 ISTA 去噪"""
        # 初始化
        x = coeff.copy()
        
        # 计算步长（Lipschitz 常数的倒数）
        alpha = 1.0
        
        # ISTA 迭代
        for i in range(self.max_iter):
            x_old = x.copy()
            
            # 梯度下降步
            gradient = x
            x = x - alpha * gradient
            
            # 软阈值收缩
            x = soft_threshold(x, self.lam * alpha)
            
            # 检查收敛
            if np.linalg.norm(x - x_old) < self.tol:
                break
        
        return x
