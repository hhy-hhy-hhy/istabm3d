# src/ista_denoiser.py
import numpy as np
import pywt
from typing import Tuple, Dict, List

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """软阈值Function"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def ista_denoise(
    noisy_image: np.ndarray,
    lambda_reg: float = 0.1,
    max_iter: int = 100,
    step_size: float = 1.0,
    wavelet: str = 'db4',
    level: int = None,  # 改为None，自动Calculate
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    ISTAAlgorithm进行ImageDenoising
    """
    # ProcessColorImage：如果是3Channel或4Channel，转换为Grayscalechart
    original_shape = noisy_image.shape
    is_color = len(original_shape) == 3
    
    if is_color:
        if original_shape[2] == 4:  # RGBA -> RGB
            noisy_image = noisy_image[:, :, :3]
            print(f"  ⚠️  将RGBAImage转换为RGB (Shape: {noisy_image.shape})")
        elif original_shape[2] == 3:  # RGB -> Grayscale
            # 使用加权平均转换为Grayscale
            noisy_image = 0.299 * noisy_image[:, :, 0] + \
                          0.587 * noisy_image[:, :, 1] + \
                          0.114 * noisy_image[:, :, 2]
            print(f"  ⚠️  将RGBImage转换为Grayscale (Shape: {noisy_image.shape})")
    
    # 自动Calculate合适的小波分解层数
    if level is None:
        # 根据ImageSizeCalculate最大可用层数
        min_dim = min(noisy_image.shape)
        level = int(np.floor(np.log2(min_dim))) - 2
        level = max(1, min(level, 3))  # 限制在1-3层
        if verbose:
            print(f"  自动Settings小波分解层数为: {level}")
    
    # Initialization
    x = noisy_image.copy()
    L = 1.0  # Lipschitz常数
    
    # 存储ConvergenceInfo
    cost_history = []
    
    if verbose:
        print(f"  🚀 开始ISTAIteration (λ={lambda_reg}, 层数={level}, 最大Iteration={max_iter})")
    
    for i in range(max_iter):
        x_prev = x.copy()
        
        # 梯度下降步
        gradient_step = x - (step_size / L) * (x - noisy_image)
        
        # 小波变换
        try:
            coeffs = pywt.wavedec2(gradient_step, wavelet, level=level)
        except ValueError as e:
            print(f"  小波变换Error: {e}")
            # 减少层数重试
            level = max(1, level - 1)
            coeffs = pywt.wavedec2(gradient_step, wavelet, level=level)
        
        # 软阈值Process
        coeffs_thresh = []
        for coeff in coeffs:
            if isinstance(coeff, tuple):
                coeff_thresh = tuple([
                    soft_threshold(c, lambda_reg * step_size / L) 
                    for c in coeff
                ])
                coeffs_thresh.append(coeff_thresh)
            else:
                coeffs_thresh.append(coeff)
        
        # 小波反变换
        try:
            x = pywt.waverec2(coeffs_thresh, wavelet)
        except ValueError as e:
            print(f"  小波反变换Error: {e}")
            # 如果Shape不匹配，裁剪到正确Size
            target_shape = gradient_step.shape
            if x.shape != target_shape:
                x = x[:target_shape[0], :target_shape[1]]
        
        # 确保ImageRange
        x = np.clip(x, 0, 1)
        
        # CalculateCostFunction
        data_fidelity = 0.5 * np.sum((x - noisy_image) ** 2)
        
        # Calculate小波系数的稀疏性惩罚
        try:
            coeffs_x = pywt.wavedec2(x, wavelet, level=level)
            sparsity = 0
            for coeff in coeffs_x:
                if isinstance(coeff, tuple):
                    for c in coeff:
                        sparsity += lambda_reg * np.sum(np.abs(c))
            cost = data_fidelity + sparsity
        except:
            cost = data_fidelity  # 如果CalculateFailed，只使用Data保真项
        
        cost_history.append(cost)
        
        # 检查Convergence
        error = np.linalg.norm(x - x_prev) / np.linalg.norm(x_prev + 1e-10)
        
        if verbose and i % 20 == 0:
            print(f"    Iteration {i:3d}, Cost: {cost:.6f}, 误差: {error:.6f}")
        
        if error < tol:
            if verbose:
                print(f"  ✅ 在第 {i} 次IterationConvergence")
            break
    
    # 将GrayscaleImage扩展回原始Shape（如果需要）
    if is_color and len(original_shape) == 3:
        # 如果是原始是Colorchart，但我们已经转换为Grayscale，现在复制到三个Channel
        x = np.stack([x, x, x], axis=-1)
    
    info = {
        'iterations': i + 1,
        'cost_history': cost_history,
        'final_cost': cost_history[-1] if cost_history else None,
        'converged': error < tol,
        'wavelet_level': level
    }
    
    return x, info

def ista_denoise_color(
    noisy_image: np.ndarray,
    lambda_reg: float = 0.1,
    max_iter: int = 100,
    step_size: float = 1.0,
    wavelet: str = 'db4',
    level: int = None,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    ColorImageISTADenoising - 分别Process每个Channel
    """
    if len(noisy_image.shape) != 3 or noisy_image.shape[2] not in [3, 4]:
        # 如果不是Colorchart，使用原始Function
        return ista_denoise(noisy_image, lambda_reg, max_iter, step_size, 
                           wavelet, level, tol, verbose)
    
    # ProcessRGBAImage
    if noisy_image.shape[2] == 4:
        noisy_image = noisy_image[:, :, :3]  # 丢弃alphaChannel
        print("  ⚠️  丢弃AlphaChannel，使用RGBChannel")
    
    # 分别Process每个Channel
    denoised_channels = []
    channel_infos = []
    
    for channel in range(3):
        if verbose:
            print(f"\n  ProcessChannel {channel+1}/3...")
        
        noisy_channel = noisy_image[:, :, channel]
        denoised_channel, info = ista_denoise(
            noisy_channel, lambda_reg, max_iter, step_size,
            wavelet, level, tol, verbose=False
        )
        
        denoised_channels.append(denoised_channel)
        channel_infos.append(info)
    
    # 合并Channel
    denoised = np.stack(denoised_channels, axis=-1)
    
    # Calculate平均Info
    avg_iterations = np.mean([info['iterations'] for info in channel_infos])
    avg_cost = np.mean([info['final_cost'] for info in channel_infos 
                       if info['final_cost'] is not None])
    
    combined_info = {
        'iterations': int(avg_iterations),
        'final_cost': avg_cost,
        'channel_infos': channel_infos,
        'wavelet_level': channel_infos[0]['wavelet_level'] if channel_infos else level
    }
    
    return denoised, combined_info

# TestingFunction
if __name__ == "__main__":
    print("TestingISTAAlgorithm...")
    
    # TestingGrayscaleImage
    print("\n1. TestingGrayscaleImage...")
    np.random.seed(42)
    gray_img = np.random.rand(128, 128)
    noisy_gray = gray_img + np.random.randn(128, 128) * 0.1
    noisy_gray = np.clip(noisy_gray, 0, 1)
    
    denoised_gray, info_gray = ista_denoise(noisy_gray, verbose=True)
    print(f"  GrayscaleImageDenoising完成，Iteration次数: {info_gray['iterations']}")
    
    # TestingColorImage
    print("\n2. TestingColorImage...")
    color_img = np.random.rand(64, 64, 3)
    noisy_color = color_img + np.random.randn(64, 64, 3) * 0.1
    noisy_color = np.clip(noisy_color, 0, 1)
    
    denoised_color, info_color = ista_denoise_color(noisy_color, verbose=True)
    print(f"  ColorImageDenoising完成，Iteration次数: {info_color['iterations']}")
    print(f"  OutputShape: {denoised_color.shape}")
