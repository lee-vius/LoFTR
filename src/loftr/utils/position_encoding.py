import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape)) # 这里pe的维度是(d_model, 256, 256)  注意这里的pe矩阵实际上不会改变，但逆推求导会使用其求导
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0) # 根据1矩阵沿着y方向做running sum构造坐标，相当于[1, 2, 3...]
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0) # 同上，根据1矩阵沿着x方向做running sum构造坐标
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2))) # 注意 // 表示整除，获得值为整数
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2)) # 这里的 div term的维度是 (d_model / 4)
        div_term = div_term[:, None, None]  # [C//4, 1, 1] 将维度拓展
        pe[0::4, :, :] = torch.sin(x_position * div_term) # 这里的实现有个特点 -- 会以4个feature为循环
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
