# Uncertainty .raw 流文件格式说明

## 概述

`.raw` 文件是 `run.py --uncertainty` 输出的像素级不确定性数据流文件，包含模型对每个像素预测结果的方差估计（越高表示越不确定）。

## 二进制格式

| 字段 | 类型 | 说明 |
|------|------|------|
| 数据体 | `float32` 数组 | 连续存储，**无文件头、无对齐填充** |

### 排列顺序

**C-order (row-major)**: `[H, W, 3]`

```
像素 (0,0) R通道方差 | 像素 (0,0) G通道方差 | 像素 (0,0) B通道方差
像素 (0,1) R通道方差 | 像素 (0,1) G通道方差 | 像素 (0,1) B通道方差
...
像素 (H-1,W-1) R通道方差 | 像素 (H-1,W-1) G通道方差 | 像素 (H-1,W-1) B通道方差
```

### 文件大小

```
文件字节数 = 宽度 × 高度 × 3 × 4
```

示例：1920×1080 图像 → 1920 × 1080 × 3 × 4 = **24,883,200 bytes ≈ 23.7 MB**

### 与其他文件的关系

| 文件 | 内容 | 坐标系 |
|------|------|--------|
| `*_uncertainty.jpg` | 三通道方差均值经 JET 伪彩映射后的可视化图 | 与输入图像尺寸一致 |
| `*_uncertainty.raw` | 三通道原始 float32 方差值 | 与输入图像尺寸一致 |

`.jpg` 是给人看的，`.raw` 是给程序用的。

---

## 读取示例

### Python

```python
import numpy as np

def read_uncertainty_raw(path, height, width):
    """读取 .raw 文件为 (H,W,3) float32 数组."""
    raw = np.fromfile(path, dtype=np.float32)
    return raw.reshape(height, width, 3)

# 使用
var_map = read_uncertainty_raw('output_uncertainty.raw', 1080, 1920)
# var_map[h, w, c] 为像素 (h,w) 在通道 c 上的预测方差

# 提取灰度方差（三通道平均）
gray_var = np.mean(var_map, axis=2)

# 提取最大方差通道（取每个像素最不确定的通道）
max_var = np.max(var_map, axis=2)
```

### C++

```cpp
#include <fstream>
#include <vector>

struct UncertaintyRaw {
    int H, W, C = 3;
    std::vector<float> data;  // size = H * W * 3

    static UncertaintyRaw load(const char* path, int height, int width) {
        UncertaintyRaw raw;
        raw.H = height;
        raw.W = width;
        raw.data.resize(height * width * 3);
        std::ifstream file(path, std::ios::binary);
        file.read(reinterpret_cast<char*>(raw.data.data()),
                  height * width * 3 * sizeof(float));
        return raw;
    }

    float at(int h, int w, int c) const {
        return data[(h * W + w) * 3 + c];
    }
};
```

### MATLAB

```matlab
function var_map = read_uncertainty_raw(path, height, width)
    fid = fopen(path, 'rb');
    var_map = fread(fid, [width * 3, height], 'float32');
    var_map = reshape(var_map, [width, 3, height]);
    var_map = permute(var_map, [3, 1, 2]);  % → (H, W, 3)
    fclose(fid);
end
```

---

## 数据含义

每个像素的方差值越大，表示模型在该位置的预测越不稳定（不确定性越高）。

| 典型场景 | 方差值趋势 | 含义 |
|----------|-----------|------|
| 平坦区域（天空、墙面） | 低 | 模型有把握，预测一致 |
| 边缘/纹理区域 | 中 | 模型对细粒度恢复有一定不确定性 |
| 噪声突变/异常区域 | 高 | 模型不确定，需人工关注 |
| 图像边界 | 中-高 | 边界填充导致的信息缺失 |

方差值未归一化（绝对尺度），建议在应用层做 min-max 或分位数归一化后使用。
