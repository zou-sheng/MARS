import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 数据预处理（确保数据与表格完全一致）
data = {
    "PE列数": [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
    "EDP": [7.62E+04, 7.66E+04, 7.24E+04, 6.84E+04, 6.44E+04, 6.06E+04, 
            5.68E+04, 5.71E+04, 5.34E+04, 4.98E+04, 5.00E+04, 4.65E+04],
    "GFLOPs": [240.82, 248.35, 255.39, 262.38, 269.29, 276.13, 282.9, 
               290.34, 297.0, 303.56, 310.96, 317.38],
    "Area": [1.28, 1.3, 1.31, 1.32, 1.33, 1.35, 1.36, 1.37, 1.38, 1.39, 1.41, 1.42],
    "Cycles": [8912896, 8912896, 8650752, 8388608, 8126465, 7864320, 
               7602176, 7602176, 7340033, 7077889, 7077889, 6815745],
    "Energy": [8550.84, 8591.45, 8372.3, 8150.65, 7926.51, 7699.93, 
               7470.93, 7504.6, 7271.84, 7036.71, 7067.67, 6828.86]
}
df = pd.DataFrame(data)

# 2. 设置中文字体（避免中文乱码，适配Ubuntu系统）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']  # 兼容不同系统
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 3. 创建画布（主图+3个子图，调整尺寸适配所有内容）
fig = plt.figure(figsize=(16, 12))

# ---------------------- 主图：双Y轴（EDP + GFLOPs）----------------------
ax1 = plt.subplot(2, 2, 1)
# 修复：颜色用缩写（b=蓝色），格式字符串用 'o'（圆点标记），分开设置
line1 = ax1.plot(df["PE列数"], df["EDP"], color='b', marker='o', linestyle='-', 
                 label="EDP", linewidth=2, markersize=6)
ax1.set_xlabel("PE列数（32×N）", fontsize=12)
ax1.set_ylabel("EDP（×10⁴）", color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title("核心指标趋势（EDP vs GFLOPs）", fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)  # 添加网格，提升可读性
ax1.legend(loc='upper right')

# 右侧Y轴（GFLOPs）
ax2 = ax1.twinx()
# 修复：颜色用缩写（r=红色），格式字符串用 's'（方形标记）
line2 = ax2.plot(df["PE列数"], df["GFLOPs"], color='r', marker='s', linestyle='-', 
                 label="GFLOPs", linewidth=2, markersize=6)
ax2.set_ylabel("GFLOPs", color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')
ax2.legend(loc='lower right')

# ---------------------- 子图1：Area变化趋势 ----------------------
ax3 = plt.subplot(2, 2, 2)
ax3.plot(df["PE列数"], df["Area"], color='g', marker='^', linestyle='-', 
         linewidth=2, markersize=6)
ax3.set_xlabel("PE列数（32×N）", fontsize=12)
ax3.set_ylabel("Area", fontsize=12)
ax3.set_title("面积变化趋势", fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# ---------------------- 子图2：Cycles变化趋势（单位：百万周期）----------------------
ax4 = plt.subplot(2, 2, 3)
# 修复：颜色用 'orange'（完整颜色名），单独设置marker，数值除以1e6转为百万级
ax4.plot(df["PE列数"], df["Cycles"]/1e6, color='orange', marker='d', linestyle='-', 
         linewidth=2, markersize=6)
ax4.set_xlabel("PE列数（32×N）", fontsize=12)
ax4.set_ylabel("Cycles（百万）", fontsize=12)
ax4.set_title("延迟周期变化趋势", fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# ---------------------- 子图3：Energy变化趋势 ----------------------
ax5 = plt.subplot(2, 2, 4)
ax5.plot(df["PE列数"], df["Energy"], color='purple', marker='v', linestyle='-', 
         linewidth=2, markersize=6)
ax5.set_xlabel("PE列数（32×N）", fontsize=12)
ax5.set_ylabel("Energy", fontsize=12)
ax5.set_title("能耗变化趋势", fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 调整布局，避免标签重叠
plt.tight_layout(pad=3.0)

# 保存高分辨率图片（可选，支持PNG/PDF格式）
plt.savefig("pe_performance_analysis.png", dpi=300, bbox_inches='tight')
plt.savefig("pe_performance_analysis.pdf", dpi=300, bbox_inches='tight')

# 显示图片
plt.show()
