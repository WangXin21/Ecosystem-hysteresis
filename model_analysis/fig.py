##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from S import ModelS
import matplotlib as mpl
##
plt.rcParams["axes.unicode_minus"] = False

# 生成3行2列的子图网格，figsize设置画布大小
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))  # (宽度, 高度)
# 调整子图间距（避免标签重叠）
plt.tight_layout(pad=3.0)  # pad控制子图间# Para K


model_set = {}
var = {}
k_interval = range(5, 11)
for k in k_interval:
    model_set[k] = ModelS(k=k)
    model_set[k].init()
    x, y = model_set[k].s()
    var[k] = [x, y]
mpl.rcParams['figure.dpi'] = 100
# plt.style.use('seaborn-dark')
for parameter in k_interval:
    axes[0, 0].plot(var[parameter][0], var[parameter][1], label='k='+str(parameter) )
axes[0, 0].set(xlim=(0, 800))
axes[0, 0].grid(linestyle='--', alpha=0.7)
axes[0, 0].legend()
axes[0, 0].set_ylabel('State $y$')
axes[0, 0].set_title('Para K  $(C_0=1,C_1=e)$')
axes[0, 0].text(
        0.05, 0.95,  # 子图内坐标（0,0左下角，1,1右上角）
        'A',
        fontsize=18,  # 字体大小，醒目
        fontweight='bold',  # 加粗
        color='black',  # 颜色
        ha='left', va='top',  # 对齐方式
        transform=axes[0, 0].transAxes  # 基于子图的归一化坐标
    )

# Para C_0
model_set = {}
var = {}
c_0_interval = np.linspace(1, 2, 5)
for c_0 in c_0_interval:
    model_set[c_0] = ModelS(c_0=c_0, k=9)
    model_set[c_0].init()
    x, y = model_set[c_0].s()
    var[c_0] = [x, y]

for parameter in c_0_interval:
    axes[1, 0].plot(var[parameter][0], var[parameter][1], label='$C_0$='+str(parameter) )
axes[1, 0].set(xlim=(0, 800))
axes[1, 0].grid(linestyle='--', alpha=0.7)
axes[1, 0].legend()
axes[1, 0].set_ylabel('State $y$')
axes[1, 0].set_title('Para $C_0   (K=9,C_1=e)$')
axes[1, 0].text(
        0.05, 0.95,  # 子图内坐标（0,0左下角，1,1右上角）
        'B',
        fontsize=18,  # 字体大小，醒目
        fontweight='bold',  # 加粗
        color='black',  # 颜色
        ha='left', va='top',  # 对齐方式
        transform=axes[1, 0].transAxes  # 基于子图的归一化坐标
    )

##
# Para C_1
model_set = {}
var = {}
c_1_interval = np.linspace(np.exp(1), np.exp(1)+2, 5)
for c_1 in c_1_interval:
    model_set[c_1] = ModelS(c_1=c_1, k=9)
    model_set[c_1].init()
    x, y = model_set[c_1].s()
    var[c_1] = [x, y]

for parameter in c_1_interval:
    axes[2, 0].plot(var[parameter][0], var[parameter][1], label='$C_1$='+str(parameter) )
axes[2, 0].set(xlim=(0, 800))
axes[2, 0].grid(linestyle='--', alpha=0.7)
axes[2, 0].legend()
axes[2, 0].set_xlabel('Driver $x$')
axes[2, 0].set_ylabel('State $y$')
axes[2, 0].set_title('Para $C_1  (k=9,C_0=1)$')
axes[2, 0].text(
        0.05, 0.95,  # 子图内坐标（0,0左下角，1,1右上角）
        'C',
        fontsize=18,  # 字体大小，醒目
        fontweight='bold',  # 加粗
        color='black',  # 颜色
        ha='left', va='top',  # 对齐方式
        transform=axes[2, 0].transAxes  # 基于子图的归一化坐标
    )


s = ModelS(k=6.33)
x, y = s.s()
axes[0, 1].plot(x, y, label='$ K=K^*$')
axes[0, 1].set(xlim=(0, 75))
axes[0, 1].grid(linestyle='--', alpha=0.7)
axes[0, 1].legend()
axes[0, 1].set_ylabel('State $y$')
axes[0, 1].set_title('$C_0=1, C_1=e$')
axes[0, 1].text(
        0.05, 0.95,  # 子图内坐标（0,0左下角，1,1右上角）
        'D',
        fontsize=18,  # 字体大小，醒目
        fontweight='bold',  # 加粗
        color='black',  # 颜色
        ha='left', va='top',  # 对齐方式
        transform=axes[0, 1].transAxes  # 基于子图的归一化坐标
    )

s = ModelS(k=4.33)
x, y = s.s()
axes[1, 1].plot(x, y, label='$ K<K^*$')
axes[1, 1].set(xlim=(0, 75))
axes[1, 1].grid(linestyle='--', alpha=0.7)
axes[1, 1].legend()
axes[1, 1].set_xlabel('Driver $x$')
axes[1, 1].set_ylabel('State $y$')
axes[1, 1].set_title('$C_0=1, C_1=e$')
axes[1, 1].text(
        0.05, 0.95,  # 子图内坐标（0,0左下角，1,1右上角）
        'E',
        fontsize=18,  # 字体大小，醒目
        fontweight='bold',  # 加粗
        color='black',  # 颜色
        ha='left', va='top',  # 对齐方式
        transform=axes[1, 1].transAxes  # 基于子图的归一化坐标
    )

axes[2, 1].set_visible(False)
plt.savefig('param_analysis.png', dpi=300)
plt.show()