import matplotlib.pyplot as plt
from matplotlib_venn import venn2

fig, axs = plt.subplots(1, 2, figsize=(14, 7))

for ax in axs:
    ax.set_axis_off()

venn1 = venn2(subsets=(50, 0, 50), set_labels=('Ruby Space (IFM)', 'Perfect Factorization (PFM)'),
subset_label_formatter=lambda x: '', ax=axs[0])

# axs[0].set_title("Original Problem: P ⊂ R")
axs[0].set_title("Original Problem: P ⊂ R", fontsize=24, fontweight='bold')

for subset in ('10', '01', '11'):
    patch = venn1.get_patch_by_id(subset)
    if patch:
        # 设置边框颜色与线宽
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    # 根据区域区分填充颜色
    if subset == '10':
        patch.set_facecolor('#FFCCCC')   # 浅红色
    elif subset == '11':
        patch.set_facecolor('#CCE5FF')   # 浅蓝色
    elif subset == '01':
        patch.set_facecolor('#CCFFCC')   # 浅绿色

if venn1.set_labels is not None:
    for label in venn1.set_labels:
        if label:
            label.set_fontsize(16)  # 设置集合标签字体大小

if venn1.subset_labels is not None:
    for label in venn1.subset_labels:
        if label:
            label.set_fontsize(14)  # 设置子集内数字或文字的字体大小

venn2_diagram = venn2(subsets=(0, 0, 50), set_labels=('Ruby Space (IFM)', '/Expanded PFM'),
subset_label_formatter=lambda x: '', ax=axs[1])
for subset in ('10', '01', '11'):
    label = venn2_diagram.get_label_by_id(subset)
if label:
    label.set_text('')

axs[1].set_title("Expanded Problem: \nEquivalence of Spaces", fontsize=24, fontweight='bold')

for subset in ('10', '01', '11'):
    patch = venn2_diagram.get_patch_by_id(subset)
    if patch:
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
        # 全部使用浅蓝色，加上半透明效果
        patch.set_facecolor('#CCE5FF')
        patch.set_alpha(0.8)


if venn2_diagram.set_labels is not None:
    for label in venn2_diagram.set_labels:
        if label:
            label.set_fontsize(16)

if venn2_diagram.subset_labels is not None:
    for label in venn2_diagram.subset_labels:
        if label:
            label.set_fontsize(14)

plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)
plt.tight_layout()
plt.savefig("test.png") 
plt.show()