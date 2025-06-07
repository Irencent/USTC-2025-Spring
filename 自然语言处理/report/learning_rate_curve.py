import numpy as np
import matplotlib.pyplot as plt

# 参数设置
lr_max = 2.5e-4
warmup_steps = 2000
total_steps = 200000  # 假设训练总步数为20万步（根据论文推测，可调整）

# 学习率调度函数
def get_lr(step):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_max * 0.5 * (1 + np.cos(np.pi * progress))

# 生成曲线
steps = np.arange(0, total_steps)
lrs = np.array([get_lr(s) for s in steps])

# 绘制曲线
plt.figure(figsize=(10,6))
plt.plot(steps, lrs)
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('GPT Learning Rate Warm-up and Cosine Decay')
plt.grid(True)
plt.show()
