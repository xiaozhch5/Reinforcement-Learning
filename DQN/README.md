# 为什么使用两个网络

在计算Q-target时如果使用同一个网络参数，这可能导致选择过高的估计值，从而导致过于乐观的值估计。为了避免这种情况的出现，我们可以对选择和衡量进行解耦，从而就有了双Q学习