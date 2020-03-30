# 线性回归预测房价

## 数据预处理
 +	无效数据去除：根据数据的物理意义进行筛选，例如landsize < BuildingArea的数据；nan的数据。
 + 	数值化处理：例如将Suburb列转化为数值数据，每个字符串映射为一个数值。
 +	特征的选取：原始数据集中一些特征具有强相关性，例如Suburb可以用Postcode替代，选其一即可。
 +	归一化：采用最大值-最小值归一化，即X=(X–X.mean())/(X.max()-X.min())。

## Code

`sklearn-regression`: 线性回归、岭回归、决策树、极端随机树

`keras-regression`: 简单神经网络回归

`split-regression`: 单特征分析 - `BuildingArea - Price分布`
