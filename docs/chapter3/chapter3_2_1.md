# OBD公式推导

&emsp;&emsp;Optimal Brain Damage (OBD)是1989年提出来的一种剪枝方法，该方法通过利用二阶导数信息在网络复杂度和训练集误差之间进行权衡，有选择地从网络中删除了一些不重要的权重，减少学习网络大小。

&emsp;&emsp;已知等式：
$$
Y=XW \tag{1}
$$
&emsp;&emsp;其中，$X$为输入，$Y$为输出，$W$为权重。对式（1）变换后可得：
$$
W=X^{-1}Y=\left(X^T X\right)^{-1} X^T Y  \tag{2}
$$

&emsp;&emsp;泰勒公式用多项式来近似表示函数在某点周围的情况，一元函数$f(x)$在$x_k$处的泰勒展开式如下：

$$
f(x)=f(x_k)+f^{\prime}(x_k)(x-x_k)+\frac{f^{\prime \prime}(x_k)}{2 !}(x-x_k)^2+\frac{f^{\prime \prime \prime}(x_k)}{3 !}(x-x_k)^3+\ldots \tag{3}
$$

&emsp;&emsp;二元函数在点 $(x_k,y_k)$ 处的泰勒展开式为：
$$
\begin{gathered}
f(x, y)=f\left(x_k, y_k\right)+\left(x-x_k\right) f_x^{\prime}\left(x_k, y_k\right)+\left(y-y_k\right) f_y^{\prime}\left(x_k, y_k\right) \\
+\frac{1}{2 !}\left(x-x_k\right)^2 f_{x x}^{\prime \prime}\left(x_k, y_k\right)+\frac{1}{2 !}\left(x-x_k\right)\left(y-y_k\right) f_{x y}^{\prime \prime}\left(x_k, y_k\right) \\
+\frac{1}{2 !}\left(x-x_k\right)\left(y-y_k\right) f_{y x}^{\prime \prime}\left(x_k, y_k\right)+\frac{1}{2 !}\left(y-y_k\right)^2 f_{y y}^{\prime \prime}\left(x_k, y_k\right) +\ldots 
\end{gathered} \tag{4}
$$

&emsp;&emsp;多元函数(n)在点 $x_k$ 处的泰勒展开式为：

$$
\begin{gathered}
f\left(x^1, x^2, \ldots, x^n\right)=f\left(x_k^1, x_k^2, \ldots, x_k^n\right)+\sum_{i=1}^n\left(x^i-x_k^i\right) f_{x^i}^{\prime}\left(x_k^1, x_k^2, \ldots, x_k^n\right) \\
+\frac{1}{2!} \sum_{i, j=1}^n\left(x^i-x_k^i\right)\left(x^j-x_k^j\right) f_{i j}^{\prime \prime}\left(x_k^1, x_k^2, \ldots, x_k^n\right)  +o^n \tag{5}
\end{gathered}
$$


&emsp;&emsp;推广到矩阵形式，可表示为：

$$
f(X)=f\left(X_k\right)+ \nabla f\left(X_k\right)\left(X-X_k\right)^T+\frac{1}{2 !}\left(X-X_k\right)^T H\left(X_k\right)\left(X-X_k\right)+o^n \tag{6}
$$
&emsp;&emsp;其中，$x^T=[x^1, x^2, x^3 ... x^n]$，$H$矩阵为海森矩阵，是一个多元函数的二阶偏导数构成的方阵，描述了函数的局部曲率。海森矩阵常用于牛顿法解决优化问题，利用海森矩阵可判定多元函数的极值问题。

$$
H\left(\mathbf{x}_k\right)=\left[\begin{array}{cccc}
\frac{\partial^2 f\left(x_k\right)}{\partial x_1^2} & \frac{\partial^2 f\left(x_k\right)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f\left(x_k\right)}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f\left(x_k\right)}{\partial x_2 \partial x_1} & \frac{\partial^2 f\left(x_k\right)}{\partial x_2^2} & \cdots & \frac{\partial^2 f\left(x_k\right)}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f\left(x_k\right)}{\partial x_n \partial x_1} & \frac{\partial^2 f\left(x_k\right)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f\left(x_k\right)}{\partial x_n^2}
\end{array}\right] 
$$

&emsp;&emsp;对于神经网络的损失函数$E(w;x)$，假设当权重$w$收敛至$w_0$时，此时目标函数$E$获得极小值。将权重$w$和极值$w_0$代入式中可得：

$$
{E(w)}=E\left(w_0\right)+E^{\prime}\left(w_0\right)\left(w-w_0\right)+\frac{1}{2 !}\left(w-w_0\right)^T\cdot H\cdot \left(w-w_0\right)+o^n \tag{7}
$$

&emsp;&emsp;其中，$H=\partial^2 E / \partial w^2$,进一步，$\delta E$可表示为：

$$
\delta E=E(w)-E\left(w_0\right)=E^{\prime}\left(w_0\right)\left(w-w_0\right)+\frac{1}{2 !}\left(w-w_0\right)^T\cdot H\cdot \left(w-w_0\right)+o^n \tag{8}
$$


&emsp;&emsp;关于目标函数的变化$\delta E$泰勒展开式：
$$
\delta E=\left(\frac{\partial E}{\partial w}\right)^T \cdot \delta w+\frac{1}{2} \delta w^T \cdot H \cdot \delta w+o^n \tag{9}
$$


&emsp;&emsp;目标是找到一组参数，使得删除之后目标函数$E$的变化最小。OBD算法有三个重要的假设条件：
- 对角线假设：参数对目标函数的影响是相互独立的。
- 极值假设：在训练收敛后执行参数删除。
- 二次假设：目标函数是近似二次的。

&emsp;&emsp;根据上面的假设(2)和（3），训练到局部误差最小的网络，第一项为0，第三项和后面的高阶项复杂度太高，所以只保留第二项。即：
$$
\delta E=\frac{1}{2} \delta w^T \cdot H \cdot \delta w \tag{10}
$$

&emsp;&emsp;$H$可表示为：
$$
H=\sum_{i=j} \frac{\partial^2 E}{\partial \omega_{i j}^2}+\sum_{i \neq j} \frac{\partial^2 E}{\partial \omega_{i j}^2} \tag{11}
$$

&emsp;&emsp;根据上面的假设(1)可知，式11的第二项为0。即只需要考虑对角线的元素。

## 参考链接

- [Optimal Brain Surgeon公式推导](https://zhuanlan.zhihu.com/p/656316235)
- [剪枝界的双弹瓦斯OBD与OBS](https://zhuanlan.zhihu.com/p/680853298)