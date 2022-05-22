### 创建tensor

| 函数                              |           功能            |
| :-------------------------------- | :-----------------------: |
| Tensor(*sizes)                    |       基础构造函数        |
| tensor(data,)                     |  类似np.array的构造函数   |
| ones(*sizes)                      |         全1Tensor         |
| zeros(*sizes)                     |         全0Tensor         |
| eye(*sizes)                       |    对角线为1，其他为0     |
| arange(s,e,step)                  |    从s到e，步长为step     |
| linspace(s,e,steps)               | 从s到e，均匀切分成steps份 |
| rand/randn(*sizes)                |       均匀/标准分布       |
| normal(mean,std)/uniform(from,to) |     正态分布/均匀分布     |
| randperm(m)                       |         随机排列          |

### 选择函数

| 函数                            | 功能                                                  |
| :------------------------------ | ----------------------------------------------------- |
| index_select(input, dim, index) | 在指定维度dim上选取，比如选取某些行、某些列           |
| masked_select(input, mask)      | 例子如上，a[a>0]，使用ByteTensor进行选取              |
| nonzero(input)                  | 非0元素的下标                                         |
| gather(input, dim, index)       | 根据index，在dim维度上选取数据，输出的size与index一样 |

#### 改变形状

`view`共享数据 ，即改变了某一个值会同时发生改变，但还是两个不同的内存地址

Pytorch还提供了一个`reshape()`可以改变形状，但是此函数并不能保证返回的是其拷贝，所以不推荐使用。推荐先用`clone`创造一个副本然后再使用`view`。

使用`clone`还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源`Tensor`。

**最后，感觉没必要，reshape直观点**

#### 线性代数

| 函数                              | 功能                              |
| --------------------------------- | --------------------------------- |
| trace                             | 对角线元素之和(矩阵的迹)          |
| diag                              | 对角线元素                        |
| triu/tril                         | 矩阵的上三角/下三角，可指定偏移量 |
| mm/bmm                            | 矩阵乘法，batch的矩阵乘法         |
| addmm/addbmm/addmv/addr/baddbmm.. | 矩阵运算                          |
| t                                 | 转置                              |
| dot/cross                         | 内积/外积                         |
| inverse                           | 求逆矩阵                          |
| svd                               | 奇异值分解                        |

### Tensor 和 Numpy 相互转换

常用`numpy()`和`from_numpy()`，这两个函数是共享数据；可用`torch.tensor()`数据拷贝，但是不再共享数据

---



### 线性回归简洁

#### 读取数据

```python
batch_size = 10
num_works = 4
data = data.TensorDataset(*data_arrays) # 将多个数据集打包（如数据和标签）
train_iter = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
```

#### 定义模型

```python
net = nn.Sequential(nn.Linear(2, 1)) # 2，1分别为输入输出的维度
nn.Flatten()
nn.ReLU()
nn.Conv2d(1, 1, kernel_size=3, padding=1)

# 卷积层的输入输出
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
Y = conv2d(X.reshape((1, 1) + X.shape))

# 查看模型各层信息
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
```

#### 初始化参数

```python
# 方法1
nn.init.normal_(net[0].weight, mean=0, std=0.01) # 采用正态分布[均值0，方差0.01]对第1层的权值初始化
nn.init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 方法2
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 方法3
def init_weights(m): # m为某层，为该层的参数(仅权重)初始化
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01) 

net.apply(init_weights)
```

#### 损失函数和优化算法

```python
loss = nn.MSELoss() # 均方误差（L2范数）
loss = nn.CrossEntropyLoss() # 交叉熵，会自动计算softmax

optimizer = torch.optim.SGD(net.parameters(), lr=0.03) # SGD 优化
```

#### 设计模型

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
        self.weight = nn.Parameter(torch.rang(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

```

#### 权重修正

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
# 从零开始
l = loss(net(X), y) + lambd * l2_penalty(w)
l.sum().backward()
# 简洁实现
trainer = torch.optim.SGD([
	{"params":net[0].weight,'weight_decay': wd}, # 对w设置衰减
    {"params":net[0].bias}], lr=lr)

# dropout法
dropout1, dropout2 = 0.2, 0.5
nn.Linear(784, 256),
nn.ReLU(),
nn.Dropout(dropout1), # 在第一个全连接层之后添加一个dropout层
```

#### 训练模型模板

```python
def train_epoch(net, train_iter, loss, optimizer):
def train(net, train_iter, val_iter, loss, num_epochs, optimizer):

for X, y in train_iter:
    # 计算梯度并更新参数
    l = loss(net(X), y)
    if isinstance(updater, torch.optim.Optimizer):
        # 使用PyTorch内置的优化器和损失函数
        updater.zero_grad()
        l.backward()
        updater.step()
	else:
		# 使用定制的优化器和损失函数
    	l.sum().backward()
    	updater(X.shape[0]) # 小批量的样本数, 自己设计的update函数中梯度已经清零了
```

#### 保存&加载

```python
torch.save(net.state_dict(), './data/mlp.params')

model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()

# checkpoint save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    ...
}, PATH)
# checkpoint lo'a'd
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
```



### 一个网络的要素

#### 网络模型：预期的输出的形式，如拟合w和b的线性函数模型、再盖一层softmax就是概率模型，一般表现为一个矩阵（n个特征 * m个输出）的再处理

#### 损失函数：预测结果和实际的误差，包括L2均方误差，交叉熵等方法

#### 优化函数：通过该函数调整模型的参数，以达到最优，常见的就是sgd梯度下降（这过程就是求损失函数关于参数的导数，对于每个批量：计算损失，优化函数清零求导，更新模型参数）trainer = updater = optimizer

#### 数据：一条数据应表示成一个向量（横，如果是图片就拉成一条），各个元素表示一个特征，多条数据构成一个数据集，表现为 n条数据（列）* m个特征（行） ；经过网络模型就变成了输出，此时 n条数据（列）* m个结果（简单线性模型结果m=1，表示预测值；分类模型结果m=类别数，表示各类别概率）

#### 一般来说：param、lr、epoch 都是公共变量，batch_size 因为最后一批不足batch_size个，因此在优化函数中涉及的批量个数计算应以实际的作为输入（X.shape[0]）

#### 模型+损失函数+优化函数+训练集批量迭代器 = 一次模型训练

#### 包装 epoch数+测试集批量迭代器 = 完整模型训练+最终评测



#### 在自己实现中，参数需要显示的写出来，在框架中，参数是隐式在net定义中，通过net.parameter()调用，参数数组在优化函数中使用
