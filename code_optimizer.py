import sklearn.datasets     #引入数据集
import torch
import numpy as np
import matplotlib.pyplot as plt
from code_use_module import LogicNet,predict,plot_decision_boundary

np.random.seed(0)          
X, Y = sklearn.datasets.make_moons(200,noise=0.2) 

model = LogicNet(inputdim=2,hiddendim=3,outputdim=2)      #初始化模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #定義優化器

# 將 numpy 資料轉為張量。再轉為 Float
xt = torch.from_numpy(X).type(torch.FloatTensor)
yt = torch.from_numpy(Y).type(torch.LongTensor)

epochs = 1000
losses = []
lr_list = []  # 儲存學習率衰減實現過程的 lr
# 對 optimizer 設定學習率衰減 : 每 50 步乘以 0.99
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.99)

for i in range(epochs):
    # 預測並計算 loss 值
    loss = model.getloss(xt,yt)
    losses.append(loss.item())
    optimizer.zero_grad()   # 每一輪訓練需清空之前的梯度
    loss.backward()         # 反向傳播計算損失值梯度
    optimizer.step()        # 更新參數
    scheduler.step()        # 更新學習率
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

plt.plot(range(epochs),lr_list,color = 'r')
plt.show()


from sklearn.metrics import accuracy_score
print(accuracy_score(model.predict(xt),yt))








