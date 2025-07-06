import sklearn.metrics
import torch.nn as nn
import torch
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# 定義一個 LogicNet 神經網路類別，繼承 nn.module
class LogicNet(nn.Module):
    # 初始化函式：傳入輸入層、隱藏層與輸出層的維度
    def __init__(self,inputdim,hiddendim,outputdim):
        super(LogicNet,self).__init__()  # 呼叫父類別 nn.Module 的建構子
        self.Linear1 = nn.Linear(inputdim,hiddendim)   # 建立全連接線性層，輸入為 imputdim，輸出為 hiddendim
        self.Linear2 = nn.Linear(hiddendim,outputdim)
        # self.add_module("Linear1",nn.Linear(inputdim,hiddendim))
        # self.add_module("Linear2",nn.Linear(hiddendim,outputdim))
        self.criterion = nn.CrossEntropyLoss() # 定義交叉熵函數
    
    # 搭建用兩層全連接層組成的網路模型
    def forward(self,x):
        x = self.Linear1(x)  # 將輸入傳入第一層
        x = torch.tanh(x)    # 對第一層輸出做非線性變換 (激活函數)
        x = self.Linear2(x)  
        return x
    
    # 實現 LogicNet 的預測接口
    def predict(self,x):
        # 調用自身網路模型，並對所有 x 的預測結果做 softmax 處理，分別得出每一類的預測機率
        pred = torch.softmax(self.forward(x),dim = 1)
        return torch.argmax(pred,dim = 1) # 返回每組預測機率中的最大值索引
    
    # 定義損失函數
    def getloss(self,x,y):
        # 之所以用 forward() 計算輸出是因為 CE 函數參數為張量與標籤類別，並不是直接輸入預測的類別值
        y_pred = self.forward(x)
        loss = self.criterion(y_pred,y)
        return loss

def plot_losses(losses):
    # 計算每 w 筆的移動平均損失值
    def moving_avg(a,w = 10):
        if len(a) < w:
            return a[:]
        return [val if idx < w else sum(a[(idx - w):idx]) / w for idx ,val in enumerate(a)]

    avg_loss = moving_avg(losses) 
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(len(avg_loss)), avg_loss, 'b--')
    plt.xlabel('step number')
    plt.ylabel('Training loss')
    plt.show()
    

# 調用 LogicNet model 的預測函數，輸出 numpy 型別的預測
def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()

def plot_decision_boundary(pred_func,X,Y):
    # 計算 x 軸、y 軸取值範圍
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # 生成網格矩陣
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # np.c_[] 是將 xx 和 yy 合併成 [[x1, y1], [x2, y2], ...]，進行預測
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    # Z 為預測出的一維陣列，要將他 reshape 回 xx 的形狀才能對應到每一點的預測
    Z = Z.reshape(xx.shape)
    # 依照預測結果畫出決策邊界
    plt.contourf(xx, yy, Z, cmap = plt.cm.Spectral)
    plt.title("Linear predict")
    arg = np.squeeze(np.argwhere(Y==0),axis = 1)
    arg2 = np.squeeze(np.argwhere(Y==1),axis = 1)
    plt.scatter(X[arg,0], X[arg,1], s=100,c='b',marker='+')
    plt.scatter(X[arg2,0], X[arg2,1],s=40, c='r',marker='o')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    X, Y = sklearn.datasets.make_moons(200, noise = 0.2)

    arg = np.squeeze(np.argwhere(Y==0),axis = 1)    
    arg2 = np.squeeze(np.argwhere(Y==1),axis = 1)

    plt.title("moons data")
    plt.scatter(X[arg,0], X[arg,1], s=100,c='b',marker='+',label='data1')
    plt.scatter(X[arg2,0], X[arg2,1],s=40, c='r',marker='o',label='data2')
    plt.legend()
    plt.show()

    model = LogicNet(inputdim = 2,hiddendim = 3,outputdim = 2)
    # 定義優化器，model.parameters() 會回傳網路所有需要被訓練的參數
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

    # 印出模型資訊
    for sub_module in model.children():
        print(sub_module)

    for name, module in model.named_children():
        print(name,"is: ",module)

    for module in model.modules():
        print(module)

    # 印出模型參數
    for name, param in model.named_parameters():
        print(type(param.data),param.size(),name)

    # 將 numpy 資料轉為張量。再轉為 Float
    xt = torch.from_numpy(X).type(torch.FloatTensor)
    yt = torch.from_numpy(Y).type(torch.LongTensor)

    epochs = 1000
    losses = []
    for i in range(epochs):
        # 預測並計算 loss 值
        loss = model.getloss(xt,yt)
        losses.append(loss.item())
        optimizer.zero_grad()   # 每一輪訓練需清空之前的梯度
        loss.backward()         # 反向傳播計算損失值梯度
        optimizer.step()        # 更新參數

    plot_losses(losses)

    print(accuracy_score(model.predict(xt),yt))

    plot_decision_boundary(predict ,xt.numpy(), yt.numpy())
