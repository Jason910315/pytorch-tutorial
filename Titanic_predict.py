import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from scipy import stats

titanic_data = pd.read_csv("titanic3.csv")
print(titanic_data.columns)
# 用虛擬變數將離散型特徵轉成 one-hot encoding
# get_dummies() 會根據列中的離散值重新生成新的列，用 True/False 表示該筆資料是否具有該列的屬性
titanic_data = pd.concat([titanic_data,
                          pd.get_dummies(titanic_data['Sex']),
                          pd.get_dummies(titanic_data['Embarked'],prefix = 'Embarked'),
                          pd.get_dummies(titanic_data['Pclass'],prefix = 'Pclass')],axis = 1)

print(titanic_data.columns)

# 處理 Nan 值
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
titanic_data = titanic_data.drop(['PassengerId','Name','Ticket','Sex','Cabin','Embarked','Pclass'],axis = 1)

# 定義類別與特徵
labels = titanic_data['Survived'].to_numpy()
titanic_data = titanic_data.drop(['Survived'],axis = 1)
data = titanic_data.to_numpy()
feature_names = list(titanic_data.columns)

# 將樣本切割為 70% 訓練集、30% 測試集
np.random.seed(10)
# 隨機抽取 70% 的資料集索引值
train_indices = np.random.choice(len(labels),int(0.7 * len(labels)),replace = False)
test_indices = list(set(range(len(labels))) - set(train_indices))
train_features = data[train_indices]
train_labels = labels[train_indices]
test_features = data[test_indices]
test_labels = labels[test_indices]

# 將先前做 one-hot 的欄位值轉為 0/1
train_features = train_features.astype(np.float32)
test_features = test_features.astype(np.float32)

# 定義 Mish 激活函数
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.nn.Tanh()(nn.Softplus()(x)))
        return x
    
torch.manual_seed(0)  # 設定隨機種子，使每次執行用的權重張量都是固定的

class ThreelinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12,12)   # 資料集特徵維度為 12
        self.Mish1 = Mish()
        self.linear2 = nn.Linear(12,8)
        self.Mish2 = Mish()
        self.linear3 = nn.Linear(8,2)     # 最後一層將輸出轉為與類別數量一致
        self.softmax = nn.Softmax(dim = 1)
        self.criterion = nn.CrossEntropyLoss()  # 損失函數是以類別形式封裝，故要先實例化才能使用

    # 定義全連接網路，並返回 softmax 輸出
    def forward(self,x):
        lin1_out = self.linear1(x)
        out1 = self.Mish1(lin1_out)
        out2 = self.Mish2(self.linear2(out1))
        return self.softmax(self.linear3(out2))
    
    def get_loss(self,x,y):
        y_pred = self.forward(x)
        # CE 函數接收的是輸出的張量與真實標籤，故不用將 softmax 額外輸出為預測類別
        loss = self.criterion(y_pred,y)
        return loss
    
# 當程式被直接執行時才會執行以下程式碼，被引用時不會執行
if __name__ == '__main__':
    net = ThreelinearModel()
    epochs = 200
    optimizer = torch.optim.Adam(net.parameters(),lr = 0.04)  # 定義優化器
    # 設定學習率衰減物件
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.99)
    
    # 將輸入的訓練集特徵轉為張量
    input_tensor = torch.from_numpy(train_features)
    label_tensor = torch.from_numpy(train_labels)

    losses = []
    for epoch in range(epochs):
        loss = net.get_loss(input_tensor,label_tensor)  # 訓練並計算 loss
        losses.append(loss.item())
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()        # 反向傳播損失值並計算梯度
        optimizer.step()       # 更新 model 裡的參數
        scheduler.step()       # 學習率衰減
        if epoch % 20 == 0:
            print('Epoch {}/{} => Loss: {:.2f}'.format(epoch + 1,epochs,loss.item()))

    # 建立 models 資料夾，以便儲存訓練完的模型
    os.makedirs('models',exist_ok = True)  
    torch.save(net.state_dict(),'models/titanic_model.pt')

    from code_use_module import plot_losses
    plot_losses(losses)

    # 輸出訓練結果，net 已經被訓練完畢
    train_output_probs = net(input_tensor).detach().numpy()  # 這裡輸出都是 softmax 後的機率，再用 detach() 脫離計算圖
    train_output_classes = np.argmax(train_output_probs,axis = 1) 
    print("Trianing Accuracy:",sum(train_output_classes == train_labels) / len(train_labels))

    # 測試模型
    test_input_tensor = torch.from_numpy(test_features)
    test_output_probs = net(test_input_tensor).detach().numpy()
    test_output_classes = np.argmax(test_output_probs,axis = 1)
    print("Test Accuracy:",sum(test_output_classes == test_labels) / len(test_labels))