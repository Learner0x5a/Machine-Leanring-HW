import torch
import torch.nn as nn
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

# mlb_dataset = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
mlb_dataset = pd.read_csv('Melbourne_housing_FULL.csv')
mlb_dataset = mlb_dataset.dropna(axis=0)
# mlb_dataset.to_csv('no-nan-Melbourne_housing_FULL.csv')
print(mlb_dataset.head())

# raw_buildingarea = mlb_dataset.BuildingArea.values
# raw_price = mlb_dataset.Price.values/1e5
# plt.scatter(raw_buildingarea,raw_price,s=2)
# plt.xlabel('BuildingArea')
# plt.ylabel('Price (1e5)')
# plt.savefig('raw-price.png')
# plt.clf()

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 输入和输出的维度都是1
    def forward(self, x):
        preds = self.linear(x)
        return preds

# 将字符串数据转为数值
def str2num(array):
    dict_ = []
    toindex = []
    for sub in array:
        if not sub in dict_:
            dict_.append(sub)
        toindex.append(dict_.index(sub))
    # print(toindex,len(toindex))
    res = np.array(toindex)
    return res,dict_

'''
按地区分别标准化
'''
Suburb = mlb_dataset.Suburb.values
Suburb,subdict = str2num(Suburb)

def norm(sub):
    sub_index = np.where(Suburb==subdict.index(sub))
    print(len(sub_index[0]))
    '''
    筛选错误数据：landsize - BuildingArea < 0
    '''
    sub_BuildingArea = mlb_dataset.BuildingArea.values[sub_index]
    sub_Landsize = mlb_dataset.Landsize.values[sub_index]
    filter_index = np.where(sub_Landsize-sub_BuildingArea > 0)
    '''
    选取特征列
    '''
    # sub_Landsize = Landsize[sub_index][:,np.newaxis]
    sub_BuildingArea = sub_BuildingArea[filter_index][:,np.newaxis]
    sub_Rooms = mlb_dataset.Rooms.values[sub_index][filter_index][:,np.newaxis]
    sub_Suburb = Suburb[sub_index][filter_index][:,np.newaxis]

    # x = np.hstack((sub_Suburb,sub_BuildingArea))
    x = sub_BuildingArea
    y = mlb_dataset.Price.values[sub_index][filter_index][:,np.newaxis]
    y = y/1e5
    # plt.scatter(x,y)
    # plt.xlabel('BuildingArea')
    # plt.ylabel('Price (1e5)')
    # plt.title(sub)
    # plt.savefig('images/'+sub+'.png')
    # plt.clf()

    if len(x) > 1 and not x.max() == x.min():
        x = (x - x.min())/(x.max()-x.min())
    else:
        x = np.full(np.shape(x),0.5)
    



    return x,y

x1,y1 = norm(subdict[0])
x2,y2 = norm(subdict[1])

x = np.vstack((x1,x2))
y = np.vstack((y1,y2))

for i in range(2,len(subdict)):
    xi,yi = norm(subdict[i])
    x = np.vstack((x,xi))
    y = np.vstack((y,yi))

# '''
# 去除最大、最小值
# '''
# id0 = np.where(0<x)
# x = x[id0]
# y = y[id0]
# id1 = np.where(x<1)
# x = x[id1]
# y = y[id1]
# x = x[:,np.newaxis]
# y = y[:,np.newaxis]

plt.scatter(x,y,s=2)
plt.savefig('merge-sub.png')
plt.clf()
    


x_train = x.astype(np.float32)
y_train = y.astype(np.float32)

if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

loss_func = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(),lr=1e-2,weight_decay=1e-3)

num_epochs = 2000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train),requires_grad=False).cuda()
        target = Variable(torch.from_numpy(y_train),requires_grad=False).cuda()
    else:
        inputs = Variable(torch.from_numpy(x_train),requires_grad=False)
        target = Variable(torch.from_numpy(y_train),requires_grad=False)

    preds = model(inputs)
    loss = loss_func(preds, target)

    opt.zero_grad() 
    loss.backward()
    opt.step()

    if epoch % 20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch, num_epochs, loss.data))

if torch.cuda.is_available():
    inputs = Variable(torch.from_numpy(x_train),requires_grad=False).cuda()
    target = Variable(torch.from_numpy(y_train),requires_grad=False).cuda()
else:
    inputs = Variable(torch.from_numpy(x_train),requires_grad=False)
    target = Variable(torch.from_numpy(y_train),requires_grad=False)
test_preds = model(inputs)
loss = loss_func(test_preds, target)
print('test loss:{:.6f}'.format(loss.data))

# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error,r2_score
# model = LinearRegression()
# model.fit(x_train,y_train)
# preds = model.predict(x_train)
# loss = mean_squared_error(preds,y_train)
# print(loss,model.intercept_,model.coef_)

plt.scatter(x_train,y_train,s=2)
plt.scatter(x_train,test_preds.cpu().detach().numpy(),c='red',s=2,label='predict,loss='+str(loss.cpu().detach().numpy()))
plt.xlabel(u'BuildingArea')
plt.ylabel(u'Price (1e5)')
plt.legend()
plt.savefig("lr.png")
plt.clf()
