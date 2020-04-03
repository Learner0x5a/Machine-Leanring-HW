import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD,Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

# mlb_dataset = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
mlb_dataset = pd.read_csv('Melbourne_housing_FULL.csv')
mlb_dataset = mlb_dataset.dropna(axis=0)

print(mlb_dataset.head())


# 将字符串数据转为数值
def str2num(array):
    sub_dict = []
    toindex = []
    for sub in array:
        if not sub in sub_dict:
            sub_dict.append(sub)
        toindex.append(sub_dict.index(sub))
    # print(toindex,len(toindex))
    res = np.array(toindex)
    return res

'''
地区字符串转数值
但其实可以由postcode代替
'''
Suburb = mlb_dataset.Suburb.values
Suburb = str2num(Suburb)

'''
筛选错误数据：landsize - BuildingArea < 0
'''
BuildingArea = mlb_dataset.BuildingArea.values
Landsize = mlb_dataset.Landsize.values
index = np.where(Landsize-BuildingArea>0)


'''
选取特征列
'''
# Landsize = Landsize[index][:,np.newaxis]
BuildingArea = BuildingArea[index][:,np.newaxis]
Suburb = Suburb[index][:,np.newaxis]

Rooms = mlb_dataset.Rooms.values[index][:,np.newaxis]
Bathroom = mlb_dataset.Bathroom.values[index][:,np.newaxis]
Postcode = mlb_dataset.Postcode.values[index][:,np.newaxis]
Propertycount = mlb_dataset.Propertycount.values[index][:,np.newaxis]
Car = mlb_dataset.Car.values[index][:,np.newaxis]

x = np.hstack((BuildingArea,Postcode,Rooms,Bathroom,Propertycount,Car))
y = mlb_dataset.Price.values[index]
x = (x - x.min())/(x.max()-x.min())
y = y/1e5
# y = (y - y.min())/(y.max()-y.min())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
print(np.shape(x_train),np.shape(y_train))

model = Sequential()
model.add(Dense(10,activation='linear'))
model.add(Dense(10,activation='linear'))
model.add(Dense(1,activation='linear'))
model.compile(optimizer=Adam(),loss='mse')

model.fit(x_train,y_train,epochs=200)
model.summary()
preds = model.predict(x_test)
loss = mean_squared_error(preds,y_test)



line_x = np.linspace(min(y_test),max(y_test),10)
line_y = np.linspace(min(y_test),max(y_test),10)
plt.scatter(y_test,preds,c='black',s=8,label='MSE='+str(loss))
plt.xlabel(u'True (1e5)')
plt.ylabel(u'Predicted (1e5)')
plt.plot(line_x,line_y,color='r',label='y=x')
plt.legend()
plt.savefig("nn.png")
