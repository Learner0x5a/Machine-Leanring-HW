import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


# mlb_dataset = pd.read_csv('MELBOURNE_HOUSE_PRICES_LESS.csv')
mlb_dataset = pd.read_csv('Melbourne_housing_FULL.csv')
mlb_dataset = mlb_dataset.dropna(axis=0)

print(mlb_dataset.head())

y = mlb_dataset.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'Propertycount','Car','Distance']
x= mlb_dataset[features]
print(x.head(),y.head())
x = (x - x.min())/(x.max()-x.min())
# y = (y - y.min())/(y.max()-y.min())
print(x.head(),y.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
print(np.shape(x_train),np.shape(y_train))


# from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor,ExtraTreesRegressor
# model = ExtraTreesRegressor()

model = LinearRegression()
model.fit(x_train,y_train)
preds = model.predict(x_test)
loss = r2_score(preds,y_test)
print(loss)


line_x = np.linspace(min(y_test),max(y_test),10)
line_y = np.linspace(min(y_test),max(y_test),10)
plt.scatter(y_test,preds,c='black',marker='s',s=8,label='R2='+str(loss))
plt.xlabel(u'True',FontSize=16)
plt.ylabel(u'Predicted',FontSize=16)
plt.plot(line_x,line_y,color='r',label='y=x')
plt.legend()
# plt.show()
plt.savefig("lr.png")

