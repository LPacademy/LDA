import numpy as np
import scipy as sc
import math
import matplotlib.pyplot as plt

pcategory1=0.5
pcategory2=0.5
n1=100                        #number of data in category1
n2=100                        #number of data in category2
mu1=np.array([2,0])          #mean of category1
mu2=np.array([-2,0])         #mean of category2
beta=math.pi/4.*4
sigma=np.array([[9.-8.*(math.cos(beta))**2,8*math.cos(beta)*math.sin(beta)],[8*math.cos(beta)*math.sin(beta),9.-8.*(math.sin(beta))**2]]) #covariance matrix
data1=np.random.multivariate_normal(mu1,sigma,n1).T #1行目にｘ座標2行目にy座標を置くために転置
data2=np.random.multivariate_normal(mu2,sigma,n2).T #転置しないと（ｘ、ｙ）の組で出てきちゃう

#MLE to each category
mu1ML=np.mean(data1,axis=1)             #axis=1にしとかないとxもyも全部足して平均取るからキケン
mu1ML=np.reshape(mu1ML,(1,len(mu1ML)))  #こうしないと転置しても縦行列にならない…
sigma1ML=np.cov(data1)                  #covariance matrix of category1
mu2ML=np.mean(data2,axis=1)
mu2ML=np.reshape(mu2ML,(1,len(mu2ML)))    
sigma2ML=np.cov(data2)                  #covariance matrix of category2

#Fisher's linear discriminant analysis
sigmaML=((n1*1.0)/(n1+n2))*sigma1ML+((n2*1.0)/(n1+n2))*sigma2ML #各共分散行列の重み付き和
sigmainv=sc.linalg.inv(sigmaML)
a=sigmainv.dot((mu1ML.T-mu2ML.T)) #なるべくデータ内共分散が小さく、データ間共分散が大きいように射影できる方向がa
b=-0.5*(mu1ML.dot(sigmainv.dot(mu1ML.T))-mu2ML.dot(sigmainv.dot(mu2ML.T)))+math.log(n1*1.0/n2)#numpyでは*は要素同士の積だからdotを使おう

#calculate decision boundary
y=np.arange(-10,10.2,0.2)
x=(-b[0]-a[1]*y)/a[0]

#plot 
mu=(mu1ML[0]+mu2ML[0])/2.0
plt.title('no rotation, n1/n2=%e'%(n1*1.0/n2))
plt.plot(x,y)                       #plot decision boundary
plt.plot(data1[0],data1[1],"rs",label='category1')    #plot category1 by red square
plt.plot(data2[0],data2[1],"bs",label='category2')    #plot category1 by blue square
plt.plot(mu1ML[0][0],mu1ML[0][1],"ko")
plt.plot(mu2ML[0][0],mu2ML[0][1],"ko")
plt.plot(mu[0],mu[1],"ko")
plt.legend