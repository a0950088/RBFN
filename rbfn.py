import numpy as np
from kmeans import Kmeans
from math import exp

class RBF:
    def __init__(self, data, kmeans_k, kmeans_eps, epoch, learnrate): # 初始化RBF所需的參數
        self.inputdata = data[:,:-1]
        self.emin = min(data[:,-1:].T[0])
        self.emax = max(data[:,-1:].T[0])
        self.eoutputdata = self.normalized(data[:,-1:],self.emin,self.emax)
        self.m, self.sigma, self.w = self.initialKmeans(kmeans_k,kmeans_eps)
        self.epoch = epoch
        self.learnrate = learnrate
        self.hiddennode = np.zeros(self.m.shape[0]+1)
        self.hiddennode[0] = 1
        self.output = 0
    
    def normalized(self, edata, emin, emax): # 將期望輸出透過函數逼近正規化
        for i in range(edata.shape[0]):
            edata[i] = (edata[i]-emin)/(emax-emin)
        return edata
    
    def originalval(self, data): # 將輸出還原成原來的資料區間
        data = data*(self.emax-self.emin)+self.emin
        return data
    
    def initialKmeans(self, k, eps): # 初始化Kmeans找到高斯函數的初始值
        km = Kmeans(k,self.inputdata,eps)
        rbf_m, sigma = km.train_kmeans()
        rbf_w = np.random.uniform(-10,10,((rbf_m.shape[0])+1))
        rbf_w[0] = 1
        return rbf_m, sigma, rbf_w
    
    def forward(self, x): # 前饋階段
        for i in range(1,self.hiddennode.shape[0]):
            self.hiddennode[i] = self.gaussian(x, self.m[i-1], self.sigma[i-1])
        output = sum(self.hiddennode*self.w)
        return output
    
    def changeweight(self, x, yminusF): # 更新權重值
        self.w = self.w + (self.learnrate*yminusF*self.hiddennode)
        self.m = self.m + (self.learnrate*yminusF*self.w[1:]*self.hiddennode[1:]*((1/(self.sigma**2))*((x-self.m).T))).T
        self.sigma = self.sigma + (self.learnrate*yminusF*self.w[1:]*self.hiddennode[1:]*((1/(self.sigma**3))*sum(((x-self.m)**2).T)))
    
    def gaussian(self, x, m, s): # 高斯函數
        d = sum(((x-m)**2))
        g = exp(-(d)/(2*(s**2)))
        return g
    
    def train_rbf(self): # 訓練rbf網路
        for _ in range(self.epoch):
            loss = 0
            for i in range(self.inputdata.shape[0]):
                self.output = self.forward(self.inputdata[i])
                yminusF = self.eoutputdata[i][0]-self.output 
                loss += (yminusF**2)/2
                self.changeweight(self.inputdata[i], yminusF)
            loss = loss/self.inputdata.shape[0]
            print("Loss: ",loss)
    
    def test_rbf(self, dist): # 輸出測試輸入的結果
        theta = self.originalval(self.forward(dist))
        return theta