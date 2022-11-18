import numpy as np

class Kmeans:
    def __init__(self, k, data, eps): # 初始化kmeans所需的參數
        self.data = data
        self.k = k
        self.eps = eps
        self.w = data[np.random.choice(data.shape[0], self.k),:]
        self.dividezero = False
    
    def cal_new_w(self): # 計算kmeans新的中心
        d = []
        dataresult = []
        for i in range(self.data.shape[0]):
            d = []
            for j in range(self.w.shape[0]):
                d.append(sum((abs(self.w[j]-self.data[i])**2))**0.5)
            dataresult.append(d.index(min(d)))
        dataresult = np.array(dataresult)
        
        new_w = np.zeros((self.w.shape[0],self.w.shape[1]))
        addtime = np.zeros(self.w.shape[0])
        for i in range(dataresult.shape[0]):
            new_w[dataresult[i]] += self.data[i]
            addtime[dataresult[i]] += 1
        
        for nw in new_w:
            if nw.any()==0:
                print("DIVIDE ZERO OCCURED")
                print("KMEANS RECOMPUTING...")
                self.w = self.data[np.random.choice(self.data.shape[0], self.k),:]
                self.dividezero = True
                return new_w, dataresult, self.dividezero
        
        self.dividezero = False
        for i in range(new_w.shape[0]):
            new_w[i] = new_w[i]/addtime[i]
                    
        return new_w, dataresult, self.dividezero
    
    def train_kmeans(self): # 訓練kmeans直到前一個中心和現在的中心差距小於epsilon
        new_w, correspondWdata, dividezero = self.cal_new_w()
        while dividezero:
            new_w, correspondWdata, dividezero = self.cal_new_w()
        while True:
            if sum(sum(abs(new_w-self.w)<self.eps)) == np.size(self.w):# Sum of all True
                self.w = new_w
                break
            self.w = new_w
            new_w,correspondWdata,_ = self.cal_new_w()
        
        sigma = np.zeros(self.w.shape[0])
        for i in range(self.w.shape[0]):
            p=0
            for k in range(self.data.shape[0]):
                if correspondWdata[k] == i:
                    sigma[i] += sum((self.w[i]-self.data[k])**2)
                    p+=1
            sigma[i] = (sigma[i]/p)**0.5
        return self.w, sigma