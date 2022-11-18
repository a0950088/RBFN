import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
import numpy as np
import math as m
import matplotlib
import os
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from rbfn import RBF

# 定義全域變數
cwd = os.getcwd()
file_base = ''
rc = [] # 車子軌跡的紀錄

def convertfile(content): # 轉換.txt檔案資料
    for c in range(len(content)):
        content[c] = float(content[c])
    return content

def drawpoint(): # 畫出車子軌跡
    global ax, canvas, rc
    r = rc.pop(0)
    front.set(str(r[2]))
    right.set(str(r[3]))
    left.set(str(r[4]))
    ax.scatter(r[0],r[1],s=400)
    canvas.draw()
    if rc != []:
        root.after(100,drawpoint)
    
def drawmap(): # 畫出地圖
    global fig, ax, canvas, cwd
    fig = Figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    filepath = f"{cwd}\軌道座標點.txt"
    f = open(filepath,mode='r')
    file = f.read().split('\n')
    data = [convertfile(fc.split(',')) for fc in file if fc != '']
    ax.title.set_text('Map')
    ax.set_aspect(1)
    ax.set_xlim(-11,42)
    ax.set_ylim(-5,55)
    
    #end rectangle
    rtg_x = [data[1][0],data[2][0],data[2][0],data[1][0],data[1][0]]
    rtg_y = [data[1][1],data[1][1],data[2][1],data[2][1],data[1][1]]
    ax.plot(rtg_x,rtg_y)
    
    #line
    mp = data[3:]
    mx,my = zip(*mp)
    ax.plot(mx,my)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5,column=1,columnspan=5)
    
def _readfile(): # 讀取檔案
    global data,file_base
    filename = filedialog.askopenfilename()
    f = open(filename,mode='r')
    file = f.read().split('\n')
    if type(file_base) == str:
        fileentry.delete(0, 'end')
    file_base=os.path.basename(filename)
    fileentry.insert(0, file_base)
    data = np.array([convertfile(fc.split(' ')) for fc in file if fc != ''])
    f.close()

def _training(): # 訓練車子轉向角度並將成功結果寫入檔案
    global fig, rc
    lr = float(learnrate.get())
    ep = epoch.get()
    car = Car(data,lr,ep)
    car.trainRBF()
    while True:
        car.predict()
        flag, suc = car.run()
        if flag:
            break
    if suc:
        print("Success")
        if file_base == 'train4dAll.txt':
            f = open('track4D.txt', 'w')
            for record in car.record:
                for r in record[2:]:
                    f.write(str(r)+' ')
                f.write('\n')
        elif file_base == 'train6dAll.txt':
            f = open('track6D.txt', 'w')
            for record in car.record:
                for r in record:
                    f.write(str(r)+' ')
                f.write('\n')
    else:
        print("Fail")
    rc = car.record
    plt.ion()
    drawmap()
    drawpoint()
    plt.ioff()
    
def _quit(): # 結束程式
    root.quit()
    root.destroy()

class Car:
    def __init__(self,data,lr,ep): # 初始化車子所需的參數
        self.carx=0
        self.cary=0
        self.phi=90
        self.r = 3
        self.theta = 0
        self.maxtheta = 40
        self.mintheta = -40
        self.maxphi = 270
        self.minphi = -90
        self.front = 0
        self.left = 0
        self.right = 0
        self.rbf = RBF(data, 10, 0.000001, ep, lr)
        self.dist = data[0][:-1]
        self.checkdist = True if len(self.dist) == 3 else False
        self.record = [[self.carx,self.cary,22.0000,8.4853,8.4853,self.theta]]
        self.line = np.array([[1,0,-6],
                              [0,1,22],
                              [1,0,18],
                              [0,1,50],
                              [1,0,30],
                              [0,1,10],
                              [1,0,6],
                              [0,1,-3]])
    
    def trainRBF(self): # 訓練RBF網路
        self.rbf.train_rbf()
    
    def predict(self): # 預測車子轉向角度theta
        #distinguish 4D or 6D
        if len(self.dist) == 3:
            pass
        elif self.checkdist:
            self.dist = self.dist[2:]
        
        #compute theta
        self.theta = self.rbf.test_rbf(self.dist)
        if self.theta > self.maxtheta:
           self.theta = self.maxtheta
        elif self.theta < self.mintheta:
            self.theta = self.mintheta
    
    def setphi(self): # 設定phi的角度範圍
        self.phi%=360
        if self.phi > self.maxphi:
            self.phi -= (self.maxphi - self.minphi)
    
    def changexyphi(self): # 改變車子中心xy和車子與水平軸的角度phi
        self.setphi()
        rtheta = m.radians(self.theta)
        rphi = m.radians(self.phi)
        rphi -= m.asin((2*m.sin(rtheta))/6)
        self.carx += m.cos(rphi+rtheta) + m.sin(rtheta)*m.sin(rphi)
        self.cary += m.sin(rphi+rtheta) - m.sin(rtheta)*m.cos(rphi)
        self.phi = m.degrees(rphi)
        
    def getLinearEquation(self,x1,y1,x2,y2): # 計算車子感測器與水平線的方程式
        a = y2-y1
        sign = -1 if a < 0 else 1
        a *= sign
        b = sign*(x1-x2)
        c = sign*((y1*x2) - (x1*y2))
        return np.array([a,b,c])
    
    def computeLine(self, phi): # 計算感測器、水平線與車子的交點
        rphi = m.radians(phi)
        circlepointx = self.r*m.cos(rphi)+self.carx
        circlepointy = self.r*m.sin(rphi)+self.cary
        line = self.getLinearEquation(self.carx,self.cary,circlepointx,circlepointy)
        return line, circlepointx, circlepointy
    
    def check(self, paraline, inputline, checkPosOrNeg): # 確認牆壁與感測器的交點並計算距離
        points = []
        for l in self.line:
            #compute intersection
            intersection = []
            if round(inputline[0],6) == 0:
                if l[0] == 1:
                    intersection = [l[2],self.cary]
                else:
                    continue
            elif round(inputline[1],6) == 0:
                if l[1] == 1:
                    intersection = [self.carx,l[2]]
                else:
                    continue
            else:
                if l[0] == 1:
                    intersection = [l[2],(-inputline[0]*l[2]-inputline[2])/inputline[1]]
                elif l[1] == 1:
                    intersection = [(-inputline[1]*l[2]-inputline[2])/inputline[0],l[2]]
            
            #check intersection is in front
            if checkPosOrNeg:
                if paraline[0]*intersection[0]+paraline[1]*intersection[1]+paraline[2] >= 0:
                    points.append(intersection)
            else:
                if paraline[0]*intersection[0]+paraline[1]*intersection[1]+paraline[2] < 0:
                    points.append(intersection)
        points = np.array(points)
        
        #check point is not on hidden wall
        d = 0
        for p in points:
            if p[1]==22 and p[0] > 18 and p[0] < 30:
                continue
            elif p[0]==18 and p[1] > 10 and p[1] < 22:
                continue
            elif p[1]==10 and p[0] > -6 and p[0] < 6:
                continue
            elif p[0]==6 and p[1] > 10 and p[1] < 22:
                continue
            else:
                if d == 0:
                    d = ((p[0]-self.carx)**2+(p[1]-self.cary)**2)**0.5
                elif d > ((p[0]-self.carx)**2+(p[1]-self.cary)**2)**0.5:
                    d = ((p[0]-self.carx)**2+(p[1]-self.cary)**2)**0.5
                else:
                    continue
        return d
    
    def run(self): # 計算每一步車子的所有資訊(感測器、中心xy、輸出角度)
        breakflag = False
        success = False
        self.changexyphi()
        frontline, cx, cy = self.computeLine(self.phi)
        rightline, _, _ = self.computeLine(self.phi-45)
        leftline, _, _ = self.computeLine(self.phi+45)
        paraline, _, _ = self.computeLine(self.phi+90)
        checkPosOrNeg = True if paraline[0]*cx+paraline[1]*cy+paraline[2] >= 0 else False
        self.front = self.check(paraline, frontline, checkPosOrNeg)
        self.right = self.check(paraline, rightline, checkPosOrNeg)
        self.left = self.check(paraline, leftline, checkPosOrNeg)
        self.dist = [self.carx, self.cary, self.front, self.right, self.left]
        self.record.append([self.carx, self.cary, self.front, self.right, self.left, self.theta])
        print("--------------------------")
        print("DIST: ",self.dist[2:])
        print("THETA: ",self.theta)
        print("CARXY: ",self.carx, self.cary)
        
        #check car collided or arrived
        if self.front < 3 or self.right < 3 or self.left < 3:
            breakflag = True
            return breakflag, success
        if self.cary >= 37 and self.carx >= 21 and self.carx <= 27:
            breakflag = True
            success = True
            return breakflag, success
        return breakflag, success
     
#UI interface
root = tk.Tk()
root.title("Test")
root.geometry("450x700")
file_base = tk.StringVar()
learnrate = tk.StringVar()
epoch = tk.IntVar()
front = tk.StringVar()
right = tk.StringVar()
left = tk.StringVar()

f = tkFont.Font(family='Ink Free')
tk.Label(master=root, text="Dataset: ",width=10,height=1,font=f).grid(row=0,column=0,columnspan=2)
fileentry = tk.Entry(master=root, textvariable=file_base)
fileentry.grid(row=0,column=2,columnspan=2)

tk.Label(master=root, text="Learnrate: ",width=10,height=1,font=f).grid(row=1,column=0,columnspan=2)
rateentry = tk.Spinbox(master=root, from_=0.01, to=20, increment=0.1, textvariable=learnrate, format="%.2f")
rateentry.grid(row=1,column=2,columnspan=2)

tk.Label(master=root, text="Epoch: ",width=10,height=1,font=f).grid(row=2,column=0,columnspan=2)
epochentry = tk.Spinbox(master=root, from_=1, to=1000000, increment=10, textvariable=epoch)
epochentry.grid(row=2,column=2,columnspan=2)

readfile_btn = tk.Button(master=root, text="Choose File", command=_readfile,width=10,height=1,font=f)
readfile_btn.grid(row=0,column=4,columnspan=2)

train_btn = tk.Button(master=root, text="TRAINING!", command=_training,width=10,height=1,font=f)
train_btn.grid(row=2,column=4,columnspan=2)

tk.Label(master=root).grid(row=9,column=0,columnspan=2)
tk.Label(master=root).grid(row=9,column=2,columnspan=2)
quit_btn = tk.Button(master=root, text="QUIT", command=_quit,width=10,height=1,font=f)
quit_btn.grid(row=9,column=4)
tk.Label(master=root).grid(row=3,column=0)
tk.Label(master=root).grid(row=4,column=0)

tk.Label(master=root, text="Front Distence: ",font=f).grid(row=6,column=0,columnspan=2)
tk.Label(master=root, textvariable=front,font=f).grid(row=6,column=2,columnspan=4)
tk.Label(master=root, text="Right Distence: ",font=f).grid(row=7,column=0,columnspan=2)
tk.Label(master=root, textvariable=right,font=f).grid(row=7,column=2,columnspan=4)
tk.Label(master=root, text="Left Distence: ",font=f).grid(row=8,column=0,columnspan=2)
tk.Label(master=root, textvariable=left,font=f).grid(row=8,column=2,columnspan=4)

fig = Figure(figsize=(4,4))
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=5,column=1,columnspan=5)

root.mainloop()
