"""
boids.py 

Implementation of Craig Reynold's BOIDs

Author: Mahesh Venkitachalam
"""

import sys, argparse
import math
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

# 定义画布的宽度和高度
width, height = 640, 480

class Boids:
    """表示Boids仿真的类"""
    def __init__(self, N):
        """ 初始化Boid仿真 """
        # 初始化位置和速度
        # 位置初始化在画布中心附近, 加入一定的随机扰动
        self.pos = [width/2.0, height/2.0] + 10*np.random.rand(2*N).reshape(N, 2)
        # 随机生成角度, 初始化速度为单位向量
        angles = 2*math.pi*np.random.rand(N)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.N = N
        # 最小接近距离
        self.minDist = 25.0
        # 规则计算得到的最大速度
        self.maxRuleVel = 0.03
        # 最终速度的最大值
        self.maxVel = 2.0

    def tick(self, frameNum, pts, beak):
        """仿真推进一步, 更新所有Boid的位置和速度"""
        # 计算所有Boid之间的距离矩阵
        self.distMatrix = squareform(pdist(self.pos))
        # 应用三条规则, 更新速度
        self.vel += self.applyRules()
        # 限制速度不超过最大值
        self.limit(self.vel, self.maxVel)
        # 更新位置
        self.pos += self.vel
        # 应用边界条件
        self.applyBC()
        # 更新绘图数据
        pts.set_data(self.pos.reshape(2*self.N)[::2], 
                     self.pos.reshape(2*self.N)[1::2])
        vec = self.pos + 10*self.vel/self.maxVel
        beak.set_data(vec.reshape(2*self.N)[::2], 
                      vec.reshape(2*self.N)[1::2])

    def limitVec(self, vec, maxVal):
        """限制2D向量的大小"""
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag
    
    def limit(self, X, maxVal):
        """限制数组X中2D向量的大小不超过maxValue"""
        for vec in X:
            self.limitVec(vec, maxVal)
            
    def applyBC(self):
        """应用边界条件"""
        deltaR = 2.0
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = - deltaR
            if coord[0] < - deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = - deltaR
            if coord[1] < - deltaR:
                coord[1] = height + deltaR
    
    def applyRules(self):
        # 应用规则 #1 - 分离
        D = self.distMatrix < 25.0
        vel = self.pos*D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)

        # 不同的距离阈值
        D = self.distMatrix < 50.0

        # 应用规则 #2 - 对齐
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2;

        # 应用规则 #1 - 聚合
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3

        return vel

    def buttonPress(self, event):
        """matplotlib按钮按下事件的处理函数"""
        # 左键点击 - 添加一个boid
        if event.button is 1:
            self.pos = np.concatenate((self.pos, 
                                       np.array([[event.xdata, event.ydata]])), 
                                      axis=0)
            # 随机速度
            angles = 2*math.pi*np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis=0)
            self.N += 1 
        # 右键点击 - 散射
        elif event.button is 3:
            # 添加散射速度 
            self.vel += 0.1*(self.pos - np.array([[event.xdata, event.ydata]]))
        
def tick(frameNum, pts, beak, boids):
    #print frameNum
    """动画更新函数"""
    boids.tick(frameNum, pts, beak)
    return pts, beak

# main() function
def main():
  # use sys.argv if needed
  print('starting boids...')

  parser = argparse.ArgumentParser(description="Implementing Craig Reynold's Boids...")
  # add arguments
  parser.add_argument('--num-boids', dest='N', required=False)
  args = parser.parse_args()

  # number of boids
  N = 100
  if args.N:
      N = int(args.N)

  # create boids
  boids = Boids(N)

  # setup plot
  fig = plt.figure()
  ax = plt.axes(xlim=(0, width), ylim=(0, height))

  pts, = ax.plot([], [], markersize=10, 
                  c='k', marker='o', ls='None')
  beak, = ax.plot([], [], markersize=4, 
                  c='r', marker='o', ls='None')
  anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, boids), 
                                 interval=50)

  # add a "button press" event handler
  cid = fig.canvas.mpl_connect('button_press_event', boids.buttonPress)

  plt.show()

# call main
if __name__ == '__main__':
  main()