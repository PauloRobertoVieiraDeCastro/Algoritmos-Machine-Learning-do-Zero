import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

class RegressaoLogistica():
    def __init__(self,X,y,alfa=0.05):
        self.alfa = alfa
        self.X = X
        self.y = y

    def Logistic(self):
        self.y = np.matrix(self.y).T
        LL = [np.random.normal() for i in range(len(self.X.T))]
        self.beta = np.matrix(LL)
        self.N = len(self.y)
        for i in range(200):
            L = 1.0/(1.0 + np.exp(-np.dot(self.X,self.beta.T)))
            self.grad = np.dot(self.X.T,L-self.y)
            #print('g',self.grad,self.grad.shape)
            #print('b',self.beta,self.beta.shape)
            self.betanovo = self.beta - self.grad.T*self.alfa/self.N
            erro = np.sum(abs(self.betanovo - self.beta))
            if erro <=0.7:
                break
            self.beta = self.betanovo
        
        return self.beta

    def Logistic_predict(self,XX):
        self.k = 1.0/(1.0 + np.exp(-np.dot(XX,self.Logistic().T)))
        self.j = []
        for i in self.k:
            if i>0.5:
                self.j.append(1)
            else:
                self.j.append(0)
        return self.j

    def Logistic_score(self,X_teste,y_teste):
        z = self.Logistic_predict(X_teste)
        cont = 0
        for i in range(len(z)):
            if z[i] == y_teste[i]:
                cont += 1
        return 100*cont/len(z)

    

df = pd.DataFrame({"Altura":[171,188,185,159,157,177,157,154,198,175,163,177],"Peso":[83,128,78,46,44,67,45,66,101,73,54,97],
                  "Sexo":[0,0,1,1,1,0,1,1,0,0,1,0]})

X = df.iloc[:,:2].values
y = df.iloc[:,2].values
X_new = np.array([[175,73],[159,46]])
x_teste = np.array([[192,103],[165,54],[149,37],[178,79],[182,88],[184,95]])
y_teste = [0,1,1,0,0,0]
r = RegressaoLogistica(X,y)
print(r.Logistic_score(x_teste,y_teste))
