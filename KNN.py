import numpy as np
import pandas as pd
import math

class KNN():
    def __init__(self,X,y,k=3):
        self.k = k
        self.X = X
        self.y = y

    def calculo(self,x):
        self.x = x
        #método de distância euclidiana
        self.d = np.array([[abs(self.x[j] - self.X[i,j])**2 for i in range(len(self.X))] for j in range(len(self.X[0]))]) #[abs(x[0] - X[i,0])**2 for i in range(len(X))]
        self.f = [[math.sqrt(sum(self.d[:,i])) for i in range(len(np.transpose(self.d)))],list(self.y)]#[sum(d[:,i]) for i in range(len(np.transpose(d)))]#
        self.dg = pd.DataFrame(self.f).T
        self.dgg = self.dg.sort_values([0]).iloc[:self.k,:]
        self.dgg['Prop']= self.dgg[0].apply(lambda x: x/(self.dgg[0].sum()))
        self.dgh = self.dgg[1].value_counts().idxmax() #dgg.groupby(1)['Prop'].sum().idxmax(),dgg.groupby(1)['Prop'].sum().max()
        return self.dgh

    def KNN_Score(self,x_teste,y_teste):
        self.x_teste = x_teste
        self.y_teste = y_teste
        sco = [self.calculo(self.x_teste[i]) for i in range(len(self.x_teste))]
        cont = 0
        for i in range(len(sco)):
            if sco[i] == self.y_teste[i]:
                cont += 1
        return 100*cont/len(sco)

    def KNN_predict(self,x_new):
        p = []
        for i in x_new:
            p.append(self.calculo(i))
        return p

df = pd.DataFrame({"Altura":[171,188,185,159,157,177,157,154,198,175,163,177],"Peso":[83,128,78,46,44,67,45,66,101,73,54,97],
                  "Sexo":[0,0,1,1,1,0,1,1,0,0,1,0]})

X = df.iloc[:,:2].values
y = df.iloc[:,2].values
X_new = np.array([[175,73],[159,46]])
x_teste = np.array([[192,103],[165,54],[149,37],[178,79],[182,88],[184,95]])
y_teste = [0,1,1,0,0,0]

g = KNN(X,y)
print(g.KNN_Score(x_teste,y_teste))
print(g.KNN_predict(X_new))
