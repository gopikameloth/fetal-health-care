import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
with open(r"dt.pkl", "rb") as f:
   e = pickle.load(f)
df = pd.read_csv('fetal_health.csv')
df = df.dropna()
X = df.drop(columns='fetal_health')
x=[[120.0,0.0,0.0,0.0,0.0,0.0,0.0,73.0,0.5,43.0,2.4,64.0,62.0,126.0,2.0,0.0,120.0,137.0,121.0,73.0,1.0]]
X.loc[len(X.index)] = x[0]
X=scaler.fit_transform(X)
x=[X[-1]]
out=e.predict(x)
print(out)