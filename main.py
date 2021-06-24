import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset.csv")
df = pd.DataFrame(data=data)
pixels = df["pixels"].str.split()

count = 0
for i in pixels:
    ci = 0
    for b in i:
        i[ci] = int(b)
        ci += 1
    pixels[count] = i
    count += 1

x = pixels.tolist()
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf = SVC(kernel='poly')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))