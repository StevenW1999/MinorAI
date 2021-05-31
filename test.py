from collections import Counter

import pandas as pd
import numpy as np


# happy_data = {'x': [1], 'y': [1], 'emotion': 'happy'}
# sad_data = {'x': [1], 'y': [1],' emotion': 'happy'}
# angry_data = {'x': [1], 'y': [1], 'emotion': 'happy'}
# contempt_data = {'x': [1], 'y': [1], 'emotion': 'happy'}
#
# happy = pd.DataFrame(data=happy_data)
# sad = pd.DataFrame(data=sad_data)
# angry = pd.DataFrame(data=angry_data)
# contempt = pd.DataFrame(data=contempt_data)
#
# t = pd.Series((happy, sad, angry, contempt), index=['happy', 'anger', 'contempt', 'sadness'])
#
# print(happy)
#
# t['happy'].loc[-1] = [1, 3]
# happy.index = happy.index + 1
# happy = happy.sort_index()
# print(happy)
distance = (1, 2)
landmarks = [1,2]
data = {'x': [], 'y': [], 'emotion': ''}
t = pd.DataFrame(data=data)
print(t)
t.loc[-1] = [1,1, 'angry']
t.index = t.index + 1
t = t.sort_index()
t.loc[-1] = [1,2, 'angry']
t.index = t.index + 1
t = t.sort_index()
t.loc[-1] = [1,1, 'sad']
t.index = t.index + 1
t = t.sort_index()
t.loc[-1] = [1,3, 'sad']
t.index = t.index + 1
t = t.sort_index()
t.loc[-1] = [2,4, 'happy']
t.index = t.index + 1
t = t.sort_index()

df2 = t.sort_values(by=['x', 'y'], ascending=[True, True], axis=0)[:5]
counter = Counter(df2['emotion'])
prediction = counter.most_common()[0][0]
print(df2)
if prediction == 'angry':

    print('prediction correct')
else:
    print('false')

