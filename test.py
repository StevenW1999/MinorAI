# from collections import Counter
#
# import pandas as pd
# import numpy as np
#
#
# # happy_data = {'x': [1], 'y': [1], 'emotion': 'happy'}
# # sad_data = {'x': [1], 'y': [1],' emotion': 'happy'}
# # angry_data = {'x': [1], 'y': [1], 'emotion': 'happy'}
# # contempt_data = {'x': [1], 'y': [1], 'emotion': 'happy'}
# #
# # happy = pd.DataFrame(data=happy_data)
# # sad = pd.DataFrame(data=sad_data)
# # angry = pd.DataFrame(data=angry_data)
# # contempt = pd.DataFrame(data=contempt_data)
# #
# # t = pd.Series((happy, sad, angry, contempt), index=['happy', 'anger', 'contempt', 'sadness'])
# #
# # print(happy)
# #
# # t['happy'].loc[-1] = [1, 3]
# # happy.index = happy.index + 1
# # happy = happy.sort_index()
# # print(happy)
# distance = (1, 2)
# landmarks = [1,2]
# data = {'x': [], 'y': [], 'emotion': ''}
# t = pd.DataFrame(data=data)
# print(t)
# t.loc[-1] = [1,1, 'angry']
# t.index = t.index + 1
# t = t.sort_index()
# t.loc[-1] = [1,2, 'angry']
# t.index = t.index + 1
# t = t.sort_index()
# t.loc[-1] = [1,1, 'sad']
# t.index = t.index + 1
# t = t.sort_index()
# t.loc[-1] = [1,3, 'sad']
# t.index = t.index + 1
# t = t.sort_index()
# t.loc[-1] = [2,4, 'happy']
# t.index = t.index + 1
# t = t.sort_index()
#
# df2 = t.sort_values(by=['x', 'y'], ascending=[True, True], axis=0)[:5]
# counter = Counter(df2['emotion'])
# prediction = counter.most_common()[0][0]
# print(df2)
# if prediction == 'angry':
#
#     print('prediction correct')
# else:
#     print('false')
#
import numpy as np
import pandas as pd
import data_process
df = pd.read_csv('dataset.csv')
#
# emotions = pd.get_dummies(df['emotion']).to_numpy()

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

df2 = pd.get_dummies(df['emotion'])
df3 = undummify(df2)
print(df3)
