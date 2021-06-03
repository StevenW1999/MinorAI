import model
import data_process
import data


train_x, train_y, val_x, val_y = data_process.process_data('dataset.csv', "KNN")
model.train(train_x, train_y)
model.test(val_x, val_y)
# print(model.euclidean_distance(train_x[0],data.emotion_data.iloc[0]['pixels']))
# for i, r in data.emotion_data.iterrows():
#     distance = ()
#     print(distance)
# print()
#
