import data as data
import model as model
import time
import glob
import os
import data_collector as collector

# for emotion in model.emotions:
#     data.order_data(emotion)
# data.get_files_CNN()
# # print('preparing to create dataset, please wait...')
# # time.sleep(5.0)
# # for emotion in model.emotions:
# #     if collector.cap(emotion) == 'done':
# #         print('preparing next emotion, please wait...')
# #         time.sleep(5.0)
#
# print('preparing training data and test data...')
# data.initial_data()
# training_data, test_data = data.get_files()
# time.sleep(5.0)
#
# print('training model...')
# model.train(training_data)
#
# print('testing model...')
# model.test(test_data)
#
# print('preparing live emotion recognition...')
# time.sleep(5.0)
#
# # if model.detect_face('video') == 'close':
# #     if input('would you like to delete dataset? type: yes/no') == 'yes':
# #         files = glob.glob('dataset/*.jpg')
# #         for f in files:
# #             os.remove(f)
# #     else:
# #         print('goodbye')
# #         quit()
