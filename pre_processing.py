# Kaggle Challenge Beforehand Data Pre-processing
# https://www.kaggle.com/competitions/dm2022-isa5810-lab2-homework
# ID: 110061542
# Name: Hsu, Wei-tung 許暐彤
# Date: 2022/11/17

import pandas as pd
from csv import DictWriter

import json

# read dataset
def readData(path = './kaggle dataset/tweets_DM.json'):
    file = open(path, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        dic = json.loads(line)
        data.append(dic)                                                            # turn json into list of dict
    return data

def saveCSV(file_path, data_df, first):
    if first == 1:
        data_df.to_csv(file_path, index=False)                                      # store 1st sample with header
    else:
        data_df.to_csv(file_path, mode='a', index=False, header=False)              # append to old csv

def pre_processing(data):
    id2train_test = pd.read_csv('./kaggle dataset/data_identification.csv', header=None, index_col=0, squeeze=True).to_dict()
                                                                                    # dict to separate training & testing
    id2label = pd.read_csv('./kaggle dataset/emotion.csv', header=None, index_col=0, squeeze=True).to_dict()
                                                                                    # dict to assign label for training set
    train_num = 0                                                                   # number of training sample
    test_num = 0                                                                    # number of testing sample
    for i in range(len(data)):                                                      # for every sample
        print("\r ", (i+1)*100 / len(data), "%% done                        ", end = '', flush = True)
        score = data[i]["_score"]
        index = data[i]["_index"]
        source_dic = data[i]["_source"]["tweet"]                                    # data type: dic
        hashtags = source_dic["hashtags"]                                           # data type: list

        tweet_id = source_dic["tweet_id"]
        text = source_dic["text"]                                                   # data type: str
        date = data[i]["_crawldate"]
        _type = data[i]["_type"]

        if id2train_test[tweet_id] == "train":                                      # for training sample
            train_num += 1                                                          # count number
            label = id2label[tweet_id]                                              # get label

            train_df = pd.DataFrame(
            [[tweet_id, text, label]], 
            columns=['tweet_id', 'text', 'label']
            )                                                                       # store id, text, and label
            saveCSV("./kaggle dataset/train_base.csv", train_df, train_num)         # save to csv file
  
        elif id2train_test[tweet_id] == "test":                                     # for testing sample
            test_num += 1                                                           # count number

            test_df = pd.DataFrame(
            [[tweet_id, text]], 
            columns=['id', 'text']
            )                                                                       # store id and text
            saveCSV("./kaggle dataset/test.csv", test_df, test_num)                 # save to csv file

    print("Total ", train_num, " training samples, and ", test_num, " testing samples.")
    
def read_data(file):
    data = []
    with open(file, 'r')as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())                # get label
            text = line[line.find("]")+1:].strip()                                  # get text
            data.append([label, text])                                              # put into list
    return data

def convert_label(item, name):                                                      # from one-hot to label
    items = list(map(float, item.split()))
    label = ""
    for idx in range(len(items)): 
        if items[idx] == 1: 
            label += name[idx] + " "
    
    return label.strip()

def add_data():
    # our training set has [joy, anticipation, trust, sadness, disgust, fear, surprise, anger] 8 classes
    train_df = pd.read_csv("./kaggle dataset/train_base.csv",lineterminator='\n')
    
    # 1. 'tweet_emotions' dataset
    # https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text
    # it has [neutral, worry, happiness, sadness, love, surprise, fun, relief, hate, empty, 
    # enthusiasm, boredom, anger] 13 classes
    tweet_emotions_df = pd.read_csv("./kaggle dataset/additional/tweet_emotions.csv")
    
    result_df = tweet_emotions_df[tweet_emotions_df["sentiment"] == "happiness"]
    result_df = result_df.rename(columns={'sentiment': 'label', 'content': 'text'}) # change col name
    result_df["label"] = "joy"                                                      # view happiness as joy
    
    emotion_lst = ["sadness", "surprise", "anger"]                                  # select emotions same as ours
    for emotion in emotion_lst:
        sub_df = tweet_emotions_df[tweet_emotions_df["sentiment"] == emotion]
        sub_df = sub_df.rename(columns={'sentiment': 'label', 'content': 'text'})   # change col name
        sub_df["label"] = emotion                                                   # assign emotion
        result_df = pd.concat([result_df, sub_df], axis=0)                          # join as 1 dataframe
    train_df = pd.concat([train_df, result_df], axis=0)                             # add to train_df
    
    # 2. train.txt, val.txt, test.txt
    # https://www.kaggle.com/code/chandrug/text-emotion-classification-neural-network
    # it has [joy, sadness, anger, fear, love, surprise] 6 class

    file_lst = ["train", "test", "val"]
    x = []
    y = []
    for file in file_lst:
        f = open('./kaggle dataset/additional/' + file + '.txt','r')

        for i in f:
            l = i.split(';')
            y.append(l[1].strip())                                                  # sum train, test, val as 1
            x.append(l[0])                                                          # sum train, test, val as 1
    data_df =pd.DataFrame({'text':x,'label':y})                                     # to dataframe
    
    result_df = pd.DataFrame()
    emotion_lst = ["joy", "sadness", "fear", "surprise", "anger"]                   # select emotions same as ours
    for emotion in emotion_lst:
        sub_df = data_df[data_df["label"] == emotion]
        result_df = pd.concat([result_df, sub_df], axis=0)
    train_df = pd.concat([train_df, result_df], axis=0)                             # add to train_df
    
    # 3. text.txt
    # https://www.kaggle.com/code/jarvis11/text-emotions-detection/notebook
    # it has ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"] 7 classes
    file = './kaggle dataset/additional/text.txt'
    data = read_data(file)
    
    emotions = ["joy", 'fear', "anger", "sadness", "disgust", "shame", "guilt"]
    X_all = []
    y_all = []
    for label, text in data:
        y_all.append(convert_label(label, emotions))                                # store labels
        X_all.append(text)                                                          # store text
    text_df = pd.DataFrame({'text':X_all,'label':y_all})                            # to dataframe
    
    result_df = pd.DataFrame()
    emotion_lst = ["joy", "sadness", "fear", "disgust", "anger"]                    # select emotions same as ours
    for emotion in emotion_lst:
        sub_df = text_df[text_df["label"] == emotion]
        result_df = pd.concat([result_df, sub_df], axis=0)
    train_df = pd.concat([train_df, result_df], axis=0)                             # add to train_df
    train_df.to_csv("./kaggle dataset/train_all.csv", index=False)
    
data = readData()                                                                   # Read the given dataset
pre_processing(data)                                                                # to csv
add_data()                                                                          # Add in additional data to help
print("Pre_processing done.Training data in './kaggle dataset/train_all.csv', and testing data in './kaggle dataset/test.csv'.")

