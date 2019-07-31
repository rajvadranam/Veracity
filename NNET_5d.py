#!/usr/bin/python3
from __future__ import unicode_literals
import getopt
import itertools
import json
import linecache
import math
import os
import random
import string
import sys
import time
from collections import Counter
from datetime import datetime
from multiprocessing import Pool
# import matplotlib.pyplot as plt
import pandas as pd
import skfuzzy as fuzz
from afinn import Afinn
from joblib import Parallel, delayed
from nltk.corpus import stopwords
import numpy as np
senti = Afinn(emoticons=True)
trantab = "".maketrans('', '', string.punctuation)
import warnings
from gensim import corpora, models

'''The Function of this script to generate a N dimensional model out of the Tweets to cater the veracity model'''

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF
    from gensim import corpora, models
    from empath import Empath
    from collections import defaultdict
    from textblob import TextBlob
    from nltk.util import ngrams
    import re

'''Creating required objects such with global scope'''
senti = Afinn(emoticons=True)
trantab = "".maketrans('', '', string.punctuation)
sw = set(stopwords.words('english'))
lexicon = Empath()
com = defaultdict(lambda: defaultdict(int))


def PrintException():
    """
        A function to funnel all app thrown Errors in a nice format with line number and stact trace as such
    """
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def spinner(n):
    """
     A typical animation string appears at the start of the  program shows the script processing animation
    :param n: takes in number of times the animation should be performed
    """
    syms = ['\\', '|', '/', '-']
    sys.stdout.write("processing...")
    for _ in range(n):
        for sym in syms:
            sys.stdout.write("\b%s" % sym)
            sys.stdout.flush()
            time.sleep(.015)


def get_afinn_scores(line):
    """
     The function of this method is to segrigate a each word not positive negative and neutral and passing scores for each category
    :param line: Tweet text
    :return: Dictionary of postive negative and neutral values for given tweet
    """
    scores = {}
    scores['positive'] = 0
    scores['neutral'] = 0
    scores['negative'] = 0
    terms = line.split()
    for x in terms:
        s = int(senti.score(x))
        if s > 0:
            scores['positive'] += s
        elif s == 0:
            scores['neutral'] = 0
        else:
            scores['negative'] += s
    return scores


def clean(line, status=False):
    """
     The function of this method is that it cleans the unnecessary words and the objects for tweeet text
    :param line: Tweet Text
    :param status: True / False
    :return: if passed a true it will return a cleaned string else a list of cleaned words will be returned
    """
    lowerlist = [x.lower().translate(trantab) for x in line.split() if x.lower() not in sw]
    lowerlist = [''.join(e for e in x if e.isalnum()) for x in lowerlist if
                 x is not "" and len(x.strip()) > 2 and not x.isdigit() and 'https' not in x.lower()]
    if (status):
        ss = ' '.join(e for e in lowerlist)
        lowerlist = ss
    return lowerlist


def scale(OldList, NewMin, NewMax):
    """
Use this method if you want to scale between 0-1
    :param OldList: list of values
    :param NewMin: 0
    :param NewMax: 1
    :return: scaled values between 0-1
    """
    x = OldList
    NewRange = float(NewMax - NewMin)
    OldMin = min(x)
    OldMax = max(x)
    OldRange = float(OldMax - OldMin)
    NewList = []
    aList = []
    if OldRange != 0.0:
        ScaleFactor = NewRange / OldRange
        print('\nEquaTion:  NewValue = ((OldValue - ' + str(OldMin) + ') x ' + str(ScaleFactor) + ') + ' + str(
            NewMin) + '\n')
        for OldValue in OldList:
            NewValue = ((OldValue - OldMin) * ScaleFactor) + NewMin
            NewList.append(int(NewValue))
    else:
        NewList = [0 for x in range(len(OldList))]
    return NewList


def Find(string):
    """
     This is a regex approach to find the URL and mentions in Tweet text
    :param string: Tweet text uncleaned
    :return: Tweet text Cleaned without mentions and Urls
    """
    # findall() has been used
    # with valid conditions for urls in string
    completestring = ""
    url = re.findall('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', string)
    if len(url) > 0:
        for u in url:
            completestring = string.replace(u, "")
            string = completestring
    else:
        completestring = string

    prefix = ('@')
    stringfix = ""
    mylist = completestring.split()
    for w in mylist:
        for s in w[:]:
            if s.startswith(prefix):
                mylist.remove(s)
    return " ".join(mylist)

# definition to go through and read each tweet pulling out what we need
def Read_ownerTweets(file):
    """
        function to read Json and creates the data points for N-D model and OTC components into a numeric form
       :param file: filename containing flume json strings
       :return: a list of caliculated N values [V1,V2, V3, V4, V5] and OTC values [OTCnorm, recp]
       """
    global x, y, totalTweets, ownertweets
    fileObject = open(file, encoding="utf8")
    Lines = fileObject.readlines()
    totalTweets = len(Lines)
    for line in Lines:
        try:
            parsed_json_tweets = json.loads(line)
            if 'retweeted_status' in parsed_json_tweets:
                ownertweets += 1
                ownerName = parsed_json_tweets['retweeted_status']['user'][
                    'screen_name'].lstrip().strip()
                ownerTweetTimeStamp = parsed_json_tweets['retweeted_status'][
                    'created_at'].lstrip().strip()
                ownerFollercount = parsed_json_tweets['retweeted_status']['user']['followers_count']
                ownerretweetcount = parsed_json_tweets['retweeted_status']['retweet_count']
                try:
                    Owner_tweet_text = parsed_json_tweets['retweeted_status']['extended_tweet'][
                        'full_text'].lstrip().strip()
                except:
                    Owner_tweet_text = parsed_json_tweets['retweeted_status'][
                        'text'].lstrip().strip()

                # Step 1
                # calculate V1 ( retweet count - followercount of owner(original tweets))
                V1 = 0
                if (int(ownerretweetcount) > 0):
                    V1 = (int(ownerretweetcount) - int(ownerFollercount)) / int(ownerretweetcount)

                # splitting into words for each word
                # Step 2

                '''Calculate sentiment for each word in TWEET AND MAKE SENTIMENT ANALYSIS'''

                score = get_afinn_scores(Owner_tweet_text)
                V2 = int(score['positive'])
                V3 = int(score['negative'])
                V4 = int(score['neutral'])
                # Step 3

                '''ENTROPY STEPS Pi Log Pi'''
                wordLength = len(clean(Owner_tweet_text))
                EachWordCount = Counter.__call__(clean(Owner_tweet_text))
                p = []
                for x, y in EachWordCount.items():
                    pi = EachWordCount[x] / wordLength
                    p.append(pi * float(math.log(pi, 2)))
                V5 = 0
                for x in p:
                    V5 += x
                V5 = -(V5)
                if (V1 < 0):
                    V1 = 0
                OriginaltweetMap[ownerName + "," + ownerTweetTimeStamp] = [ownerName, ownerTweetTimeStamp,
                                                                           ownerFollercount, ownerretweetcount,
                                                                           Owner_tweet_text.replace('\n', ''), V1,
                                                                           V2, V3, V4, V5]
        except ValueError:
            continue
    return totalTweets, OriginaltweetMap


def DoLDAGetWeights(Alltweets):
    """
       Do Topic modelling using LDA with all Tweets
       :param Alltweets: List of all Tweets
       :return: List of Dict of words and model with Weights
       """
    ldamodel = None
    dicta = {}
    dictionary = corpora.Dictionary(Alltweets)
    corpus = [dictionary.doc2bow(text) for text in Alltweets]
    try:
        ldamodel = models.LdaMulticore(corpus, id2word=dictionary, passes=1, minimum_probability=0, chunksize=10000)
    except:
        pass
    tweetsModel = {}
    for i in range(len(corpus)):
        adder = 0
        for f in ldamodel[corpus[i]]:
            try:
                # print(f)
                if dictionary.id2token.get(f[0]) not in dicta.keys():
                    dicta[dictionary.id2token.get(f[0])] = f[1]
                    adder += f[1]
                else:
                    dicta[dictionary.id2token.get(f[0])] = f[1]
            except:
                adder += f[1]

    for i in range(len(Alltweets)):
        sdder = 0
        for wo in Alltweets[i]:
            try:
                sdder += dicta.get(wo)
            except:
                pass
        tweetsModel[i] = sdder
    return [dicta, tweetsModel]


def normaliz(d):
    # d is a (n x dimension) np array
    # d /=d.sum(axis=0, keepdims=True)
    s = np.sum(d, axis=0)
    return d / s


folder = ""


def main():
    try:
        TestFilename, clusterAlg, comparebothclusters, folder, nmf, outputFolder, outputfile, percentage, fitmode, pday = GetArgs()
        try:
            # variables
            modelNmf = None
            normalize = True
            roundFactor = 1000
            scaler = None
            # if outputfolder not given making sure to dump these parsed data in input folder
            if outputFolder == "":
                outputFolder = folder
            # get perday ?
            # if "y" in pday.lower():
            #     perday.PerDaySentiment(folder, outputFolder)
            TweetMapper, FiveDModel, TotalTweetCount, WordWeights = TweetExtractor(folder)
            random.seed(20)
            features = ['NFRQ', 'Positive', 'Negative', 'Shannon Entropy', 'LDA Weights']
            df1, df1Norm, scaler = GetNormalizedDF(FiveDModel, features, normalize)
            nclusters = int(Clusters)
            data1 = df1Norm.values.reshape(df1Norm.values.shape[1], df1Norm.values.shape[0])

            data1, df21norm, modelNmf = isUseNmf(data1, df1, df1Norm, features, modelNmf, nmf, normalize, outputFolder,
                                                scaler)
            # data1 = df21norm.values.reshape(df21norm.values.shape[1], df21norm.values.shape[0])
            df1Norm = df21norm
            AllCluster_Values, Cluster_Values, Kmeanslabels, cluster_maximum_ineces, cntr, means,Trained_labels = GetClusterValues(
                clusterAlg, comparebothclusters, data1, df1Norm, nclusters)
            weights = [row[4] for row in FiveDModel]
            WriteModelToFile(AllCluster_Values, Cluster_Values, Kmeanslabels, TweetMapper, cluster_maximum_ineces,
                             df1Norm, outputFolder, outputfile, weights)
            if (int(percentage) < 40):
                prop = float(int(percentage) / 100)
                # spliting the data

                clf, predicted_cmeans, x_Test, x_Train, y_Test, y_Train = TrainNetClassifier(Trained_labels, clusterAlg,
                                                                                             df1Norm, fitmode,
                                                                                             nclusters, prop)
                # experimental
                predicted = PredictionMode(clf, clusterAlg, predicted_cmeans, x_Test)
                DecideTypeOfTest(TestFilename, clf, clusterAlg, cntr, features, folder, means, modelNmf, nmf, normalize,
                                 outputFolder, outputfile, roundFactor, scaler)
                differencelist = [i == j for i, j in zip(y_Test, predicted)]
                WritePredictionResultstoFile(FiveDModel, TestFilename, TotalTweetCount, Trained_labels, TweetMapper,
                                             differencelist, nclusters, outputFolder, outputfile, predicted, x_Train,
                                             y_Test, y_Train)
            else:
                print("The Test data proportion should be less than 40")
                sys.exit(0)
        except:
            e_type, e_value, e_traceback = sys.exc_info()
            PrintException()
            print("The path specifed is not valid: " + folder)
    except:
        PrintException()
        usage()


def WritePredictionResultstoFile(FiveDModel, TestFilename, TotalTweetCount, Trained_labels, TweetMapper, differencelist,
                                 nclusters, outputFolder, outputfile, predicted, x_Train, y_Test, y_Train):
    fOriginal1 = open(outputFolder + outputfile + "_indata_Predicted_" + datetime.now().strftime(
        '%Y-%m-%d-%H-%M') + ".tsv",
                      encoding="utf8", mode='w')
    fOriginal1.write(
        "{:<18}\t {:<15}\t {:<15}\t {:<15} \t{:<17} \t{:<15} \t{:<15} \t{:<21}\n".format('V1', 'V2', 'V3',
                                                                                         'V4', 'V5',
                                                                                         'Labels',
                                                                                         'Predicted_Labels',
                                                                                         'Predicted is Equal?'))
    for x, y, z, w, in zip(FiveDModel[len(y_Train):len(Trained_labels)], y_Test, predicted,
                           differencelist):
        fOriginal1.write(
            "{:<18}\t {:<15}\t {:<15}\t {:<15} \t{:<17} \t{:<15} \t{:<15} \t{:<21}\n".format(x[0], x[1],
                                                                                             x[2], x[3],
                                                                                             x[4],
                                                                                             y, z,
                                                                                             w.__str__()))
    fOriginal1.write("Total Tweets: " + TotalTweetCount.__str__() + "\n")
    fOriginal1.write("owner Tweets: " + len(TweetMapper).__str__() + "\n")
    fOriginal1.write("no of tweets in train phase: " + len(x_Train).__str__() + "\n")
    fOriginal1.write("no of tweets in test phase: " + len(y_Test).__str__() + "\n")
    fOriginal1.write("no of k-Means Cluster : " + nclusters.__str__() + "\n")
    # print("score : "+clf.score(X_train1,predicted1).__str__())
    fOriginal1.write("Accuracy of predicted values with in data: " + (
            accuracy_score(y_Test, predicted) * 100).__str__() + " %" + "\n")
    print("Accuracy of predicted values with in data: " + (
            accuracy_score(y_Test, predicted) * 100).__str__() + " %" + "\n")
    if (TestFilename.__str__().lower() != 'na' and TestFilename != ""):
        fOriginal1.write("File saved to the " + outputfile + "_indata_Predicted" + datetime.now().strftime(
            '%Y-%m-%d-%H-%M') + ".tsv with predicted values" + "\n")
    fOriginal1.write("confusion Matrix:" + "\n")
    for data_slice in confusion_matrix(y_Test, predicted):
        np.savetxt(fOriginal1, data_slice, fmt='%-7.2f')
    fOriginal1.write("\n" + "Classification report" + "\n")
    fOriginal1.write(classification_report(y_Test, predicted))
    fOriginal1.close()


def DecideTypeOfTest(TestFilename, clf, clusterAlg, cntr, features, folder, means, modelNmf, nmf, normalize,
                     outputFolder, outputfile, roundFactor, scaler):
    if '.' in TestFilename:
        if (TestFilename.__str__().lower() != 'na' and TestFilename != ""):
            try:
                df_neg = pd.read_excel(TestFilename)
                X_train1 = df_neg.abs()
                # X_train1 =df_neg
                if normalize:
                    X_test21 = normaliz(X_train1)
                else:
                    X_test21 = X_train1
                predicted1 = clf.predict(X_test21)
                testPredictedSinglefile = []
                if 'cmeans' in clusterAlg or 'c-means' in clusterAlg or 'fuzzy' in clusterAlg:
                    data2a = df_neg.values.reshape(df_neg.values.shape[1], df_neg.values.shape[0])
                    u, u0, d, jm, p, fpc = fuzz.cmeans_predict(data2a, cntr, 2, error=0.005, maxiter=1500,
                                                               init=None)
                    Cluster_Values2 = getMaximumCmeans(u)
                    AllCluster_Values2 = getMaximumCmeans(u, True)
                    temp = [int(round(y * roundFactor)) for y in Cluster_Values2]
                    # testPredictedSinglefile = np.argmax(u, axis=0)
                    testPredictedSinglefile = temp
                else:
                    testPredictedSinglefile = means.predict(df_neg)
            except:
                print(
                    "The alternate test " + TestFilename + " Doesnot exist in the provided folder,please make sure to have the  file in : " + folder + TestFilename)
                sys.exit(-1)
            Fpredicted = open(
                outputFolder + '_Test_' + outputfile + "_predicted_" + datetime.now().strftime(
                    '%Y-%m-%d-%H-%M') + ".tsv", mode='w')
            Fpredicted.write(
                "{:<18}\t {:<15}\t {:<15}\t {:<15} \t{:<17} \t{:<15} \n".format('V1', 'V2', 'V3',
                                                                                'V4', 'V5',
                                                                                'Predicted_Label',
                                                                                ))
            for values, ja in itertools.zip_longest(df_neg.values, predicted1):
                Fpredicted.write(
                    "{:<18}\t {:<15}\t {:<15}\t {:<15} \t{:<17} \t{:<15} \n".format(values[0], values[1],
                                                                                    values[2], values[3],
                                                                                    values[4], ja))
                Fpredicted.write(
                    "Accuracy of predicted values with trained clustering model and trained MLPClassifier: " + (
                            accuracy_score(testPredictedSinglefile,
                                           predicted1) * 100).__str__() + " %" + "\n")
            Fpredicted.close()
    else:
        try:
            testTweetMapper, testFiveDModel, testTotalTweetCount, testTweetsonly = TweetExtractor(
                TestFilename)

            df3 = pd.DataFrame(testFiveDModel, columns=features)
            df3 = df3.abs()
            if 'y' in nmf.lower():
                X_test21a = NMf(df3, modelNmf)
            else:
                X_test21a = df3.values
            if normalize:
                X_test21a = scaler.transform(X_test21a)
                # X_test21a = normaliz(X_test21a)

            predicted12 = clf.predict(X_test21a)
            testPredicted = []
            df2Norma = pd.DataFrame(X_test21a, columns=features)
            if 'cmeans' in clusterAlg or 'c-means' in clusterAlg or 'fuzzy' in clusterAlg:

                data2 = df2Norma.values.reshape(df2Norma.values.shape[1], df2Norma.values.shape[0])

                utest, u0, d, jm, p, fpc = fuzz.cmeans_predict(data2, cntr, 2, error=0.005, maxiter=1500,
                                                               init=None)
                testPredicted = np.argmax(utest, axis=0)
            else:
                testPredicted = means.predict(df2Norma)
            Fpredicted12 = open(
                outputFolder + '_Test_' + outputfile + "_predicted_" + datetime.now().strftime(
                    '%Y-%m-%d-%H-%M') + ".tsv", encoding="utf8", mode='w')
            Fpredicted12.write(
                "{:<20}\t {:<14}\t {:<12}\t {:<250} \t{:<12} \t{:<12} \t{:<12} \t{:<12} \t{:<12}\t{:<15}\n".format(
                    'userName', 'follower count', 'retweetcount', 'tweet',
                    'V1', 'V2', 'V3', 'V4', 'V5', 'predicted label'
                ))
            for y, w, d in itertools.zip_longest(testTweetMapper.values(), testFiveDModel, predicted12):
                Fpredicted12.write(
                    "{:<20}\t {:<14}\t {:<12}\t {:<250} \t{:<12} \t{:<12} \t{:<12} \t{:<12} \t{:<12}\t{:<15}\n".format(
                        y[0], y[2], y[3], y[4], w[0], w[1], w[2],
                        w[3], w[4], d))
            Fpredicted12.write(
                "Accuracy of predicted values with trained clustering model and trained MLPClassifier: " + (
                        accuracy_score(testPredicted, predicted12) * 100).__str__() + " %" + "\n")
            print(
                "Accuracy of predicted values with trained clustering model and trained MLPClassifier: " + (
                        accuracy_score(testPredicted, predicted12) * 100).__str__() + " %" + "\n")
            Fpredicted12.close()
        except:
            PrintException()
            pass


def PredictionMode(clf, clusterAlg, predicted_cmeans, x_Test):
    if 'cmeans' in clusterAlg or 'c-means' in clusterAlg or 'fuzzy' in clusterAlg:
        predicted = predicted_cmeans
    else:
        predicted = clf.predict(x_Test)
    return predicted


def TrainNetClassifier(Trained_labels, clusterAlg, df1Norm, fitmode, nclusters, prop):
    x_Train, x_Test, y_Train, y_Test = train_test_split(df1Norm.values, Trained_labels, test_size=prop,
                                                        shuffle=False)
    Features = len(x_Train[0])
    clf = MLPClassifier(hidden_layer_sizes=(Features, Features, Features), max_iter=500, alpha=0.001,
                        solver='adam', random_state=21, tol=0.00000001)
    predicted_cmeans = []
    classesT = [i for i in range(nclusters)]
    if fitmode.lower() == 'full' or fitmode == "":
        try:
            # experimental
            if 'cmeans' in clusterAlg or 'c-means' in clusterAlg or 'fuzzy' in clusterAlg:

                clf = clf.fit(x_Train, y_Train)
                predicted_cmeans = clf.predict(x_Test)
            else:
                clf.fit(x_Train, y_Train)
        except:
            clf.fit(x_Train, y_Train)
    else:
        # experimental
        for xa, ya in zip(x_Train, y_Train):
            varw = ya.reshape(1, )
            aarw = xa.reshape(1, -1)
            clf._partial_fit(aarw, varw, classes=classesT)
    return clf, predicted_cmeans, x_Test, x_Train, y_Test, y_Train


def WriteModelToFile(AllCluster_Values, Cluster_Values, Kmeanslabels, TweetMapper, cluster_maximum_ineces, df1Norm,
                     outputFolder, outputfile, weights):
    if (outputfile.lower() != 'na'):
        fOriginal = open(outputFolder + outputfile + "_" + datetime.now().strftime('%Y-%m-%d-%H-%M') + ".tsv",
                         encoding="utf8", mode='w')
        fOriginal.write(
            "{:<20}\t {:<14}\t {:<12}\t {:<150} \t{:<5} \t{:<5} \t{:<5} \t{:<5} \t{:<12} \t{:<14}\t{:<14}\t{:<14}\t{:<12} \t{:<19}\t{:<12}\t{:<13}\t{:<13}\t{:<13}\t{:<13}\t{:<13}\n".format(
                'userName', 'follower count', 'retweetcount', 'tweet',
                'V1', 'V2', 'V3', 'V4', 'V5',
                'Kmean_Labels', 'fuzzy cluster1', 'fuzzy cluster2', 'fuzzy cluster3',
                'Fuzzy cluster maximum value', 'Fuzzy label', 'Normalized-V1', 'Normalized-V2', 'Normalized-V3',
                'Normalized-V4', 'Normalized-V5',
            ))
        for y, w, x, z, e, d, ha in itertools.zip_longest(TweetMapper.values(), weights, Kmeanslabels,
                                                          Cluster_Values, cluster_maximum_ineces,
                                                          AllCluster_Values, df1Norm.values):
            fOriginal.write(
                "{:<20}\t {:<14}\t {:<12}\t {:<150} \t{:<5} \t{:<5} \t{:<5} \t{:<5} \t{:<12} \t{:<14}\t{:<14}\t{:<14}\t{:<12} \t{:<19}\t{:<12}\t{:<13}\t{:<13}\t{:<13}\t{:<13}\t{:<13}\n".format(
                    y[0], y[2], y[3], y[4], y[5], y[6], y[7],
                    w, y[9], x, d[0], d[1], d[2], z, e, ha[0], ha[1], ha[2], ha[3], ha[4]))
        fOriginal.close()


def GetClusterValues(clusterAlg, comparebothclusters, data1, df1Norm, nclusters):
    cntr = None
    means=None
    if 'cmeans' in clusterAlg or 'c-means' in clusterAlg or 'fuzzy' in clusterAlg:
        cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data1, nclusters, 2, error=0.005, maxiter=1500)

        # get maximum value for each
        Cluster_Values = getMaximumCmeans(u)
        AllCluster_Values = getMaximumCmeans(u, True)
        cluster_maximum_ineces = np.argmax(u, axis=0)
        Kmeanslabels = ['NA' for x in range(data1.shape[1])]
        if 'b' in comparebothclusters or 'c' in comparebothclusters:
            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data1, nclusters, 2, error=0.005, maxiter=1500)
            Cluster_Values = getMaximumCmeans(u)
            AllCluster_Values = getMaximumCmeans(u, True)
            cluster_maximum_ineces = np.argmax(u, axis=0)
            means = KMeans(n_clusters=nclusters).fit(df1Norm)
            Kmeanslabels = means.labels_
    elif 'kmeans' in clusterAlg or 'k-means' in clusterAlg or 'means' in clusterAlg:
        means = KMeans(n_clusters=nclusters).fit(df1Norm)
        Kmeanslabels = means.labels_
        Cluster_Values = ['NA' for x in range(data1.shape[1])]
        AllCluster_Values = [data1.shape[0] * ['NA'] for x in range(data1.shape[1])]
        cluster_maximum_ineces = ['NA' for x in range(data1.shape[1])]
        if 'b' in comparebothclusters or 'c' in comparebothclusters:
            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data1, nclusters, 2, error=0.005, maxiter=1500)
            Cluster_Values = getMaximumCmeans(u)
            AllCluster_Values = getMaximumCmeans(u, True)
            cluster_maximum_ineces = np.argmax(u, axis=0)
            means = KMeans(n_clusters=nclusters).fit(df1Norm)
            Kmeanslabels = means.labels_
    else:
        means = KMeans(n_clusters=nclusters).fit(df1Norm)
        Kmeanslabels = means.labels_
        Cluster_Values = ['NA' for x in range(data1.shape[1])]
        AllCluster_Values = [data1.shape[0] * ['NA'] for x in range(data1.shape[1])]
        cluster_maximum_ineces = ['NA' for x in range(data1.shape[1])]
        if 'b' in comparebothclusters or 'c' in comparebothclusters:
            cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(data1, nclusters, 2, error=0.005, maxiter=1500)
            Cluster_Values = getMaximumCmeans(u)
            AllCluster_Values = getMaximumCmeans(u, True)
            cluster_maximum_ineces = np.argmax(u, axis=0)
            means = KMeans(n_clusters=nclusters).fit(df1Norm)
            Kmeanslabels = means.labels_
    if 'cmeans' in clusterAlg or 'c-means' in clusterAlg or 'fuzzy' in clusterAlg:
        Trained_labels = cluster_maximum_ineces
    elif 'kmeans' in clusterAlg or 'k-means' in clusterAlg or 'means' in clusterAlg:
        Trained_labels = means.labels_
    else:
        Trained_labels = means.labels_
    return AllCluster_Values, Cluster_Values, Kmeanslabels, cluster_maximum_ineces, cntr, means,Trained_labels


def isUseNmf(data1, df1, df1Norm, features, modelNmf, nmf, normalize, outputFolder, scaler):
    if 'y' in nmf.lower():
        # NMF region
        datafrm = df1
        datafrm = datafrm.abs()
        data2 = datafrm.values.reshape(datafrm.values.shape[1], datafrm.values.shape[0])
        modelNmf = NMF(n_components=data2.shape[0], init='random', random_state=0)
        W = modelNmf.fit_transform(data2)
        Wdf = pd.DataFrame(W)
        Wdf.to_csv(outputFolder + 'W_matrix.tsv', sep='\t', encoding='utf-8')
        H = np.array(modelNmf.components_).transpose()
        df21 = pd.DataFrame(H, columns=features)
        if normalize:
            df21norm = scaler.transform(df21.values)
            # df21norm =  normaliz(df21.values)
        else:
            df21norm = df21
        df21norm = pd.DataFrame(df21norm, columns=features)
        df21norm = df21norm.abs()
    else:
        df21norm=df1Norm
    return data1, df21norm, modelNmf


def GetNormalizedDF(FiveDModel, features, normalize):
    scaler = None
    df1 = pd.DataFrame(FiveDModel, columns=features)
    df1 = df1.abs()
    if normalize:
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(df1.values)
        dfnorm = scaler.transform(df1.values)
        # dfnorm =  normaliz(df1.values)
        df1Norm = pd.DataFrame(dfnorm, columns=features)
    else:
        df1Norm = df1
    return df1, df1Norm, scaler


def GetArgs():
    global Clusters
    outputfile = ""
    percentage = 0.0
    TestFilename = ""
    fitmode = ""
    pday = ""
    clusterAlg = ""
    outputFolder = ""
    nmf = ""
    comparebothclusters = ""
    try:

        opts, args = getopt.getopt(sys.argv[1:], "a:c:d:f:h:i:n:o:p:t:u:v:", ['clusterAlgo=','kClusters=','perday=','fit=','help=','iFolder=',
                                                                              'nmf=','otweetsFile=',
                                                                              'percentsage=',
                                                                              'tAlternate=',
                                                                              'outputFolder=','compareboth='
                                                                              ])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit()
            elif opt in ("-i", "--iFolder"):
                folder = arg.replace('"','') + os.path.sep
            elif opt in ("-o", "--otweetsFile"):
                outputfile = arg
            elif opt in ("-p", "--percentsage"):
                percentage = arg
            elif opt in ("-t", "--tAlternate"):
                if '.' not in arg:
                    TestFilename = arg + os.path.sep
                else:
                    TestFilename = arg
            elif opt in ("-c", "--kClusters"):
                Clusters = arg
            elif opt in ("-a", "--clusterAlgo"):
                clusterAlg = arg
            elif opt in ("-f", "--fit"):
                fitmode = arg
            elif opt in ("-d", "--perday"):
                pday = arg
            elif opt in ("-n", "--nmf"):
                nmf = arg
            elif opt in ("-v", "--compareboth"):
                comparebothclusters = arg
            elif opt in ("-u", "--outputFolder"):
                outputFolder = arg.replace('"','') + os.path.sep
    except:
        PrintException()
    return TestFilename, clusterAlg, comparebothclusters, folder, nmf, outputFolder, outputfile, percentage, fitmode, pday


def NMf(df3, modelNmf):
    X_test21a = df3
    dataa2 = X_test21a.values.reshape(X_test21a.values.shape[1], X_test21a.values.shape[0])
    W = modelNmf.fit_transform(dataa2)
    Ha = np.array(modelNmf.components_).transpose()
    X_test21a = Ha
    return X_test21a


def getMaximumCmeans(u, getAllProb=False):
    Cluster_Values = []
    cou = 0
    temp = []
    if getAllProb:
        temp = np.transpose(u)
        Cluster_Values = temp
    else:
        for n in np.argmax(u, axis=0):
            if cou == 0:
                temp = np.transpose(u)
            tt = temp[cou][n]
            cou += 1
            Cluster_Values.append(tt)
    return Cluster_Values


def TweetExtractor(folder):
    TweetMapper = {}
    TotalTweetCount = 0
    TweetsOnly = []
    FiveDModel = []
    onlyfiles = [val for sublist in [[os.path.join(i[0], j) for j in i[2]] for i in os.walk(folder.__str__())] for val
                 in sublist if 'Flum' in val]
    # onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and f.startswith('FlumeD')]
    temp = Parallel(n_jobs=-1)(delayed(Read_ownerTweets)(file) for file in onlyfiles)
    for x in temp:
        TotalTweetCount += x[0]
        for k, v in x[1].items():
            TweetMapper[k] = v
    for x, y in TweetMapper.items():
        TweetsOnly.append(clean(y[4]))
    invdividualWordWeigts, addedDict = DoLDAGetWeights(TweetsOnly)
    for y, z in itertools.zip_longest(TweetMapper.values(), addedDict.values()):
        FiveDModel.append([y[5], y[6], y[7], y[9], z])
    return (TweetMapper, FiveDModel, TotalTweetCount, invdividualWordWeigts)


def usage():
    print(100 * " ")
    print(5 * " " + "*************** Usage Example :: *****************", end='\n')
    print(100 * " ")
    print(
        "NNET.py -i {Input folder Path} -o {output file Name} -p {percentage of Test data} -c {Kmeans Clusters number}")
    print(100 * " ")
    print(5 * " " + " *************** Mandatory Inputs :: ***************** ", end='\n')
    print(100 * " ")
    print(
        "NNET.py -i {Input folder Containing Jsons -> if contains spaces need to provide in quotes  Ex:- C:\\Users\\OSU user\\Desktop\\Tweets\\NewTweets }  ",
        end='\t')
    print("-o {name of the output file} -p {percentage} -c {KmeansClustersCount} ")
    print(100 * " ")
    print(5 * " " + "*************** Optional parameters :: ***************** \n { -t {otherTestFile} }")


RequiredList = []
OriginaltweetMap = {}
totalTweets = 0
ownertweets = 0
if __name__ == "__main__":
    # folder = 'C:\\Users\\OSU user\\Desktop\\NRA_Tweets\\Backup\\'
    # folder = 'C:\\Users\\OSU user\\Desktop\\Tweets\\NewTweets\\'
    # folder = ' /home/stdemin/tweets/'
    t0 = time.time()
    pool = Pool(processes=3)
    resutlt1 = pool.apply_async(spinner, [550])
    main()
    pool.close()
    print("Time Taken :", end="\t")
    print(time.time() - t0)
    sys.exit(0)
