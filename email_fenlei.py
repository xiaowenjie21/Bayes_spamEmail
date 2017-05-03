#贝叶斯实现邮件分类
import os, re, jieba
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def get_word_list(content, wordsList, stopList, file_words_list):
    # 分词结果放入res_list
    res_list = list(jieba.cut(content))
    for i in res_list:
        if i not in stopList and i.strip() != '' and i != None:
            if i not in wordsList:
                wordsList.append(i)


def getStopWords():
    stopList=[]
    for line in open("data/中文停用词表.txt", encoding='gbk'):
        stopList.append(line[:len(line)-1])
    return stopList

stopList = getStopWords()

def WordsList():

    normal_files = os.listdir("data/normal")
    spam_files = os.listdir("data/spam")

    wordsList = []
    file_words_list = []
    spam_words_list = []


    for f in normal_files:
        wordsList.clear()
        for line in open("data/normal/" + f):
            rule = re.compile(r"[^\u4e00-\u9fa5]")
            line = rule.sub("", line)
            get_word_list(line, wordsList, stopList, file_words_list)

        file_words_list.append(' '.join(wordsList))

    for f in spam_files:
        wordsList.clear()
        for line in open("data/spam/" + f):
            rule = re.compile(r"[^\u4e00-\u9fa5]")
            line = rule.sub("", line)
            get_word_list(line, wordsList, stopList, file_words_list)

        spam_words_list.append(' '.join(wordsList))


    normal_labels = [0 for ii in range(len(normal_files))] #普通邮件labels 为0
    spam_labels = [1 for ii in range(len(spam_files))]     #垃圾邮件labels 为1

    all_labels = normal_labels+spam_labels
    all_words_list = file_words_list+spam_words_list

    return all_words_list, all_labels


def fitAndpredict():

    from sklearn import cross_validation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectPercentile, f_classif

    all_words_list, all_labels = WordsList()

    features_train, features_test, labels_train, labels_test = \
        cross_validation.train_test_split(all_words_list, all_labels,test_size=0.1, random_state=42)


    #文本向量化，提取重要词变量最大出现的概率为50%
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed = vectorizer.transform(features_test)


    # 根据训练集选择合适的特征
    # 进行特征转换
    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed = selector.transform(features_test_transformed).toarray()

    ### 邮件统计信息
    print("no. of spam training emails:", sum(labels_train))
    print("no. of norm training emails:", len(labels_train) - sum(labels_train))

    clf = GaussianNB()
    clf.fit(features_train_transformed, labels_train)
    pred = clf.predict(features_test_transformed)
    acc = accuracy_score(labels_test, pred)

    #打印准确率
    print('acc', features_test_transformed)


    #--------使用测试集里面的数据进行预测--------------
    wordsList = []
    file_words_list = []
    spam_words_list = []
    errors = 0
    files = os.listdir('data/test')

    for f in os.listdir("data/test"):

        wordsList.clear()
        spam_words_list.clear()

        for line in open("data/test/" + f):
            rule = re.compile(r"[^\u4e00-\u9fa5]")
            line = rule.sub("", line)
            get_word_list(line, wordsList, stopList, file_words_list)

        spam_words_list.append(' '.join(wordsList))

        features_test_transformed  = vectorizer.transform(spam_words_list)
        new_features_test = selector.transform(features_test_transformed).toarray()
        pred = clf.predict(new_features_test)

        print('pred: ', pred, f) #预测打印
        if int(f) <1000 and pred[0] == 1:
            errors+=1
        if int(f) >1000 and pred[0] == 0:
            errors +=1

    print(errors/len(files))#错误率