from gensim.models import LdaModel
import os
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import csv
import string
import numpy as np

from numpy.linalg import norm
import matplotlib.pyplot as plt; plt.rcdefaults()

import matplotlib.pyplot as plt



def readInDocuments(directory):
    texts = []
    print(len(os.listdir(directory)))
    for filename in os.listdir(directory):
        
        open_file = open(directory + filename, 'r',encoding="utf8")
        contents = open_file.readlines()
        tempList = [filename]
        contentList = []
        for i in range(len(contents)):
            contentList.extend(contents[i].split())
        open_file.close()
        contentList = preprocessData(contentList)
        tempList.append(contentList)
        texts.append(tempList)
    return texts

#adapted from gensim documentation
def preprocessData(doc):
    doc = [word.lower() for word in doc]  # Convert to lowercase.

    # Remove numbers, but not words that contain numbers.
    doc = [token for token in doc if not token.isnumeric()]

    # Remove words that are only one character.
    doc = [token for token in doc if len(token) > 1]

    #remove punctuation
    doc = [s.translate(str.maketrans('', '', string.punctuation)) for s in doc]

    #remove stopwords
    sw = set(stopwords.words('english'))
    doc = [word for word in doc if word not in sw]

    lemmatizer = WordNetLemmatizer()
    doc = [lemmatizer.lemmatize(token) for token in doc]

    return doc

def readMeta():
    meshDict = {}
    with open('../metadata.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            meshDict[row[0].strip()] = [topic.strip() for topic in row[1:]]
    return meshDict

# def generateLabelDocVector(metadata):
#     label_topic_vector = {}
#     for name,labels in metadata.items():
#         for label in labels:
#             if label in label_topic_vector.keys():
#                 if name not in label_topic_vector[label]:
#                     label_topic_vector[label].append(name)
#             else:
#                 label_topic_vector[label] = [name]
#     return label_topic_vector

def generateTopicLabelMatrix(model, metadata, data, dictionary, num_topics):

    topic_label_matrix = {}
    # For each article A
    for text in data:
        # For each MeSH label associated with A
        for mesh_label in metadata[text[0]]:
            corp = [dictionary.doc2bow(text[1])]
            topic_doc_vector = model[corp]
            # For each dimension j of the topic-document vector v[A]
            for j in topic_doc_vector:
                #fill in topic as 0.0 if model didn't produce it
                for i in range(0, num_topics):
                    if i not in [tup[0] for tup in j]:
                        j.append((i, 0.0))
                if mesh_label in topic_label_matrix.keys():
                    #Topic-label-M[label][j] += v[A][j]
                    for a in range(len(j)):
                        for b in range(len(topic_label_matrix[mesh_label])):
                            if j[a][0] == topic_label_matrix[mesh_label][b][0]:
                                #convert to list because tuples are immutable
                                topic_label_matrix[mesh_label][b] = list(topic_label_matrix[mesh_label][b])
                                #add the value
                                topic_label_matrix[mesh_label][b][1] += j[a][1]
                                #convert back to tuple
                                topic_label_matrix[mesh_label][b] = tuple(topic_label_matrix[mesh_label][b])
                else:
                    topic_label_matrix[mesh_label] = j.copy()

    return topic_label_matrix

######
##MAIN
######

metadata = readMeta()
#label_topic_vector = generateLabelDocVector(metadata)

#format: [[SocialChange_5, [word, word, word]], etc.]
files = readInDocuments('../_corpus/')
dictionary = Dictionary([text[1] for text in files])

#use half the data set for training
training = files[:1000]
trainingText = [text[1] for text in training]
training_dictionary = Dictionary(trainingText)
corpus = [training_dictionary.doc2bow(text) for text in trainingText]

#this is only to load the dictionary
temp = training_dictionary[0]
id2word = training_dictionary.id2token

#create model with training data
num_topics = 8
model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

#use other half for testing
testing = files[1000:]
testingText = [text[1] for text in testing]
test_dictionary = Dictionary(testingText)
test_corpus = [test_dictionary.doc2bow(text) for text in testingText]

topic_label_matrix = generateTopicLabelMatrix(model, metadata, training, dictionary, num_topics)

combinedList=[]

with open('testoutput.txt', 'w') as outfile:
    for i in range(len(test_corpus)):
        m = model[test_corpus[i]]
        doc_name = testing[i][0]
        #fill in topic as 0.0 if model didn't produce it
        for i in range(0, num_topics):
            if i not in [tup[0] for tup in m]:
                m.append((i, 0.0))
        #sort and convert the topic document matrix
        m = sorted(m, key = lambda x: x[0])
        m = [tup[1] for tup in m]
        m = np.array(m)
        outfile.write(doc_name + '\n')
        labelList=[]
        dotList=[]
        cossineList=[]
        for label in topic_label_matrix.keys():
            #sort
            t = sorted(topic_label_matrix[label], key = lambda x: x[0])
            #convert to list without topics
            t = [tup[1] for tup in t]
            #convert to numpy array
            t = np.array(t)
            outfile.write(str(label) + ' ' + str(np.dot(m, t)) + '\n')
            cos_sim = np.dot(m, t)/(norm(m)*norm(t))

            # combinedList.append([doc_name,str(label),np.dot(m, t),cos_sim])
            labelList.append(str(label))
            dotList.append(np.dot(m, t))
            cossineList.append(cos_sim)
        combinedList.append([doc_name,labelList,dotList,cossineList])
        outfile.write('\n')
        outfile.write('\n')
        outfile.write('\n')
        outfile.write('\n')

#Use a different way to calculate the similarity between an article and the topic (instead of dot product) 

with open('testoutputNOTDOT.txt', 'w') as outfile:
    for i in range(len(test_corpus)):
        m = model[test_corpus[i]]
        doc_name = testing[i][0]
        #fill in topic as 0.0 if model didn't produce it
        for i in range(0, num_topics):
            if i not in [tup[0] for tup in m]:
                m.append((i, 0.0))
        #sort and convert the topic document matrix
        m = sorted(m, key = lambda x: x[0])
        m = [tup[1] for tup in m]
        m = np.array(m)
        outfile.write(doc_name + '\n')
        for label in topic_label_matrix.keys():
            #sort
            t = sorted(topic_label_matrix[label], key = lambda x: x[0])
            #convert to list without topics
            t = [tup[1] for tup in t]
            #convert to numpy array
            t = np.array(t)

            cos_sim = np.dot(m, t)/(norm(m)*norm(t))
            outfile.write(str(label) + ' ' + str(cos_sim) + '\n')
            
        outfile.write('\n')
        outfile.write('\n')
        outfile.write('\n')
        outfile.write('\n')
            
# for x in combinedList:
#     print(x)

helper=combinedList[0]

y_pos = np.arange(len(helper[1]))

plt.barh(tuple(labelList), dotList, align='center', alpha=0.5)
plt.yticks(y_pos, helper[1])
# plt.xlabel('Usage')
plt.title(helper[0])



plt.tight_layout()
plt.show()