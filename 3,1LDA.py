from gensim.models import LdaModel
import os
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import csv

def readInDocuments(directory):
    texts = []
    for filename in os.listdir(directory):
        open_file = open(directory + filename, 'r')
        contents = open_file.readlines()
        tempList = []
        for i in range(len(contents)):
            tempList.extend(contents[i].split())
        texts.append(tempList)    
        open_file.close()
    return texts

#adapted from gensim documentation
def preprocessData(docs):
    for idx in range(len(docs)):
        docs[idx] = [word.lower() for word in docs[idx]]  # Convert to lowercase.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    #remove stopwords
    sw = set(stopwords.words('english'))
    docs = [[word for word in text if word not in sw] for text in docs]

    return docs

def readMeta():
    meshDict = {}
    with open('../metadata.csv', 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            #print(row)
            meshDict[row[0].strip()] = [topic.strip() for topic in row[1:]]
    return meshDict

def generateTopicDocVector(metadata):
    topic_doc_vector = {}
    for name,topics in metadata.items():
        for topic in topics:
            if topic in topic_doc_vector.keys():
                if name not in topic_doc_vector[topic]:
                    topic_doc_vector[topic].append(name)
            else:
                topic_doc_vector[topic] = [name]
    return topic_doc_vector

files = readInDocuments('../_corpus/')
metadata = readMeta()
topic_doc_vector = generateTopicDocVector(metadata)
print(metadata['SocialChange_18'])
files = preprocessData(files)
dictionary = Dictionary(files)
corpus = [dictionary.doc2bow(doc) for doc in files]

#this is only to load the dictionary
temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10)
top_topics = model.top_topics(corpus)
#print(top_topics[0])

#For each article A
#   For each MeSH label x associated with A
#       For each dimension j of the topic-document vector v[A]
#           Topic-label-M[x][j] += v[A][j]
for text in files:
    for mesh_label in metadata[text]:
        for j in topic_doc_vector:
            #do something here
            

