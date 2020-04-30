from gensim.models import LdaModel
import os
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import csv
import string

def readInDocuments(directory):
    texts = []
    for filename in os.listdir(directory):
        open_file = open(directory + filename, 'r')
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
            #print(row)
            meshDict[row[0].strip()] = [topic.strip() for topic in row[1:]]
    return meshDict

def generateLabelDocVector(metadata):
    label_topic_vector = {}
    for name,labels in metadata.items():
        for label in labels:
            if label in label_topic_vector.keys():
                if name not in label_topic_vector[label]:
                    label_topic_vector[label].append(name)
            else:
                label_topic_vector[label] = [name]
    return label_topic_vector

def generateTopicLabelMatrix(model, metadata, data, files_dictionary):
# For each article A
#   For each MeSH label associated with A
#       For each dimension j of the topic-document vector v[A]
#           Topic-label-M[label][j] += v[A][j]
    topic_label_matrix = {}
    for text in data:
        for mesh_label in metadata[text]:
            topic_doc_vector = model[text]
            for j in topic_doc_vector:
                if mesh_label in topic_label_matrix.keys():
                    topic_label_matrix[mesh_label].append(j)
                else:
                    topic_label_matrix[mesh_label] = [j]
    return topic_label_matrix

metadata = readMeta()
label_topic_vector = generateLabelDocVector(metadata)

#format: [[SocialChange_5, [word, word, word]], etc.]
files = readInDocuments('../_corpus/')

#use half the data set for training
training = files[:1000]
trainingText = [text[1] for text in training]
dictionary = Dictionary(trainingText)
corpus = [dictionary.doc2bow(text) for text in trainingText]

#this is only to load the dictionary
temp = dictionary[0]
id2word = dictionary.id2token

#create model with training data
model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10)

#use other half for testing
testing = files[1000:]
testingText = [text[1] for text in testing]
test_dictionary = Dictionary(testingText)
test_corpus = [test_dictionary.doc2bow(text) for text in testingText]

vector = model[test_corpus[2]]
print(vector)
for result in vector:
    print(test_dictionary[result[0]])

print(generateTopicLabelMatrix(model, metadata, testing, filesDict))
            


