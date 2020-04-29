from gensim.models import LdaModel
import os
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary

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

    return docs

files = readInDocuments('../_corpus/')
files = preprocessData(files)
dictionary = Dictionary(files)
corpus = [dictionary.doc2bow(doc) for doc in files]
print(corpus[10])

#this is only to load the dictionary
temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, num_topics=10)
top_topics = model.top_topics(corpus)
print(top_topics)