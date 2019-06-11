
# Gensim
import gensim

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'could', 'who', 'would', 'may'])#'go','come', 'become', 'be'
stop_words.remove('own')

class LDATopicDistAnnotator():

    def __init__(self, ldaModel, num_topics, bigram_mod=None):

        self.loadModel(ldaModel, bigram_mod)

        self.num_topics = num_topics
        self.num_words = 10

        #build topic-keywords dictionary, keywords vocabulary and representation
        self.kpad_word = "<kpad>"
        self.outOfTopic = "OOT"
        self.keyvocab = {}
        self.keyvocab[self.kpad_word] = 1
        self.keyvocab[self.outOfTopic] = 2
        self.topickeys = {}
        for i, keyswords in self.lda_mod.show_topics(-1, formatted=False, num_words=self.num_words):
            self.topickeys[i] = []
            for term, p in keyswords:
                self.topickeys[i].append(term)
                if term not in self.keyvocab.keys():
                    self.keyvocab[term] = len(self.keyvocab)+1

    def loadModel(self, ldaModel, bigram_mod=None):
        raise NotImplementedError

    def generateKeywordEmbeddings(self, save_dir):
        raise NotImplementedError

    def generateKeywordDict(self, save_dir):

        f = open(save_dir, 'w', encoding='utf8')
        for term in self.keyvocab.keys():
            f.write(term + "\n")
        f.close()

    #@abstractmethod
    def topicDistrib(self, phrase, caller=None):
        raise NotImplementedError

    #@abstractmethod
    def keywords(self, phrase):
        raise NotImplementedError
