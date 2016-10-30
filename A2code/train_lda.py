import logging
import gensim
import pickle
import numpy as np

from gensim.matutils import Sparse2Corpus

def load_words(word_pkl, word_names_pkl):
    with open(word_pkl) as pkl:
        words = pickle.load(pkl)

    with open(word_names_pkl) as pkl:
        word_names = pickle.load(pkl)
    return words, word_names


def train_lda(words, word_names, num_topics=30, update_every=1,
              chunksize=500, passes=10, topn=15, out_file=None):
    # convert word_names to a dict
    word_dict = dict((i, word_names[i]) for i in xrange(len(word_names)))

    corpus = Sparse2Corpus(words[0], documents_columns=False)
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                          id2word=word_dict,
                                          num_topics=num_topics,
                                          update_every=update_every,
                                          chunksize=chunksize,
                                          passes=passes)

    # print the topics
    print '--------------------------------------------'
    print 'Top %s words of each topic descovered by LDA' % topn
    print '--------------------------------------------'

    # lda show_topic returns (word,probability)
    for i in xrange(lda.num_topics):
        print i+1, ', '.join(t[0] + '({0:.2g})'.format(t[1])  for t in lda.show_topic(i,topn=topn))

    print '--------------------------------------------'

    if out_file is not None:
        with open(out_file, 'w') as f:
            pickle.dump(lda, f)
    return lda

if __name__ == '__main__':
    num_topics     = 30
    chunksize      = 500
    passes         = 5
    lda_file       = 'lda_model.pkl'
    topic_file     = 'lda_features.pkl'
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    words, word_names = load_words('word_features.pkl', 'word_names.pkl')
                                   # bag of words(3 numerical matrixes)  # list, 2500 features(words)

    lda = train_lda(words, word_names, num_topics=num_topics,
                    chunksize=chunksize, passes=passes,
                    out_file=lda_file)

    # preform inference on each document
    print '...extracting topic features of training, validation, and test data'
    topics = [lda.inference(Sparse2Corpus(w, documents_columns=False), collect_sstats=False)[0] for w in words]

    # take transpose so that we can broadcast
    topics = [t.T for t in topics]


    # normalize the probabilities
    for t in topics:
        t /= t.sum(axis=0)

    # transpose back
    topics = [t.T for t in topics]
    topicss=np.array(topics)
    print ("topic=====",topics, topics.shape)


    with open(topic_file, 'w') as f:
        pickle.dump(topics, f)
