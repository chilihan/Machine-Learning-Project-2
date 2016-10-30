import csv
import numpy as np
import json
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

RANDOM_SEED = 1234

def split_data(X, y, words, num_train=7000, randomize=True):
    '''
    Split the data into training, validation and test sets.
    The size of the training set is given as num_train, while
    the remaining data is used for a validation set.
    '''
    n = len(y)
    if randomize:
        X, y, words = shuffle(X, y, words, random_state=RANDOM_SEED)

    # split into train + test
    train_X, val_X, train_y, val_y, train_words, val_words = \
            train_test_split(X, y, words,
                             test_size=X.shape[0]-num_train,
                             random_state=RANDOM_SEED+1)
    return (
        (train_X, train_y, train_words),
        (val_X, val_y, val_words),
    )

def _read_csv(csv_file, max_word_features=5000, train=True):
    # for storing the categorical and text features
    texts = []
    categories = []
    # for storing the numerical features
    X_list = []
    y_list = []

    print '...reading %s' % csv_file
    with open(csv_file, 'rb') as csvfile:
        data_reader = csv.reader(csvfile, delimiter='\t')
        # discard the first row (headings)
        data_reader.next()
        for row in data_reader:
            # row 2 is a json encoded string of the textual attributes of the page
            # In order to convert to bag of words representation, we first need to
            # convert everything to lower case, join the title, body, url into
            # a single string and then save to a list.
            js = json.loads(row[2])
            t = [js[k].lower() for k in ['title', 'body', ] \
                 if k in js and js[k] is not None]
            texts.append(' '.join(t))

            # row 3 of csv is the alchemy_category - string names representing
            # the 'topic' returned from the alchemy api
            # this list will be used to create a one hot representation
            categories.append(row[3])

            # last column of training data is the target value
            if train:
                y_list.append(int(row.pop()))
            else:
                # use the y_list to hold the urlids
                # that we need to make predictions
                y_list.append(int(row[1]))

            # The columns of the csv that corespond to the numerical data
            # some values are given as '?', so I've arbitrarily set those
            # to 0.5
            data = [float(i) if i != '?' else 0.5 for i in row[4:]]
            X_list.append(data)
    print("this is X_list:",X_list[1:3],type(X_list[1]))
    print("this is y_list:",y_list[1],type(y_list[1]))
    print("text:",texts[12],type(texts[12]))
    #print ("this is categories:",categories[1],type(categories[1]))


    return X_list, y_list, texts, categories


def read_evergreen_csv(train_csv='train.tsv', test_csv=None,
                       max_word_features=5000, dtype='float32'):
    X, y, texts, categories = _read_csv(train_csv, max_word_features, True)

    # get distinct category names
    category_names = list(set(categories))
    category_indices = dict((category_names[i], i) for i in
                            range(len(category_names)))

    print '...data has %s categories' % len(category_names)
    print '...appending one hot categories'
    for i in xrange(len(X)):
        cat = categories[i]
        one_hot = [0] * len(category_names)
        one_hot[category_indices[cat]] = 1
        X[i] += one_hot        # append one_hot list to X

    # make numpy arrays
    X = np.array(X, dtype=dtype)  # X_list plus binary categroes list
    y = np.array(y, dtype=dtype)  # last column of trainging data, label
    print ("this is X=====", X[0])
    print("this is y======",y[0])

    print '...creating bag of words'
    vectorizer = CountVectorizer(max_features=max_word_features,
                                 stop_words='english')
    print ("this is vectorizer----", vectorizer)
    # NOTE: fit_transform returns a sparse matrix
    X_words = vectorizer.fit_transform(texts)  # covert body to matrix
    print("=====This is X-word=====",X_words[1],type(X_words[1]))

    ret = {
        'train_X': X, # X_list plus binary categroes list
        'train_y': y,  # last column of trainging data, label
        'train_words': X_words, # covert body to matrix
        'vectorizer': vectorizer, # type is vectorizer, a object
    }

    if test_csv is not None:
        # now load the test data
        test_X, test_y, test_texts, test_categories = \
                _read_csv(test_csv, max_word_features, False)

        print '...appending one hot categories'
        for i in xrange(len(test_X)):
            cat = test_categories[i]
            one_hot = [0] * len(category_names)
            one_hot[category_indices[cat]] = 1
            test_X[i] += one_hot

        print '...creating bag of words'
        # reuse the vectorizer
        test_X_words = vectorizer.transform(test_texts) # covert text to matrix depend on learned vocab from training texts

        # make numpy array
        test_X = np.array(test_X, dtype=dtype)

        # NOTE: no labels for test set
        ret['test_X'] = test_X
        ret['test_ids'] = test_y
        ret['test_words'] = test_X_words

    return ret

def normalize_data(train_X, val_X, test_X):
    # normalize some of the data dimensions
    normalize_dimensions = np.array([
        0, # alchemy_category_score
        1, # avglinksize
        2, # commonLinkRatio_1
        3, # commonLinkRatio_2
        4, # commonLinkRatio_3
        5, # commonLinkRatio_4
        6, # compression_ratio
        7, # embed_ratio
        9, # frameTagRatio
        11, # html_ratio
        12, # image_ratio
        15, # linkwordscore
        17, # non_markup_alphanum_characters
        18, # numberOfLinks
        19, # numwords_in_url
        20, # parametrizedLinkRatio
        21, # spelling_errors_ratio
    ])

    means = train_X.mean(axis=0)
    print ("this is mean:", means, means.shape)
    stds = train_X.std(axis=0)
    print("this is stds:", stds, stds.shape)

    for X in (train_X, val_X, test_X):
        X[:,normalize_dimensions] -= means[normalize_dimensions]
        X[:,normalize_dimensions] /= stds[normalize_dimensions]
    print("X------:", X, X.shape)

if __name__ == '__main__':
    data_file       = 'numerical_features.pkl'
    words_file      = 'word_features.pkl'
    word_names_file = 'word_names.pkl'
    max_word_features = 2500

    d = read_evergreen_csv('train.tsv', 'test.tsv', max_word_features)
    train_data, val_data = split_data(d['train_X'], d['train_y'],
                                      d['train_words'], randomize=False)
    train_X, train_y, train_words = train_data
    val_X,   val_y,   val_words   = val_data
    test_X,  test_y,  test_words  = d['test_X'], d['test_ids'], d['test_words']
    print "...size of train, val, test: (%s, %s, %s)" % (train_X.shape[0],
                                                      val_X.shape[0],
                                                      test_X.shape[0])

    print '...normalizing data'
    normalize_data(train_X, val_X, test_X)

    data = (
        (train_X, train_y),
        (val_X,   val_y),
        (test_X,  test_y),
    )
    print("data======",data[0])


    print "...saving data pickle: %s" % data_file
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)

    bags_of_words = (
        train_words, # matrixes generated by documnets, 1*2500, 7000 matrixes, then 7000*2500
        val_words, # 395x2500
        test_words,
    )
    print("bags_of_words=====", bags_of_words[0])

    print '...saving word features pickle: %s' % words_file
    with open(words_file, 'wb') as f:
        pickle.dump(bags_of_words, f)

    word_names = d['vectorizer'].get_feature_names() # list, len=2500, 2500 features(words) of all documnet textx
    print("word_names====", word_names,len(word_names))

    print '...saving word names pickle: %s' % word_names_file
    with open(word_names_file, 'wb') as f:
        pickle.dump(word_names, f)
