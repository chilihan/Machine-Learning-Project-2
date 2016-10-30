import pickle
import gzip


with open('numerical_features.pkl', 'rb') as f:
    numerical_features = pickle.load(f)
    print "============numerical_features=============="
    print numerical_features
 
#with gzip.open('word_features.pkl', 'rb') as f:
#    word_features = pickle.load(f)
#    print "============word_features=============="
#    print word_features
       
with open('word_features.pkl', 'r') as f:
    word_features = pickle.load(f)
    print "============word_features=============="
    print word_features


with open('word_names.pkl', 'rb') as f:
    word_names = pickle.load(f)
    print "============word_names=============="
    print word_names