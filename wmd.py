#start
import pdb, sys, numpy as np, pickle, multiprocessing as mp
sys.path.append('python-emd-master')
from emd import emd

load_file = sys.argv[1]
load_file_2 = sys.argv[2]
save_file = sys.argv[3]

with open(load_file) as f:
    [X, BOW_X, y, C, words] = pickle.load(f)
n = np.shape(X)
n = n[0]
D = np.zeros((n,n))
for i in xrange(n):
    bow_i = BOW_X[i]
    bow_i = bow_i / np.sum(bow_i)
    bow_i = bow_i.tolist()
    BOW_X[i] = bow_i
    X_i = X[i].T
    X_i = X_i.tolist()
    X[i] = X_i

# Process the second file
with open(load_file_2) as f:
    [X_2, BOW_X_2, y_2, C_2, words_2] = pickle.load(f)
n_2 = np.shape(X_2)
n_2 = n[0]
D_2 = np.zeros((n_2,n_2))
for i in xrange(n_2):
    bow_i = BOW_X_2[i]
    bow_i = bow_i / np.sum(bow_i)
    bow_i = bow_i.tolist()
    BOW_X_2[i] = bow_i
    X_i = X_2[i].T
    X_i = X_i.tolist()
    X_2[i] = X_i

def distance(x1,x2):
    return np.sqrt( np.sum((np.array(x1) - np.array(x2))**2) )

def get_wmd(ix):
    n = np.shape(X)
    n = n[0]
    Di = np.zeros((1,n))
    n_2 = np.shape(X_2)
    n_2 = n_2[0]
    Di = np.zeros((1,n_2))
    i = ix
    print '%d out of %d' % (i, n)
    for j in xrange(n_2):
        Di[0,j] = emd( (X[i], BOW_X[i]), (X_2[j], BOW_X_2[j]), distance)
    return Di 
            

def main():
    n = np.shape(X)
    n = n[0]
    n_2 = np.shape(X_2)
    n_2 = n_2[0]
    pool = mp.Pool(processes=8)

    pool_outputs = pool.map(get_wmd, list(range(n)))
    pool.close()
    pool.join()

    WMD_D = np.zeros((n,n_2))
    for i in xrange(n):
        WMD_D[:,i] = pool_outputs[i]

    with open(save_file, 'w') as f:
        pickle.dump(WMD_D, f)

if __name__ == "__main__":
    main()




