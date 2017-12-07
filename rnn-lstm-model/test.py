#!/usr/bin/python

from train import *
from random import shuffle

def classifier(X):
    pred=np.array([[x] for x in X])
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pred)
    centres=np.sort(kmeans.cluster_centers_) 
    res=[]
    for elem in X:
        val=0
        dist=float("inf")
        for i in xrange(3):
            if(abs(elem-centres[i])<dist):
                dist=abs(elem-centres[i])
                val=i
        res.append(val)
    return np.array(res)    

f=open("Prediction.csv","wb")

Bucket_size=10

for i in xrange(5):
    saver = tf.train.import_meta_graph('model_'+str(i)+'.meta')
    
    # Calculate alpha coefficients for the first test example
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tf.variables_initializer(Variable_24)

        saver.restore(sess, "model_"+str(i))
       
        c = list(zip(X_test[i],Y_test[i]))
        shuffle(c)
        X_test[i],Y_test[i]=zip(*c)
        for j in xrange(0, len(X_test[i]), 10):
            x_batch_test, y_batch_test = X_test[i][j:j+10], Y_test[i][j:j+10]
            seq_len_test = np.array([list(x).index(0) + 1 for x in x_batch_test])
            alphas_test, y_hatv = sess.run([alphas,y_hat], feed_dict={batch_ph: x_batch_test, target_ph: y_batch_test,seq_len_ph: seq_len_test, keep_prob_ph: 1.0})
            pred=classifier(y_hatv)
            for p,r in izip(pred,y_batch_test):
                f.write(inv_stance_dict[p]+"\t"+inv_stance_dict[r]+"\n")

f.close()
