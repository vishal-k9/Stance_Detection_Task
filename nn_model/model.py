import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
import csv


#fetching data
categories = ["AGAINST","FAVOR","NONE"]


cat_idx={"AGAINST":0,"FAVOR":1,"NONE":2}
idx_cat={0:"AGAINST",1:"FAVOR",2:"NONE"}

test=[]
with open("../test dataset/Tweet_Target.csv","rb") as f:
    for row in f:
        test.append(row.strip())

train=[]
with open("../train dataset/Tweet_Target.csv","rb") as f:
    for row in f:
        train.append(row.strip())  

train_res=[]
with open("../train dataset/Stance.csv","rb") as f:
    for row in f:
        train_res.append(row.strip())    

test_res=[]
with open("../test dataset/Stance.csv","rb") as f:
    for row in f:
        test_res.append(row.strip())                  

    
#creating vocabulary
#vocab is counting the frequency on each word
vocab = Counter()

vocab["UNK"]=1
for text in train:
    # print text
    # break
    for word in text.split():
        vocab[word.lower()]+=1
        
# for text in test:

#     for string in text:
#         for word in string.split()
#             vocab[word.lower()]+=1

print("Total words:",len(vocab))



total_words = len(vocab)


#indexing the words
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i
        
    return word2index


#word2index holds words mapped to their indexes
word2index = get_word_2_index(vocab)

# print("Index of the word 'the':",word2index['UNK'])

# this function give a 2D matrix of batch_size X total_words, where each row contain numbers denoting the frequency of each word in the row.
# Also it returns batch_size X categories_size matrix denoting the category each row belongs to.
def get_batch(inp,out,i,batch_size):
    batches = []
    results = []
    texts = inp[i*batch_size:i*batch_size+batch_size]
    categories = out[i*batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words,dtype=float)
        for word in text.split():
            if(not word2index.has_key(word.lower())):
                layer[word2index["UNK"]] += 1
                continue
            layer[word2index[word.lower()]] += 1
            
        batches.append(layer)
        
    for category in categories:
        y = np.zeros((3),dtype=float)
        if category == "AGAINST":
            y[0] = 1.
        elif category == "FAVOR":
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)
            
     
    return np.array(batches),np.array(results)

print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(train,train_res,1,100)[1].shape)
print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(test,test_res,1,100)[1].shape)

# Parameters
learning_rate = 0.35
training_epochs = 10000
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 100      # 1st layer number of features
n_hidden_2 = 100       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 3         # Categories

input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output")

def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)
    
    # Output layer 
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x,batch_y = get_batch(train,train_res,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    
    # print output_tensor
    # Test model
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print accuracy
    total_test_data = len(test)
    batch_x_test,batch_y_test = get_batch(test,test_res,0,total_test_data)
    res=[]
    res=tf.argmax(prediction, 1).eval({input_tensor: batch_x_test, output_tensor: batch_y_test})
    with open("pred.csv","wb") as f:
        for row in res:
            f.write(idx_cat[row]+'\n')
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))

        
