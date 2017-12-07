# Parameters
learning_rate = 0.4
training_epochs = 10000
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 100      # 1st layer number of features
n_hidden_2 = 100       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 3         # Categories


# Instructions

1. To train the model and predict results. Run 
            ''' python model.py '''


2. To calculate the F1 score of the predicted result. Run:
            ''' python f1score.py '''



Predictions of the test data will be stored in pred.csv file.
Actual labels of test data are stored in Stance.csv file.
