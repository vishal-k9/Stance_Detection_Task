# Introduction

This consist of neural network models for SemEval-2016 as Task 6: Detecting stance in tweets.

Stance detection can be formulated in different ways. In the context of this task, they define stance detection to mean automatically determining from text whether the author is in favor of the given target, against the given target, or whether neither inference is likely. Consider the target--tweet pair:

Target: Legalization of abortion
Tweet: A foetus has rights too! Make your voice heard.


Dataset Format:

The dataset contains four column:
i)    ID
ii)   Target
iii)  Tweet
iv)  Stance

There are five Targets:
i)    Atheism
ii)   Climate Change is a Real Concern
iii)  Feminist Movement
iv)  Hillary Clinton
v)  Legalization of Abortion

There are three classes of Stance:
i)   Favor
ii)  Against
iii) None



EXAMPLE:
2863  -------    Legalization of Abortion  -----------      	Abortion is not contraception. Contraception is about preventing pregnancy. Abortion is terminating a life.  #CCOT #SemST  -----   AGAINST

# Dataset

There are three dataset created.
From simple_train_dataset.csv, train dataset and dev dataset is created using:
                ''' python train_feature_seperator.py '''
From testdata.txt, test dataset is created using:
                ''' python test_feature_seperator.py '''

So, finally we have three dataset folders:
i) train dataset
ii) dev dataset
iii) test dataset



# Instruction

I have implemented two models for this task. 
i) Neural Networks Model.       ---- Folder Name --> nn_model
ii) RNN LSTM model with Attention     ---- Folder Name --> RNN-LSTM Model


Both these folder consist respective model codes. A README file is provided in both folders to guide you through the training and testing process.





