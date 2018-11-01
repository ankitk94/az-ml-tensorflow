# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from sklearn.externals import joblib
import os
import numpy as np
import pickle

os.makedirs('./outputs', exist_ok=True)

run = Run.get_submitted_run()

X, y = load_boston(return_X_y=True)

run = Run.get_submitted_run()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)
data = {"train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}}

reg = Ridge(alpha=0.65)
reg.fit(data["train"]["X"], data["train"]["y"])


import numpy as np
import argparse
import os
import tensorflow as tf

from azureml.core import Run
from utils import load_data

print("TensorFlow version:", tf.VERSION)


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep='\n')
training_set_size = X_train.shape[0]

n_inputs = 1 * 13
n_h1 = 20
n_h2 = 30
n_outputs = 100
learning_rate = 0.65
n_epochs = 100
batch_size = 20

with tf.name_scope('network'):
    # construct the DNN
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    h1 = tf.layers.dense(X, n_h1, activation=tf.nn.relu, name='h1')
    h2 = tf.layers.dense(h1, n_h2, activation=tf.nn.relu, name='h2')
    output = tf.layers.dense(h2, n_outputs, name='output')

with tf.name_scope('train'):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(output, y, 1)
    acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# start an Azure ML run
run = Run.get_context()
print("Starting session")
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):

        # randomly shuffle training set
        indices = np.random.permutation(training_set_size)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # batch index
        b_start = 0
        b_end = b_start + batch_size
        print("inside epoch")
        for _ in range(training_set_size // batch_size):
            # get a batch
            X_batch, y_batch = X_train[b_start: b_end], y_train[b_start: b_end]

            # update batch index for the next batch
            b_start = b_start + batch_size
            b_end = min(b_start + batch_size, training_set_size)

            # train
            print("Will run session")
            print (X_batch)
            print(y_batch)
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
            print("Done with run")
        # evaluate training set
        acc_train = acc_op.eval(feed_dict={X: X_batch, y: y_batch})
        # evaluate validation set
        acc_val = acc_op.eval(feed_dict={X: X_test, y: y_test})

        # log accuracies
        run.log('training_acc', np.float(acc_train))
        run.log('validation_acc', np.float(acc_val))
        print(epoch, '-- Training accuracy:', acc_train, '\b Validation accuracy:', acc_val)
        y_hat = np.argmax(output.eval(feed_dict={X: X_test}), axis=1)

    run.log('final_acc', np.float(acc_val))

    os.makedirs('./outputs/model', exist_ok=True)
    # files saved in the "./outputs" folder are automatically uploaded into run history
    saver.save(sess, './outputs/model/mnist-tf.model')
'''
preds = reg.predict(data["test"]["X"])
mse = mean_squared_error(preds, data["test"]["y"])

run.log("mse", mse)
print(mse)


model_path = 'model.pkl'
f = open(model_path, 'wb')
pickle.dump(reg, f)



with open(model_path, "wb") as file:
    from sklearn.externals import joblib
    joblib.dump(reg, file)
run.upload_file(model_path,  model_path)
os.remove(model_path)
'''