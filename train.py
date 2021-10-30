import argparse
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import data_loader
import os
import requests
from OGRU import OGRU_cell
from module import LSTM_cell, GRU_cell
import timeseries_loader
# from ppretty import ppretty

'''
Name: Kaustubh Hiware
@kaustubhhiware
run with python2

python train.py --train: Defaults to LSTM, hidden_unit 32, 30 iterations / epochs
python train.py --train --hidden_unit 32 --model lstm --iter 5: Train LSTM and dump weights
python train.py --test --hidden_unit 32 --model lstm: Test with stored weights
'''

# Network Parameters
'''image data configuration'''
seed = 123
input_nodes = 28
output_nodes = 10
learning_rate = 0.005
num_iterations = 30
batch_size = 100
data_opt = 'timeseries'

'''timeseries data configuration'''
# seed = 123
# input_nodes = 3
# output_nodes = 1
# learning_rate = 0.005
# num_iterations = 100
# batch_size = 12
data = data_loader.DataLoader()
weights_folder = '/weights/'
# timeseries = timeseries_loader.TimeseriesLoader()
# data = timeseries_loader.TimeseriesLoader()

# check if needed files are present or not. Downloads if needed.
def check_download_weights(model, hidden_unit):

    url_prefix = 'https://raw.githubusercontent.com/kaustubhhiware/LSTM-GRU-from-scratch/master'
    files = ['checkpoint', 'model', 'model.ckpt.data-00000-of-00001','model.ckpt.index', 'model.ckpt.meta']

    for file in files:
        if not os.path.exists(os.getcwd() + weights_folder + file):
            print('Downloading', file)
            url = url_prefix + weights_folder + file
            # urllib.urlretrieve(url, filename= os.getcwd() + weights_folder + file)
            r = requests.get(url)
            open(os.getcwd() + weights_folder + file, 'wb').write(r.content)


def start_training(train, test, hidden_unit, model, alpha=learning_rate, isTrain=False, num_iterations=num_iterations, batch_size=100, opt = 'sgd'):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # tf.random.set_seed(seed)
    (trainX, trainY) = train
    (testX, testY) = test
    (n_x, m, m2) = trainX.T.shape

    '''
    Tensorflow v1.x
    '''
    Y = tf.placeholder(tf.float32, shape=[None, output_nodes], name='inputs')
    
    
    '''
    Tensorflow v2.x
    Y = tf.Variable(tf.ones(shape=[None, output_nodes]), dtype=tf.float32, name='inputs')
    '''

    if model == 'lstm':
        rnn = LSTM_cell(input_nodes, hidden_unit, output_nodes)
    elif model == 'gru':
        rnn = GRU_cell(input_nodes, hidden_unit, output_nodes)
    else:
        rnn = OGRU_cell(input_nodes, hidden_unit, output_nodes)

    outputs = rnn.get_outputs()
    print('Output layer:', outputs)
    print('outputs[-1] :', outputs[-1])
    prediction = tf.nn.softmax(outputs[-1])

    
    if data_opt == 'image':
        cost = -tf.reduce_sum(Y * tf.log(prediction))
    elif data_opt == 'timeseries':
        cost = -tf.reduce_sum(Y * tf.log(prediction))
    saver = tf.train.Saver(max_to_keep=10)

    # optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
    if opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
    elif opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if not os.path.isdir(os.getcwd() + weights_folder):
        print('Missing folder made')
        os.makedirs(os.getcwd() + weights_folder)

    if isTrain:
        num_minibatches = len(trainX) / batch_size
        for iteration in range(num_iterations):
            iter_cost = 0.
            batch_x, batch_y = data.create_batches(trainX, trainY, batch_size=batch_size)
            # batch_x, batch_y = train

            for (minibatch_X, minibatch_Y) in zip(batch_x, batch_y):
                # minibatch_x_size = np.array(minibatch_X)
                # minibatch_y_size = np.array(minibatch_Y)
                # print('minibatch x:',minibatch_x_size.shape)
                # print('minibatch y:',minibatch_y_size.shape)

                _, minibatch_cost, acc = sess.run([optimizer, cost, accuracy], feed_dict={rnn._inputs: minibatch_X, Y: minibatch_Y})
                iter_cost += minibatch_cost*1.0 / num_minibatches

            print("Iteration {iter_num}, Cost: {cost}, Accuracy: {accuracy}".format(iter_num=iteration, cost=iter_cost, accuracy=acc))

        # print ppretty(rnn)
        Train_accuracy = str(sess.run(accuracy, feed_dict={rnn._inputs: trainX, Y: trainY}))
        # Test_accuracy = str(sess.run(accuracy, feed_dict={rnn._inputs: testX, Y: testY}))

        save_path = saver.save(sess, "." + weights_folder + data_opt + '/'+ "model_" + model + "_" + str(hidden_unit) + ".ckpt")
        print("Parameters have been trained and saved!")
        print("\rTrain Accuracy: %s" % (Train_accuracy))

    else:  # test mode
        # no need to download weights in this assignment
        # check_download_weights(model, hidden_unit)

        saver.restore(sess, "." + weights_folder + "model_" + model + "_" + str(hidden_unit) + ".ckpt")
        acc = sess.run(accuracy, feed_dict={rnn._inputs: testX, Y: testY})
        print("Test Accuracy:"+"{:.3f}".format(acc))

    sess.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", help="Initiate training phase and store weights")
    parser.add_argument('--test', action="store_true", help="Initiate testing phase, load model and print accuracy")
    parser.add_argument('--hidden_unit', action="store", dest="hidden_unit", type=int, choices=[32, 64, 128, 256], help="Specify hidden unit size")
    parser.add_argument('--model', action="store", dest="model", choices=["lstm", "gru", "ogru"], help="Specify model name")
    parser.add_argument('--iter', action="store", dest="iter", type=int, help="Specify number of iterations")
    parser.add_argument('--optimizer', action="store", dest="optimizer", choices=["sgd","adam"], help="Specify optimizer method")
    parser.add_argument('--data', action="store", dest="data", choices=["timeseries","image"], help="Specify data to train")

    isTrain_ = False
    num_iterations_ = num_iterations
    hidden_unit = 32
    model = 'lstm'  # lstm ,gru or ogru
    optimizer = 'sgd'
    args = parser.parse_args()
    if args.hidden_unit:
        print("> hidden unit flag has set value", args.hidden_unit)
        hidden_unit = args.hidden_unit

    if args.model:
        print("> model flag has set value", args.model)
        model = args.model

    if args.train:
        print("> Now Training")
        isTrain_ = True
        if args.iter:
            num_iterations_ = args.iter

    if args.optimizer:
        print("> Now Optimizing with ", args.optimizer)
        optimizer = args.optimizer

    if args.data:
        print("> data training with", args.data)
        global data_opt
        data_opt = args.data

    elif args.test:
        print("> Now Testing")
    else:
        print("> Need to provide train / test flag!")
        exit(0)

    train, test = dataLoaderTest(option=data_opt)

    print("> Running for", num_iterations_,"iterations")
    print("> Hidden size unit", hidden_unit)
    start_training(train,
                    test,
                    isTrain=isTrain_, 
                    num_iterations=num_iterations_,
                    hidden_unit=hidden_unit,
                    model=model,
                    batch_size=batch_size,
                    opt=optimizer
                    )

def dataLoaderTest(option='timeseries'):

    global data
    global seed 
    global input_nodes
    global output_nodes
    global learning_rate
    global num_iterations
    global batch_size

    if option == 'timeseries':
        #load the data from loader
        data = timeseries_loader.TimeseriesLoader()
        trainX, trainY, testX, testY = data.load_data()
        #set the configuration
        seed = 123
        input_nodes = 3
        output_nodes = 1
        learning_rate = 0.001
        num_iterations = 100
        batch_size = 12

    elif option == 'image':
        #load the data
        data = data_loader.DataLoader()
        trainX, trainY = data.load_data('train')
        testX, testY = data.load_data('test')
        #set the configuration
        seed = 123
        input_nodes = 28
        output_nodes = 10
        learning_rate = 0.005
        num_iterations = 30
        batch_size = 100
        
    # trainX, trainY = data.load_data('train')
    train = (trainX, trainY)
    # testX, testY = data.load_data('test')
    test = (testX, testY)

    # trainX, trainY, testX, testY = data.load_data()
    # train = (trainX, trainY)
    # testX, testY = data.load_data('test')
    # test = (testX, testY)

    # batch_x, batch_y = data.create_batches(trainX, trainY, batch_size=12)
    # size_x = np.array(batch_x)
    # size_y = np.array(batch_y)

    # print('trainX shape', trainX.shape)
    # print('trainY shape', trainY.shape)
    # print('testX shape', testX.shape)
    # print('testY shape', testY.shape)
    # print('batch_x shape:', size_x.shape)
    # print('batch_y shape:', size_y.shape)

    return train, test

    # Shape
    

    # print('test shape', test.shape)
    # print('train shape', train.shape)


if __name__ == '__main__':
    main()
    # dataLoaderTest()
