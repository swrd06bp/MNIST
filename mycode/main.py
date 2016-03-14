import network3 as nw


expanded_training_data, validation_data, test_data = nw.load_data_shared(
        "../data/mnist.pkl.gz")

mini_batch_size = 30

net = nw.Network([
        nw.ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=nw.ReLU),
        nw.ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      poolsize=(2, 2), 
                      activation_fn=nw.ReLU),
        nw.FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=nw.ReLU, p_dropout=0.5),
        nw.FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=nw.ReLU, p_dropout=0.5),
        nw.SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], 
        mini_batch_size)

net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, 
            validation_data, test_data, lmbda=0.1)