import pickle
import numpy as np


# Step 1: define a function to load traing batch data from directory
def load_training_batch(folder_path, batch_id):
    """
	Args:
		folder_path: the directory contains data files
		batch_id: training batch id (1,2,3,4,5)
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""

    ###load batch using pickle###
    with open(folder_path + '/data_batch_'+str(batch_id), 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    ###fetch features using the key ['data']###
    features = batch[b'data']
    ###fetch labels using the key ['labels']###
    labels = batch[b'labels']
    return batch


# Step 2: define a function to load testing data from directory
def load_testing_batch(folder_path):
    """
	Args:
		folder_path: the directory contains data files
	Return:
		features: numpy array that has shape (10000,3072)
		labels: a list that has length 10000
	"""
    # load batch using pickle###
    with open(folder_path + '/test_batch', 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
        ###fetch features using the key ['data']###
        features = batch[b'data']
        ###fetch labels using the key ['labels']###
        labels = batch[b'labels']

    return batch


# Step 3: define a function that returns a list that contains label names (order is matter)
"""
	airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
"""


def load_label_names(labels_idx):
    label_names = []
    classes = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',8: 'ship', 9: 'truck'}
    for label_idx in labels_idx:
        label = classes.get(label_idx)
        label_names.append(label)
    return label_names


# Step 4: define a function that reshapes the features to have shape (10000, 32, 32, 3)
def features_reshape(features):
    """
	Args:
		features: a numpy array with shape (10000, 3072)
	Return:
		features: a numpy array with shape (10000,32,32,3)
	"""

    # for i in range(len(features)):
    # feature = features[i]
    r = features[:,0:1024]
    g = features[:,1024:2048]
    b = features[:,2048:3072]
    r = np.reshape(r, (-1,32, 32,1))
    g = np.reshape(g, (-1,32, 32,1))
    b = np.reshape(b, (-1,32, 32,1))
    new_features = np.concatenate([r,g,b],axis = 3)
    return new_features


# return new_feature


# Step 5 (Optional): A function to display the stats of specific batch data.
def display_data_stat(folder_path, batch_id, data_id):
    """
	Args:
		folder_path: directory that contains data files
		batch_id: the specific number of batch you want to explore.
		data_id: the specific number of data example you want to visualize
	Return:
		None

	Descrption:
		1)You can print out the number of images for every class.
		2)Visualize the image
		3)Print out the minimum and maximum values of pixel
	"""
    pass


# Step 6: define a function that does min-max normalization on input
def normalize(x):
    """
	Args:
		x: features, a numpy array
	Return:
		x: normalized features
	"""
    x_max = np.max(x)
    x_min = np.min(x)

    if x_max == x_min:
        x_max = x_min + 1

    y = np.zeros((x.shape[0],x.shape[1]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
        
            y[i][j] = (x[i][j] - x_min) / (x_max - x_min)
    return y


# Step 7: define a function that does one hot encoding on input
def one_hot_encoding(x):
    """
	Args:
		x: a list of labels
	Return:
		a numpy array that has shape (len(x), # of classes)
	"""

    numlabel = 10
    encode = np.zeros((len(x), numlabel))
    for i, label in enumerate(x):
        idx = label
        encode[i][idx] = 1

    return encode


# Step 8: define a function that perform normalization, one-hot encoding and save data using pickle
def preprocess_and_save(features, labels, filename):
    """
	Args:
		features: numpy array
		labels: a list of labels
		filename: the file you want to save the preprocessed data
	"""

    norm_feature = normalize(features)
    one_hot = one_hot_encoding(labels)
    dict = {'features': norm_feature,'labels':one_hot}
    pickle_out = open(filename, "wb")
    pickle.dump(dict, pickle_out)
    pickle_out.close()


# Step 9:define a function that preprocesss all training batch data and test data.
# Use 10% of your total training data as your validation set
# In the end you should have 5 preprocessed training data, 1 preprocessed validation data and 1 preprocessed test data
def preprocess_data(folder_path):
    """
	Args:
		folder_path: the directory contains your data files
	"""
    # load training dataset
    train_1 = load_training_batch(folder_path, 1)
    features_1 = train_1[b'data']
    labels_1 = train_1[b'labels']

    train_2 = load_training_batch(folder_path, 2)
    features_2 = train_2[b'data']
    labels_2 = train_2[b'labels']

    train_3 = load_training_batch(folder_path, 3)
    features_3 = train_3[b'data']
    labels_3 = train_3[b'labels']

    train_4 = load_training_batch(folder_path, 4)
    features_4 = train_4[b'data']
    labels_4 = train_4[b'labels']

    train_5 = load_training_batch(folder_path, 5)
    features_5 = train_5[b'data']
    labels_5 = train_5[b'labels']

    # split the train_5 dataset for validation set
    val_features = features_5[5000:10000][:]
    val_labels = labels_5[5000:10000][:]

    features_5 = features_5[0:5000][:]
    labels_5 = labels_5[0:5000][:]

    # load test dataset
    test = load_testing_batch(folder_path)
    test_features = test[b'data']
    test_labels = test[b'labels']


    # process and save the data
    preprocess_and_save(features_1, labels_1, "train_1")
    preprocess_and_save(features_2, labels_2, "train_2")
    preprocess_and_save(features_3, labels_3, "train_3")
    preprocess_and_save(features_4, labels_4, "train_4")
    preprocess_and_save(features_5, labels_5, "train_5")
    preprocess_and_save(val_features, val_labels, "validation")
    preprocess_and_save(test_features, test_labels, "test")



# Step 10: define a function to yield mini_batch
def mini_batch(features, labels, mini_batch_size):
    """
	Args:
		features: features for one batch
		labels: labels for one batch
		mini_batch_size: the mini-batch size you want to use.
	Hint: Use "yield" to generate mini-batch features and labels
	"""
    start_idx = 0
    end_idx = mini_batch_size
    while end_idx < len(labels):
        yield (features[start_idx:end_idx], labels[start_idx:end_idx])
        start_idx += mini_batch_size
        end_idx += mini_batch_size


# Step 11: define a function to load preprocessed training batch, the function will call the mini_batch() function
def load_preprocessed_training_batch(batch_id, mini_batch_size):
    """
	Args:
		batch_id: the specific training batch you want to load
		mini_batch_size: the number of examples you want to process for one update
	Return:
		mini_batch(features,labels, mini_batch_size)
	"""
    file_name = 'train_' + str(batch_id)
    with open(file_name, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')

    features,labels = batch['features'],batch['labels']


    return mini_batch(features, labels, mini_batch_size)


# Step 12: load preprocessed validation batch
def load_preprocessed_validation_batch(batch_size):
    file_name = 'validation'
    with open(file_name, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    
    features,labels = batch['features'],batch['labels']

    return mini_batch(features,labels, batch_size)


# Step 13: load preprocessed test batch
def load_preprocessed_test_batch(test_mini_batch_size):
    file_name = 'test'
    with open(file_name, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    features,labels = dict['features'],dict['labels']

    return mini_batch(features, labels, test_mini_batch_size)



