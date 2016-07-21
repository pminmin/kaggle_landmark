import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

ftrain = '/home/pengmin/face_keypoints/kaggle/data/training.csv'
ftest = '/home/pengmin/face_keypoints/kaggle/data/test.csv'

def load(test=False, cols=None):
	fname = ftest if test else ftrain
	df = read_csv(os.path.expanduser(fname))

	# Convert the 'Image' column which has pixel values separated by space into numpy arrays
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ')) # df [7049, 31]

	if cols:
		df = df[list(cols) + ['Image']]

	print(df.count()) # print the number of values for each column(7039 2271 2267 ...)
	df = df.dropna() # drop all rows that have missing values in them (2140, 31)

	X = np.vstack(df['Image'].values) / 255. # scale pixel values to [0, 1]
	X = X.astype(np.float32)

	if not test:
		y = df[df.columns[:-1]].values
		y = (y - 48) / 48 # scale the target coordinates to [-1,1]
		X, y = shuffle(X, y, random_state=42)
		y = y.astype(np.float32)
	else:
		y = None

	return X, y

from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
	flip_indices = [
	(0,2),(1,3),
	(4,8),(5,9),(6,10),(7,11),
	(12,16),(13,17),(14,18),(15,19),
	(22,24),(23,25),
	]

	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs/2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1]

		if yb is not None:
			# Horizontal flip of all x coordinates
			yb[indices, ::2] = yb[indices, ::2] * -1

			# Swap places
			for a, b in self.flip_indices:
				yb[indices, a], yb[indices, b] = [yb[indices, b], yb[indices, a]]

		return Xb, yb

def plot_example(x, y, axis):
	img = x.reshape(96, 96)
	axis.imshow(img, cmap='gray')
	axis.scatter(y[0::2]*48+48, y[1::2]*48+48, marker='x', s=10)

def load2d(test=False, cols=None):
	X, y = load(test=test, cols=cols)
	X = X.reshape(-1, 1, 96, 96)
	return X, y

import matplotlib.pyplot as pyplot

def plot_loss(net, name):
	train_loss = np.array([i['train_loss'] for i in net.train_history_])
	valid_loss = np.array([i['valid_loss'] for i in net.train_history_])
	pyplot.plot(train_loss, linewidth=3, label='train')
	pyplot.plot(valid_loss, linewidth=3, label='valid')
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel('epoch')
	pyplot.ylabel('loss')
	pyplot.ylim(1e-3, 1e-2)
	pyplot.yscale('log')
	pyplot.savefig(name)

net1 = NeuralNet(
	layers=[
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
		],

	# layer parameters
	input_shape = (None, 9216),
	hidden_num_units = 100,
	output_nonlinearity = None, # but hidden_nonlinearity is ReLU and output is a linear combination
	output_num_units = 30, # 30 target values

	# optimization method
	update  = nesterov_momentum, # work very well for a large number of problems
	update_learning_rate = 0.01,
	update_momentum = 0.9,

	# for regression problem, the default objective function is the mean squared error(MSE)
	#
	regression = True,
	max_epochs = 400,
	verbose = 1,
	)

net2 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('hidden4', layers.DenseLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	hidden4_num_units=500, hidden5_num_units=500,
	output_num_units=30, output_nonlinearity=None,

	update_learning_rate=0.01,
	update_momentum=0.9,

	regression=True,
	max_epochs=1000,
	verbose=1,
)

net3 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('hidden4', layers.DenseLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	hidden4_num_units=500, hidden5_num_units=500,
	output_num_units=30, output_nonlinearity=None,
	
	update_learning_rate=0.01,
	update_momentum=0.9,

	regression=True,
	batch_iterator_train = FlipBatchIterator(batch_size=128),
	max_epochs=3000,
	verbose=1,
)

import theano

def float32(k):
	return np.cast['float32'](k)

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start = start
		self.stop = stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = float32(self.ls[epoch-1])
		getattr(nn, self.name).set_value(new_value)

# Changing learning rate and momentum over time
net4 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('hidden4', layers.DenseLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	hidden4_num_units=500, hidden5_num_units=500,
	output_num_units=30, output_nonlinearity=None,

	update_learning_rate = theano.shared(float32(0.03)),
	update_momentum = theano.shared(float32(0.9)),

	regression = True,
	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		],
	max_epochs = 3000,
	verbose = 1,
)

# Will overfit a bit, adding data augmentation(flip) based on net4
net5 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('hidden4', layers.DenseLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	hidden4_num_units=500, hidden5_num_units=500,
	output_num_units=30, output_nonlinearity=None,

	update_learning_rate = theano.shared(float32(0.03)),
	update_momentum = theano.shared(float32(0.9)),

	regression = True,
	batch_iterator_train = FlipBatchIterator(batch_size=128),
	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		],
	max_epochs = 3000,
	verbose = 1,
)

# Adding dropout based on overfit network(net5)
net6 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('dropout3', layers.DropoutLayer),
		('hidden4', layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	dropout1_p = 0.1,
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	dropout2_p = 0.2,
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	dropout3_p = 0.3,
	hidden4_num_units=500, hidden5_num_units=500,
	dropout4_p = 0.5,
	output_num_units=30, output_nonlinearity=None,

	update_learning_rate = theano.shared(float32(0.03)),
	update_momentum = theano.shared(float32(0.9)),

	regression = True,
	batch_iterator_train = FlipBatchIterator(batch_size=128),
	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		],
	max_epochs = 3000,
	verbose = 1,
)

# Modify the last two hidden layers to make the network more complicated
net7 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('dropout3', layers.DropoutLayer),
		('hidden4', layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	dropout1_p = 0.1,
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	dropout2_p = 0.2,
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	dropout3_p = 0.3,
	hidden4_num_units=1000, hidden5_num_units=1000,
	dropout4_p = 0.5,
	output_num_units=30, output_nonlinearity=None,

	update_learning_rate = theano.shared(float32(0.03)),
	update_momentum = theano.shared(float32(0.9)),

	regression = True,
	batch_iterator_train = FlipBatchIterator(batch_size=128),
	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		],
	max_epochs = 5000,
	verbose = 1,
)

class EarlyStopping(object):
	def __init__(self, patience=100):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()

		elif self.best_valid_epoch + self.patience < current_epoch:
			print('Early stopping.')
			print("Best valid loss was {:.6f} at epoch {}".format(self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()

net8 = NeuralNet(
	layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('dropout3', layers.DropoutLayer),
		('hidden4', layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
		('hidden5', layers.DenseLayer),
		('outputs', layers.DenseLayer),
		],
	input_shape = (None, 1, 96, 96),
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_pool_size=(2,2),
	dropout1_p = 0.1,
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_pool_size=(2,2),
	dropout2_p = 0.2,
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_pool_size=(2,2),
	dropout3_p = 0.3,
	hidden4_num_units=1000, hidden5_num_units=1000,
	dropout4_p = 0.5,
	outputs_num_units=30, outputs_nonlinearity=None,

	update_learning_rate = theano.shared(float32(0.03)),
	update_momentum = theano.shared(float32(0.9)),

	regression = True,
	batch_iterator_train = FlipBatchIterator(batch_size=128),
	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		EarlyStopping(patience=200),
		],
	max_epochs = 5000,
	verbose = 1,
)

SPECIALIST_SETTINGS = [
	dict(
		columns=('left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y'),
		flip_indices=((0,2),(1,3)),
	),
	dict(
		columns=('nose_tip_x','nose_tip_y'),
		flip_indices=(),
	),
	dict(
		columns=('mouth_left_corner_x','mouth_left_corner_y','mouth_right_corner_x','mouth_right_corner_y','mouth_center_top_lip_x','mouth_center_top_lip_y'),
		flip_indices=((0,2),(1,3)),
	),
	dict(
		columns=('mouth_center_bottom_lip_x','mouth_center_bottom_lip_y'),
		flip_indices=(),
	),
	dict(
		columns=('left_eye_inner_corner_x','left_eye_inner_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y',
			'left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y'),
		flip_indices=((0,2),(1,3),(4,6),(5,7)),
	),
	dict(
		columns=('left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y',
			'left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_y'),
		flip_indices=((0,2),(1,3),(4,6),(5,7)),
	),
]

from collections import OrderedDict
from sklearn.base import clone
import cPickle as pickle

def fit_specialists(fname_pretrain=None):
	if fname_pretrain:
		with open(fname_pretrain, 'rb') as f:
			net_pretrain = pickle.load(f)
	else:
		net_pretrain = None

	specialists = OrderedDict()

	for setting in SPECIALIST_SETTINGS:
		cols = setting['columns']
		X, y = load2d(cols=cols)
		
		model = clone(net8)
		model.outputs_num_units = y.shape[1]
		model.batch_iterator_train.flip_indices = setting['flip_indices']
		model.max_epochs = int(4e6/y.shape[0])
		if 'kwargs' in setting:
			vars(model).update(setting['kwargs'])

		if net_pretrain is not None:
			print('using pretrain model {}'.format(fname_pretrain))
			model.load_params_from(net_pretrain)
		
		print y.shape[1]
		print model.outputs_num_units

		print("Training model for columns {} for {} epochs".format(cols, model.max_epochs))
		model.fit(X, y)
		specialists[cols] = model

	with open('net-specialists.pickle', 'wb') as f:
		pickle.dump(specialists, f, -1)

def rebin(a, newshape):
	from numpy import mgrid
	assert len(a.shape) == len(newshape)

	slices = [slice(0,old,float(old)/new) for old,new in zip(a.shape, newshape)]
	coordinates = mgrid[slices]
	indices = coordinates.astype('i')
	return a[tuple(indices)]

def plot_learning_curves(fname, fname_specialists='net-specialists.pickle'):
	with open(fname_specialists,'r') as f:
		models = pickle.load(f)

	fig = pyplot.figure(figsize=(10,6))
	ax = fig.add_subplot(1,1,1)
	ax.set_color_cycle(['c','c','m','m','y','y','k','k','g','g','b','b'])
	valid_losses=[]
	train_losses=[]

	for model_number, (cg, model) in enumerate(models.items(),1):
		valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
		train_loss = np.array([i['train_loss'] for i in model.train_history_])
		valid_loss = np.sqrt(valid_loss) * 48
		train_loss = np.sqrt(train_loss) * 48

		valid_loss = rebin(valid_loss, (100,))
		train_loss = rebin(train_loss, (100,))

		valid_losses.append(valid_loss)
		train_losses.append(train_loss)
		ax.plot(valid_loss, label='{} ({})'.format(cg[0], len(cg)), linewidth=3)
		ax.plot(train_loss, linestyle='--', linewidth=3, alpha=0.6)
		ax.set_xticks([])

	weights = np.array([m.outputs_num_units for m in models.values()], dtype=float)
	weights /= weights.sum()
	mean_valid_loss = (np.vstack(valid_losses) * weights.reshape(-1,1).sum(axis=0))
	ax.plot(mean_valid_loss, color='r', label='mean', linewidth=4, alpha=0.8)

	ax.legend()
	ax.set_ylim((1.0,4.0))
	ax.grid()
	pyplot.ylabel('RMSE')
	pyplot.savefig(fname)
