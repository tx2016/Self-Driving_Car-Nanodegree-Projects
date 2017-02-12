import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from alexnet import AlexNet

nb_classes = 43
EPOCHS = 10
BATCH_SIZE = 128

#Epoch 1, used batch_size 32
#Time: 2279.780 seconds (38 mins)
#Validation Loss = 0.368124737941
#Validation Accuracy = 0.891336270186

training_file = 'train.p'

# TODO: Load traffic signs data.
with open(training_file, mode = 'rb') as f:
	data = pickle.load(f)
# TODO: Split data into training and validation sets.
X_train, X_validation, y_train, y_validation = train_test_split(data['features'], data['labels'], test_size = 0.2, random_state = 0)
# TODO: Define placeholders and resize operation.

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8w = tf.Variable(tf.truncated_normal(shape, stddev = 1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8w, fc8b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_op = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer()
#optimizer = tf.train_AdamOptimizer()
train_op = optimizer.minimize(loss_op, var_list = [fc8w, fc8b])

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))
# TODO: Train and evaluate the feature extraction model.

def evaluate(X, y, sess):
	num_examples = len(X)
	total_acc = 0
	total_loss = 0
	for offset in range(0, num_examples, BATCH_SIZE):
		X_batch = X[offset: offset + BATCH_SIZE]
		y_batch = y[offset: offset + BATCH_SIZE]

		loss, acc = sess.run([loss_op, accuracy_op], feed_dict = {features: X_batch, labels: y_batch})
		total_loss += (loss * len(X_batch))
		total_acc += (acc * len(X_batch))

	return total_loss/num_examples, total_acc/num_examples


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	num_examples = len(X_train)

	for i in range(EPOCHS):
    # training
	    X_train, y_train = shuffle(X_train, y_train)
	    t0 = time.time()
	    for offset in range(0, num_examples, BATCH_SIZE):
	    	end = offset + BATCH_SIZE
	    	X_batch = X_train[offset: offset + BATCH_SIZE]
	    	y_batch = y_train[offset: offset + BATCH_SIZE]
	    	sess.run(train_op, feed_dict={features: X_batch, labels: y_batch})

	    val_loss, val_acc = evaluate(X_validation, validation, sess)
	    print("Epoch", i+1)
	    print("Time: %.3f seconds" % (time.time() - t0))
	    print("Validation Loss =", val_loss)
	    print("Validation Accuracy =", val_acc)
	    print("")



