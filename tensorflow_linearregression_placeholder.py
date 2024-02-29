import tensorflow as tf

# Define placeholders for input data
x_train_placeholder = tf.placeholder(tf.float32, shape=[None])
y_train_placeholder = tf.placeholder(tf.float32, shape=[None])

w = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

def hypothesis(x):
    return x * w + b

def compute_cost():
    return tf.reduce_mean(tf.square(hypothesis(x_train_placeholder) - y_train_placeholder))

optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Training data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Training loop
for step in range(2001):
    # Feed the training data through the placeholders
    feed_dict = {x_train_placeholder: x_train, y_train_placeholder: y_train}
    
    # Perform optimization
    optimizer.minimize(lambda: compute_cost(), var_list=[w, b], tape=tf.GradientTape(), options={'feed_dict': feed_dict})
    
    # Print the progress
    if step % 20 == 0:
        cost = compute_cost().numpy()
        print(step, cost, w.numpy(), b.numpy())
