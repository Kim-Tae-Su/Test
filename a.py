import tensorflow as tf
import matplotlib.pyplot as plt

# Training data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Convert input data to NumPy arrays
x_train = tf.constant(x_train, dtype=tf.float32)
y_train = tf.constant(y_train, dtype=tf.float32)

# Define the linear regression model
def hypothesis(x, w):
    return x * w

# Mean squared error as the cost function
def cost_function(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Gradient descent optimizer
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# TensorFlow eager execution
tf.config.run_functions_eagerly(True)

# Training loop
w_val = []
cost_val = []

# Initialize the weight within the range -3 to 3
w = tf.Variable(tf.random.uniform(shape=[], minval=-3.0, maxval=3.0), dtype=tf.float32)

for i in range(1000):  # Adjust the number of iterations as needed
    with tf.GradientTape() as tape:
        y_pred = hypothesis(x_train, w)
        cost = cost_function(y_train, y_pred)

    gradients = tape.gradient(cost, [w])
    optimizer.apply_gradients(zip(gradients, [w]))

    w_val.append(w.numpy())
    cost_val.append(cost.numpy())

plt.plot(w_val, cost_val)
plt.xlabel('Weight (w)')
plt.ylabel('Cost')
plt.title('Cost Function vs. Weight')
plt.show()
