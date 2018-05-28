import tensorflow as tf

# A placeholder like a variable that won't actually receive its data until a later point.
# In a placeholder you need to specify the data type, as well as the data type's precision in terms of the number of bits.
a = tf.placeholder(tf.float32)

b = 2*a

# The placeholder doesn't hold a value.
# We can solve this by passing an extra argument when we call the session with the argument 'feed_dict',
# a dictionary that contains each placeholder name followed by its respective data:
with tf.Session() as session:
    result = session.run(b, feed_dict={a: 5.5}) #
    print(result)
