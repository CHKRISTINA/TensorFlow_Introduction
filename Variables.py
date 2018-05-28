import tensorflow as tf

#Create a simple counter

a = tf.Variable(0) #change in each step
i = tf.constant(1) #the step
new_value = tf.add(a,i)
update = tf.assign(a, new_value)

# Variables need to be initialized before a graph can be run in a session, which can be done using 'tf.global_variables_initializer'.
inizialize_variables = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(inizialize_variables)
    for _ in range(3):
        session.run(update)
        print("Step", session.run(a))


