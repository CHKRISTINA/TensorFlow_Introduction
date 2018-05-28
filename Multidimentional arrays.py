import tensorflow as tf

#Multidimentional arrays with Tensorflow

Scalar = tf.constant([2])
Vector = tf.constant([5,6,2])
Matrix = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
Tensor = tf.constant([[[1,2,3],[4,5,6],[7,8,9]], [[2,2,3],[5,5,6],[8,8,9]], [[1,2,2],[4,5,5],[7,8,8]]])

with tf.Session() as session:
    print("Scalar (1 entity)\n", session.run(Scalar))
    print("Vector (3 entities)\n", session.run(Vector))
    print("Matrix (3*3 entities)\n", session.run(Matrix))
    print("Tensor (3*3*3 entities)\n", session.run(Tensor))



