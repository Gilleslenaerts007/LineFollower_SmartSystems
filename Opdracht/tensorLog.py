import tensorflow as tf

a = tf.constant([10])
b = tf.constant([20])
c=tf.add(a,b)

logs_dir = './logs'

with tf.session() as sess:
    criter = tf.summart.FileWriter(logs_dir, sess.graph)
    result=sess.run(c)
    print('outcome:', result)
    
writer.close()
