import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_states(model, processed_input, initial_hidden):
    all_hidden_states = tf.scan(model, processed_input, initializer=initial_hidden, name='states')
    all_hidden_states = all_hidden_states[:, 0, :, :]
    return all_hidden_states
    
def get_output(Wo, bo, hidden_state):
    output = tf.nn.relu(tf.matmul(hidden_state, Wo) + bo)
    return output

class OGRU_cell(object):

    def __init__(self, input_nodes, hidden_unit, output_nodes):

        self.input_nodes = input_nodes
        self.hidden_unit = hidden_unit
        self.output_nodes = output_nodes
        #weight and bias initialization
        self.Wx = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))

        self.Wr = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.br = tf.Variable(tf.truncated_normal([self.hidden_unit], mean=1))
        
        self.Wz = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.bz = tf.Variable(tf.truncated_normal([self.hidden_unit], mean=1))

        self.Wh = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))

        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_unit, self.output_nodes], mean=1, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_nodes], mean=1, stddev=.01))

        self._inputs = tf.placeholder(tf.float32,shape=[None, None, self.input_nodes], name='inputs')
        
        batch_input_ = tf.transpose(self._inputs, perm=[2, 0, 1])
        self.processed_input = tf.transpose(batch_input_)

        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_nodes, hidden_unit]))

    def Gru(self, previous_hidden_state, x):

        r = tf.sigmoid(tf.matmul(x, self.Wr) + self.br)
        z = tf.sigmoid(tf.multiply(tf.matmul(x, self.Wz), r) + self.bz)

        h_ = tf.tanh(tf.matmul(x, self.Wx) +
                     tf.matmul(previous_hidden_state, self.Wh) * r)

        current_hidden_state = tf.multiply( (1 - z), h_) + tf.multiply(previous_hidden_state, z)

        return current_hidden_state

    def get_states(self):
        all_hidden_states = tf.scan(self.Gru, self.processed_input, initializer=self.initial_hidden, name='states')
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self):
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs