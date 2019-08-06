# from keras import initializations
import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine import InputSpec
from keras.layers import LSTM, GRU
from keras.layers.recurrent import Recurrent
from funcom.tdd import _time_distributed_dense # todo: 这里import了新的函数

class PointerLSTM(GRU):
    def __init__(self, hidden_shape, *args  , **kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        super(PointerLSTM, self).__init__(*args, **kwargs)

    def get_initial_states(self, x_input):
        return Recurrent.get_initial_state(self, x_input)

    # todo: 这里仿照get_initial_states改了
    def get_constants(self, x_input):
        return Recurrent.get_constants(self, x_input)

    # todo: 这里仿照get_initial_states改了
    def preprocess_input(self, x_input):
        return Recurrent.preprocess_input(self, x_input)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        print("input_shape:",input_shape)
        print("-----------------------------")
        # init = initializations.get('orthogonal')
        self.W1 = self.add_weight(name="W1",
                                  shape=(self.hidden_shape, 1),  # hidden: 64
                                  initializer="uniform",
                                  trainable=True)
        self.W2 = self.add_weight(name="W2",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.W3 = self.add_weight(name="W3",
                                  shape=(self.hidden_shape, 1),
                                  initializer="uniform",
                                  trainable=True)
        self.vt = self.add_weight(name="vt",
                                  shape=(input_shape[1], 1),    # input_shape[1]:3
                                  initializer='uniform',
                                  trainable=True)
        self._trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):    # input [bz, steps, hidden]
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1] - 1, :]                   #
        x_input = K.repeat(x_input, input_shape[1])       #
        initial_states = self.get_initial_states(x_input) # get inital_state based on x_input(may be replaced later as we have specific state)

        constants = self.get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])


        print("latst_output:{}, outputs:{}, states:".format(last_output, outputs, states))
        return outputs

    def step(self, x_input, states):
        # print "x_input:", x_input, x_input.shape
        # <TensorType(float32, matrix)>

        input_shape = self.input_spec[0].shape
        en_seq = states[-1]

        _, [h] = self.cell.call(x_input, states[:-1]) # todo: 这里改了

        # vt*tanh(W1*e+W2*d)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = _time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = _time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        print("E:{} , D:{}, U:{}".format(Eij, Dij, U))
        U = K.squeeze(U, 2)

        # make probability tensor
        pointer = softmax(U)
        #print(pointer, h, c, self.vt)
        return pointer, [h]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

