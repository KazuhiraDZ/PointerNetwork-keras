from keras.models import Model
from keras.layers import Input, Lambda, Concatenate, Maximum, Dense, Embedding, Reshape, GRU, add, merge, LSTM, Dropout, \
    BatchNormalization, Activation, concatenate, Add, multiply, MaxPooling1D, Conv1D, Flatten, Bidirectional, CuDNNGRU, \
    RepeatVector, Permute, TimeDistributed, dot

import keras.backend as K



# This is the ICSE'19 submission version.  Use this model to reproduce:

# LeClair, A., Jiang, S., McMillan, C., "A Neural Model for Generating
# Natural Language Summaries of Program  Subroutines", in Proc. of the
# 41st ACE/IEEE International Conference on Software Engineering 
# (ICSE'19), Montreal, QC, Canada, May 25-31, 2019. 

class AstAttentionGRUModel:
    def __init__(self, config=None):

        if not config:
            config = {}
        # override default data sizes to what was used in the ICSE paper
        config['tdatlen'] = 50
        #config['smllen'] = 10
        config['comlen'] = 13
        config['use_pointer'] = True
        config['use_coverage'] = False

        self.config = config
        self.tdatvocabsize = config['tdatvocabsize']
        self.comvocabsize = config['comvocabsize']
        self.smlvocabsize = config['smlvocabsize']
        self.tdatlen = config['tdatlen']
        self.comlen = config['comlen']
        self.smllen = config['smllen']
        #
        # self.tdatvocabsize = 50000
        # self.comvocabsize = 50000
        # self.smlvocabsize = 72
        #
        # self.tdatlen = config['tdatlen']
        # self.comlen = config['comlen']
        # self.smllen = config['smllen']

        self.embdims = 100
        #self.smldims = 10
        self.recdims = 256

        self.config['num_input'] = 2
        self.config['num_output'] = 1

        # self.pointer = PointerLSTM(hidden_shape=self.recdims, units=self.recdims)

    def create_model(self):

        dat_input = Input(shape=(self.tdatlen,))
        com_input = Input(shape=(self.comlen,))
        #sml_input = Input(shape=(self.smllen,))
        max_src_oov_inp = Input(shape=(1,), dtype='int32')
        max_src_oov = Lambda(lambda x: x[0][0])(max_src_oov_inp)
        print("max_src_oov_input:{}, max_src_oov_input:{}".format(max_src_oov_inp, max_src_oov))
        enc_batch_extend_vocab = Input(shape=(self.tdatlen,), dtype='int32')

        ee = Embedding(output_dim=self.embdims, input_dim=self.tdatvocabsize, mask_zero=False)(dat_input) # source code
        de = Embedding(output_dim=self.embdims, input_dim=self.comvocabsize, mask_zero=False)(com_input) # comment

        enc = CuDNNGRU(self.recdims, return_state=True, return_sequences=True)  # text part
        encout, state_h = enc(ee)
        dec = CuDNNGRU(self.recdims, return_sequences=True) # comment part
        decout = dec(de, initial_state=state_h)

        attn = dot([decout, encout], axes=[2, 2])  # attn of text
        attn = Activation('softmax')(attn)
        text_context = dot([attn, encout], axes=[2, 1])

        context = concatenate([text_context, decout])

        concate_1 = concatenate([text_context, decout, de])
        out = TimeDistributed(Dense(self.recdims, activation="relu"))(context)

        # V^T*(W1*x_i + W2*h^* + W3*s_i)
        # self.p_gens = list()
        p_gen = TimeDistributed(Dense(1))(concate_1)
        p_gen = Lambda(lambda x: K.mean(x, axis=1))(p_gen) # compute the mean the
        p_gen = Activation('sigmoid')(p_gen)

        out = Flatten()(out)
        out = Dense(self.comvocabsize, activation="softmax")(out)  # output dist

        attn_concate = Lambda(lambda x: K.mean(x, axis=1))(
            attn)  # [batch_size, dec_length, enc_length] -> [batch_size, enc_length]

        if self.config['use_pointer']:
            final_dists = self._calc_final_dist(out, attn_concate, p_gen, max_src_oov, enc_batch_extend_vocab)
        else:
            final_dists = out

        print("final_dists: {}".format(final_dists))

        if self.config['use_pointer']:
            model = Model(inputs=[dat_input, com_input, max_src_oov_inp, enc_batch_extend_vocab],
                      outputs=final_dists)
        else:
            model = Model(inputs=[dat_input, com_input],
                          outputs=final_dists)

        #(loss_functions, loss_weights) = ([_coverage_loss, _loss],
        #                                  [1.,1.]) if self.config['use_coverage'] else ([_loss],[1.])

        if self.config['use_pointer']:
            model.compile(loss=_loss, optimizer='adam', metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.config, model


    def _calc_final_dist(self, vocab_dists, attn_dists, p_gen, max_src_oov, enc_batch_extend_vocab):
        """

        :param vocab_dists: List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file. default max_dec_steps is 1
        :param attn_dists: List length max_dec_steps of (batch_size, attn_len) arrays. default max_dec_steps is 1
        :return:
        """

        WeightMultLayer = Lambda(lambda x: x[0] * x[1])
        SupWeightMultLayer = Lambda(lambda x: (1 - x[0]) * x[1])
        DistPlus = Lambda(lambda x: x[0] + x[1])

        # 合并第一维度
        ConcatenateAxis1 = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))

        vocab_dists = WeightMultLayer([p_gen, vocab_dists])
        attn_dists_weighted = SupWeightMultLayer([p_gen, attn_dists])
        print("vocab_dists:{}".format(vocab_dists))
        print(K.shape(vocab_dists))
        bz = Lambda(lambda x: K.shape(x)[0])(vocab_dists)  # hidden_dim
        print("batchsize:{},  max_src_oov:{}".format(bz, max_src_oov))

        def get_zeros_tensor(shape):
            return K.zeros(shape)

        extra_zeros = Lambda(get_zeros_tensor)([bz, max_src_oov])

        extended_vsize = Lambda(lambda x: self.comvocabsize + x)(max_src_oov)
        print("extra_zeros:{}, vocab_dist:{}".format(extra_zeros, vocab_dists))
        vocab_dists_extended = ConcatenateAxis1([vocab_dists, extra_zeros])

        # Project the values in the attention distributions onto the appropriate entries in the final distributions
        # This means that if a_i = 0.1 and the ith encoder word is w,
        # and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
        # This is done for each decoder timestep.
        # This is fiddly; we use tf.scatter_nd to do the projection

        shape = [bz, extended_vsize]

        def preparation(x):
            batch_nums = K.tf.range(0, limit=bz)  # shape (batch_size)
            batch_nums = K.tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
            attn_len = K.tf.shape(enc_batch_extend_vocab)[1]  # number of states we attend over
            batch_nums = K.tf.tile(batch_nums, multiples=[1, attn_len])
            indices = K.tf.stack((batch_nums, enc_batch_extend_vocab), axis=2)
            return indices

        indices = Lambda(preparation)([])
        ScatterNdList = Lambda(
            lambda x: K.tf.scatter_nd(indices, x, shape=shape, name='making_attn_dists_projected_at_step_0'))
        attn_dists_projected = ScatterNdList(attn_dists_weighted)
        print("attn_dists_projected:{}".format(attn_dists_projected))

        final_dists = DistPlus([vocab_dists_extended, attn_dists_projected])

        def _add_epsilon(epsilon=1e-9):
            # return add-epsilon layer
            _AddEpsilon = Lambda(lambda x: x + K.tf.ones_like(x) * epsilon)
            return _AddEpsilon

        AddEpsilon = _add_epsilon()
        final_dists = AddEpsilon(final_dists)

        return final_dists


def _mask_and_avg(values, padding_mask=None):
    """Applies mask to values then returns overall average (a scalar)
    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
    Returns:
      a scalar
    """
    #dec_lens = K.tf.reduce_sum(padding_mask, axis=1, name='dec_lens')  # shape batch_size. float32
    # values_per_step = []
    # for dec_step, v in enumerate(values):
    #     values_per_step.append(v * padding_mask[:, dec_step])
    values_per_ex = sum(values)  # shape (batch_size); normalized value for each batch member
    return K.tf.reduce_mean(values_per_ex, name='reduce_mean_in_mask_avg')  # overall average

def calc_loss_at_timestep_t(range_batch, t, dist_at_t, _target_batch):
    # Args:
    #     dist_at_t: the distribution that we get step by step
    # losses: loss of all samples in a batch at time step t
    targets = K.tf.strided_slice(_target_batch, [0, t], [K.tf.shape(_target_batch)[0], t + 1], shrink_axis_mask=2,
                                 name='slicing_for_targets_in_calc_loss_at_timestep_t')  # shape: (batch_size, )
    indices = K.tf.stack((range_batch, targets), axis=1)  # shape (batch_size, 2)
    gold_probs = K.tf.gather_nd(dist_at_t, indices)  # shape (batch_size). prob of correct words on this step
    losses = -K.tf.log(gold_probs)
    return losses

def _loss(y_true, y_pred):
    # Params:
    # y_pred : final_dists, distributions of words, shape (batch_size, time_steps, vocab_size) (float)
    # y_true : indices of true words, shape (batch_size, time_steps, ) (int)
    y_true = K.tf.expand_dims(y_true,axis=1)
    y_pred = K.tf.expand_dims(y_pred,axis=1)

    y_true = K.tf.cast(y_true[:, :, 0], 'int32', 'cast_to_int_in_loss')

    loss_per_step = []
    _batchsize = K.shape(y_pred)[0]
    batch_nums = K.tf.range(0, limit=_batchsize)  # shape: (batch_size, )

    # unstack by steps
    for dec_step, dist in enumerate(K.tf.unstack(y_pred, axis=1)):
        losses = calc_loss_at_timestep_t(batch_nums, dec_step, dist, y_true)
        loss_per_step.append(losses)
    _loss_ret = _mask_and_avg(loss_per_step)

    return _loss_ret

def _coverage_loss(y_true, y_pred):
    # Params:
    # y_pred : attn_dists, distributions of words, shape (batch_size, time_steps, vocab_size) (float)
    # y_true : indices of true words, shape (batch_size, time_steps, vocab_size ) (int)
    # keras requires y_true and y_pred to be the same shape,
    # thus y_true is repeated vocab_size times on the last dim

    _y_pred = K.tf.unstack(y_pred, axis=1, name='unstacking_attn_dists_in_coverage_loss')
    coverage = K.tf.zeros_like(_y_pred[0])
    covlosses = []
    for a in _y_pred:
        covloss = K.tf.reduce_sum(K.tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    _coverage_loss_ret = _mask_and_avg(covlosses)
    return _coverage_loss_ret
