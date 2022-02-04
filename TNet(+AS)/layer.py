# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from nn_utils import *
import pickle

class LSTM:
    def __init__(self, bs, n_in, n_out, name):
        """
        :param bs: batch size
        :param n_in: input size
        :param n_out: hidden size
        :param name: alias of layer
        """
        self.bs = bs
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        # W shape: (n_in, 4*n_out)
        # U shape: (n_out, 4*n_out)
        # b shape: (4*n_out)
        self.W, self.U, self.b = lstm_init(n_in=self.n_in, n_out=self.n_out, component=name)
        self.h0 = theano.shared(value=zeros(size=(self.bs, n_out)), name='h0')
        self.c0 = theano.shared(value=zeros(size=(self.bs, n_out)), name='c0')
        self.params = [self.W, self.U, self.b]

    def __str__(self):
        return "%s: LSTM(%s, %s)" % (self.name, self.n_in, self.n_out)

    __repr__ = __str__

    def __call__(self, x):
        """

        :param x: input tensor, shape: (bs, seq_len, n_in)
        :return: generated hidden states
        """
        h0 = T.zeros_like(self.h0)
        c0 = T.zeros_like(self.c0)
        rnn_input = x.dimshuffle(1, 0, 2)
        [H, _], _ = theano.scan(fn=self.recurrence, sequences=rnn_input, outputs_info=[h0, c0])
        return H.dimshuffle(1, 0, 2)

    def recurrence(self, xt, htm1, ctm1):
        """

        :param xt: x[t] \in (bs, n_in)
        :param htm1: h[t-1] \in (bs, n_out)
        :param ctm1: c[t-1] \in (bs, n_out)
        :return:
        """
        Wx = T.dot(xt, self.W)
        Uh = T.dot(htm1, self.U)
        Sum_item = Wx + Uh + self.b
        it = T.nnet.hard_sigmoid(Sum_item[:, :self.n_out])
        ft = T.nnet.hard_sigmoid(Sum_item[:, self.n_out:2*self.n_out])
        ct_tilde = T.tanh(Sum_item[:, 2*self.n_out:3*self.n_out])
        ot = T.nnet.hard_sigmoid(Sum_item[:, 3*self.n_out:])
        ct = ft * ctm1 + it * ct_tilde
        ht = ot * T.tanh(ct)
        return ht, ct

class Linear:
    """
    fully connected layer
    """
    def __init__(self, n_in, n_out, name, use_bias=True):
        """

        :param n_in: input size
        :param n_out: output size
        :param name: layer name
        :param use_bias: use bias or not
        """
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.use_bias = use_bias
        # sample weight from uniform distribution [-INIT_RANGE, INIT_RANGE]
        # initialize bias as zero vector
        self.W = theano.shared(value=uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_in, n_out)), name="%s_W" % name)
        self.b = theano.shared(value=zeros(size=n_out), name="%s_b" % name)
        self.params = [self.W]
        if self.use_bias:
            self.params.append(self.b)

    def __str__(self):
        return "%s: Linear(%s, %s)" % (self.name, self.n_in, self.n_out)

    __repr__ = __str__

    def __call__(self, x, bs=None):
        """

        :param x: input tensor, shape: (bs, *, n_in)
        :return:
        """
        if bs is None:
            output = T.dot(x, self.W)
        else:
            # current shape: (bs, n_in, n_out)
            padded_W = T.tile(self.W, (bs, 1, 1))
            # output shape: (bs, seq_len, n_out)
            output = T.batched_dot(x, padded_W)
        if self.use_bias:
            output = output + self.b
        return output

class Dropout:
    def __init__(self, p):
        self.p = p
        self.retain_prob = 1 - p

    def __str__(self):
        return "Dropout(%s)" % (1.0 - self.retain_prob)

    __repr__ = __str__

    def __call__(self, x):
        """

        :param x: input tensor
        :return:
        """
        rng = np.random.RandomState(1344)
        srng = RandomStreams(rng.randint(999999))
        mask = srng.binomial(size=x.shape, n=1, p=self.retain_prob, dtype='float32')
        scaling_factor = 1.0 / (1.0 - self.p)
        return x * mask

class CPT_AS:
    # Context-Preserving Transformation with Adaptive-Scaling
    def __init__(self, bs, sent_len, n_in, n_out, name):
        self.bs = bs
        self.sent_len = sent_len
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.fc_gate = Linear(n_in=self.n_in, n_out=self.n_out, name="Gate")
        self.fc_trans = Linear(n_in=2*self.n_in, n_out=self.n_out, name="Trans")
        # for model with highway transformation
        self.layers = [self.fc_gate, self.fc_trans]
        # for model without highway transformation
        #self.layers = [self.fc_trans]
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __str__(self):
        des_str = 'CPT(%s, %s)' % (self.n_in, self.n_out)
        for layer in self.layers:
            des_str += ', %s' % layer
        return des_str

    __repr__ = __str__

    def __call__(self, x, xt):
        """

        :param x: input sentence, shape: (bs, sent_len, n_in)
        :param xt: input target, shape: (bs, target_len, n_in)
        :return:
        """
        trans_gate = T.nnet.hard_sigmoid(self.fc_gate(x, bs=self.bs))
        # (sent_len, bs, n_in)
        x_ = x.dimshuffle(1, 0, 2)
        # (bs, n_in, target_len)
        xt_ = xt.dimshuffle(0, 2, 1)
        x_new = []
        for i in range(self.sent_len):
            # (bs, n_in)
            xi = x_[i]
            # attention
            # (bs, target_len, n_in) batched_dot (bs, n_in, 1) -> (bs, target_len, 1)
            alphai = T.nnet.softmax(T.batched_dot(xt, xi.dimshuffle(0, 1, 'x')).flatten(2))
            # (bs, n_in, target_len) batched_dot (bs, target_len, 1) -> (bs, n_in, 1)
            ti = T.batched_dot(xt_, alphai.dimshuffle(0, 1, 'x')).flatten(2)
            # (bs, 2 * n_in) -> (bs, n_out)
            xi_new = T.tanh(self.fc_trans(x=T.concatenate([xi, ti], axis=1)))
            x_new.append(xi_new)
        # (bs, sent_len, n_out)
        x_new = T.stack(x_new, axis=0).dimshuffle(1, 0, 2)
        return trans_gate * x_new + (1.0 - trans_gate) * x

class CPT_LF:
    # Context-Preserving Transformation with Lossless-Forwarding
    def __init__(self, bs, sent_len, n_in, n_out, name):
        self.bs = bs
        self.sent_len = sent_len
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.fc_trans = Linear(n_in=2*self.n_in, n_out=self.n_out, name="Trans")
        self.layers = [self.fc_trans]
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __str__(self):
        des_str = 'CPT(%s, %s)' % (self.n_in, self.n_out)
        for layer in self.layers:
            des_str += ', %s' % layer
        return des_str

    __repr__ = __str__

    def __call__(self, x, xt):
        """

        :param x: input sentence, shape: (bs, sent_len, n_in)
        :param xt: input target, shape: (bs, target_len, n_in)
        :return:
        """
        # (max_len, bs, n_in)
        x_ = x.dimshuffle(1, 0, 2)
        # (bs, n_in, target_len)
        xt_ = xt.dimshuffle(0, 2, 1)
        x_new = []
        for i in range(self.sent_len):
            # (bs, n_in)
            xi = x_[i]
            # shape: (bs, sent_len)
            alphai = T.nnet.softmax(T.batched_dot(xt, xi.dimshuffle(0, 1, 'x')).flatten(2))
            ti = T.batched_dot(xt_, alphai.dimshuffle(0, 1, 'x')).flatten(2)
            xi_new = T.nnet.relu(self.fc_trans(x=T.concatenate([xi, ti], axis=1)))
            x_new.append(xi_new)
        x_new = T.stack(x_new, axis=0).dimshuffle(1, 0, 2)
        return x_new + x

class TNet:
    """
    Transformation Networks for Target-Oriented Sentiment Analysis
    """
    def __init__(self, args):
        if args.ds_name != '14semeval_rest' and args.ds_name != '14semeval_rest_val':
            self.seed = 14890
        else:
            self.seed = 11456
        self.bs = args.bs
        self.n_in = args.dim_w
        self.n_rnn_out = args.dim_h
        self.embedding_weights = args.embeddings
        self.n_y = args.dim_y
        self.dropout_rate = args.dropout_rate
        self.sent_len = args.sent_len
        self.target_len = args.target_len
        self.ds_name = args.ds_name
        self.connection_type = args.connection_type
        self.lamda = args.lamda
        
        self.set_seed()
        self.Words = theano.shared(value=np.array(self.embedding_weights, 'float32'), name="embedding") # 词向量
        self.Dropout_ctx = Dropout(p=self.dropout_rate) # 输入句子Dropout，0.3
        self.Dropout_tgt = Dropout(p=self.dropout_rate) # 输入目标词Dropout，0.3
        self.Dropout = Dropout(p=self.dropout_rate) # 参数Dropout，0.3
        self.LSTM_ctx = LSTM(bs=self.bs, n_in=self.n_in, n_out=self.n_rnn_out, name="CTX_LSTM") # 句子的LSTM
        self.LSTM_tgt = LSTM(bs=self.bs, n_in=self.n_in, n_out=self.n_rnn_out, name="TGT_LSTM") # 目标词的LSTM
        if self.connection_type == 'AS':
            self.CPT = CPT_AS(bs=self.bs, sent_len=self.sent_len, n_in=2 * self.n_rnn_out, n_out=2 * self.n_rnn_out, name="CPT")
        else:
            self.CPT = CPT_LF(bs=self.bs, sent_len=self.sent_len, n_in=2 * self.n_rnn_out, n_out=2 * self.n_rnn_out, name="CPT")

        # 最后全连接层，分类
        self.FC = Linear(n_in=2 * self.n_rnn_out, n_out=self.n_y, name="LAST_FC")
        # 模型所有的部分
        self.layers = [self.LSTM_ctx, self.LSTM_tgt, self.CPT, self.FC]
        # 模型所有参数
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)
        #print(self.params)
        self.build_model()
        self.make_function()

    def __str__(self):
        strs = []
        for layer in self.layers:
            strs.append(str(layer))
        return ', '.join(strs)

    __repr__ = __str__

    def build_model(self):
        """
        build the computational graph of ASTN
        :return:
        """
        self.x = T.imatrix('wids') # 词id
        self.xt = T.imatrix('wids_target') # 目标词id
        self.y = T.ivector('label') # 标签
        self.pw = T.fmatrix("position_weight") # 位置权重
        self.mask = T.fmatrix("attention_mask") # !!!
        self.amask = T.fmatrix("attention_amask") # !!!
        self.avalue = T.fmatrix("attention_avalue") # !!!
        self.avalue = self.avalue / self.avalue.sum(axis=1).dimshuffle(0, 'x') # 归一化和为1
        self.is_train = T.iscalar("is_training") # 训练还是测试
        # input: input sentence, shape: (bs, sent_len, n_in)
        # input_target: input target, shape: (bs, target_len, n_in)
        input = self.Words[T.cast(self.x.flatten(), 'int32')].reshape((self.bs, self.sent_len, self.n_in))
        input_target = self.Words[T.cast(self.xt.flatten(), 'int32')].reshape((self.bs, self.target_len, self.n_in))
        # 输入句子Dropout，0.3
        # 输入目标词Dropout，0.3
        input = T.switch(T.eq(self.is_train, np.int32(1)), self.Dropout_ctx(input), input * (1 - self.dropout_rate))
        input_target = T.switch(T.eq(self.is_train, np.int32(1)), self.Dropout_tgt(input_target), input_target * (1 - self.dropout_rate))

        # model component for TNet
        rnn_input = input
        rnn_input_reverse = reverse_tensor(tensor=rnn_input)

        rnn_input_target = input_target
        rnn_input_target_reverse = reverse_tensor(tensor=rnn_input_target)
        # 双向LSTM
        H0_forward = self.LSTM_ctx(x=rnn_input)
        Ht_forward = self.LSTM_tgt(x=rnn_input_target)
        H0_backward = reverse_tensor(tensor=self.LSTM_ctx(x=rnn_input_reverse))
        Ht_backward = reverse_tensor(tensor=self.LSTM_tgt(x=rnn_input_target_reverse))
        H0 = T.concatenate([H0_forward, H0_backward], axis=2)
        Ht = T.concatenate([Ht_forward, Ht_backward], axis=2)
        # 根据论文，层数2层时效果最佳
        H1 = self.CPT(H0, Ht)
        if self.pw is not None:
            H1 = H1 * self.pw.dimshuffle(0, 1, 'x')
        H2 = self.CPT(H1, Ht)
        if self.pw is not None:
            H2 = H2 * self.pw.dimshuffle(0, 1, 'x')

        ##### attention
        # (bs, 2 * self.n_rnn_out)
        quary = T.concatenate([Ht_forward[:,-1,:], Ht_backward[:,0,:]], axis=1)
        # mask (bs, sent_len)
        ret = (1.0 - self.mask) * -1e9
        # (bs, sent_len, 2 * self.n_rnn_out) batched_dot (bs, 2 * self.n_rnn_out, 1) -> (bs, sent_len)
        self.alpha = T.nnet.softmax(T.batched_dot(H2, quary.dimshuffle(0, 1, 'x')).flatten(2) + ret)
        # (bs, 2 * self.n_rnn_out, sent_len) batched_dot (bs, sent_len, 1) -> (bs, 2 * self.n_rnn_out)
        feat = T.batched_dot(H2.dimshuffle(0, 2, 1), self.alpha.dimshuffle(0, 1, 'x')).flatten(2)
        #####
        
        # we do not use the self-implemented Dropout class
        feat_dropout = T.switch(T.eq(self.is_train, np.int32(1)), self.Dropout(feat), feat * (1 - self.dropout_rate))
        
        # softmax
        # shape: (bs, n_y)
        self.p_y_x = T.nnet.softmax(self.FC(feat_dropout))
        # 误差
        self.loss = T.nnet.categorical_crossentropy(coding_dist=self.p_y_x, true_dist=self.y).mean()
        self.aloss = (((self.alpha * self.amask - self.avalue)**2).sum(axis=1)).mean()

        # 预测
        self.pred_y = T.argmax(self.p_y_x, axis=1)

    def make_function(self):
        print("Use adam...")
        self.updates = adam(cost=self.loss, params=self.params)
        model_inputs = [self.x, self.xt, self.y, self.pw, self.mask, self.is_train]
        model_outputs = [self.pred_y, self.y, self.loss, self.alpha]
        
        self.updates_final = adam(cost=self.loss + self.lamda * self.aloss, params=self.params)
        model_inputs_final = [self.x, self.xt, self.y, self.pw, self.mask, self.amask, self.avalue, self.is_train]
        model_outputs_final = [self.pred_y, self.y, self.loss, self.aloss]
        
        self.train = theano.function(
            inputs=model_inputs,
            outputs=model_outputs,
            updates=self.updates,
        )
        
        self.train_final = theano.function(
            inputs=model_inputs_final,
            outputs=model_outputs_final,
            updates=self.updates_final,
        )
        
        self.test = theano.function(
            inputs=model_inputs,
            outputs=model_outputs
        )

    def set_seed(self):
        print("Use seed %s..." % self.seed)
        np.random.seed(self.seed)

    def save_model(self, model_file):
        params_list = [self.Words.get_value()]
        for param in self.params:
            params_list.append(param.get_value())
        pickle.dump(params_list, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, model_file):
        params_list = pickle.load(model_file)
        self.Words.set_value(params_list[0])
        for param, param_value in zip(self.params, params_list[1:]):
            param.set_value(param_value)


