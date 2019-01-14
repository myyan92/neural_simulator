from model_fc_concat import Model as Model_fc_concat
from model_fc_add import Model as Model_fc_add
from model_conv_add import Model as Model_conv_add
from model_linear import Model as Model_linear
from model_biLSTM_concat import Model as Model_biLSTM
from model_conditional_biLSTM_concat import Model as Model_cond_biLSTM

def Model(type):
    if type=='fc_concat':
        return Model_fc_concat()
    elif type=='fc_add':
        return Model_fc_add()
    elif type=='conv_add':
        return Model_conv_add()
    elif type=='linear':
        return Model_linear()
    elif type=='LSTM':
        return Model_biLSTM()
    elif type=='cond_LSTM':
        return Model_cond_biLSTM()
    else:
        raise ValueError('no such model type')
