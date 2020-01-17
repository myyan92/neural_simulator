from neural_simulator.model_fc_concat import Model as Model_fc_concat
from neural_simulator.model_fc_add import Model as Model_fc_add
from neural_simulator.model_conv_add import Model as Model_conv_add
from neural_simulator.model_linear import Model as Model_linear
from neural_simulator.model_biLSTM_concat import Model as Model_biLSTM
from neural_simulator.model_conditional_biLSTM_concat import Model as Model_cond_biLSTM
from neural_simulator.model_biLSTM_attention import Model as Model_biLSTM_attention
from neural_simulator.model_GRU_attention import Model as Model_GRU_attention
#from neural_simulator.model_graphnet import Model as Model_graphnet
#from neural_simulator.model_edgefunc import Model as Model_edgefunc # for debuging
#from neural_simulator.model_nodefunc import Model as Model_nodefunc # for debuging
#from neural_simulator.model_substep import Model as Model_substep # for debuging

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
    elif type=='LSTM_attention':
        return Model_biLSTM_attention()
    elif type=='GRU_attention':
        return Model_GRU_attention()
    elif type=='cond_LSTM':
        return Model_cond_biLSTM()
#    elif type=='graphnet':
#        return Model_graphnet()
#    elif type=='edgefunc':
#        return Model_edgefunc()
#    elif type=='nodefunc':
#        return Model_nodefunc()
#    elif type=='substep':
#        return Model_substep()
    else:
        raise ValueError('no such model type')
