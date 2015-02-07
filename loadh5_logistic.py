import numpy
import h5py

def load_weight(weight_file):
    #weight_file = 'weight.h5'
    combo_h5 = h5py.File(weight_file, 'r')
    weight = combo_h5['weight'].value
    bias = combo_h5['bias'].value
    combo_h5.close()
    print 'weight shape :',weight.shape
    print 'bias shape:', bias.shape
    return (weight, bias)

if __name__ == '__main__':
    params = load_weight('weight.h5')
    weight = params[0]
    bias = params[1]
    print type(weight)
    print type(bias)

    print weight.shape
    print  bias.shape
