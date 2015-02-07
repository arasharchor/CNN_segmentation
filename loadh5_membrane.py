import numpy 
import h5py
import sys
import gzip

def load_data_segmentation():
    '''
    sub1='/home/zxx/CNN_zxx/cnn_zxx/sub1.h5'
    sub1_label='/home/zxx/CNN_zxx/cnn_zxx/sub1_label.h5'

    sub1_512='/home/zxx/CNN_zxx/cnn_zxx/sub1.h5'
    sub1_label_512='/home/zxx/CNN_zxx/cnn_zxx/sub1_label.h5'

    sub3_512='/home/zxx/CNN_zxx/cnn_zxx/sub3_512.h5'
    sub3_label_512='/home/zxx/CNN_zxx/cnn_zxx/sub3_label_512.h5'
    '''
    print "loading data from hdf5 file ..."
    baseDir='/mnt/zxx_2T/CNN/'
    sub1= baseDir + 'train-volume01.h5'
    sub1_label= baseDir + 'train-label01.h5'
    sub1_512= baseDir + 'train-volume09.h5'
    sub1_label_512= baseDir + 'train-label09.h5'
    sub3_512= baseDir + 'train-volume10.h5'
    sub3_label_512= baseDir + 'train-label10.h5'
    print "load file:", sub1, sub1_label
    print "load file:", sub1_512, sub1_label_512
    print "load file:", sub3_512, sub3_label_512

    combo_h5 = h5py.File(sub1,'r')
    trainx=combo_h5['train'].value
    trainx = numpy.divide(trainx, 256)

    combo_h5 = h5py.File(sub1_label,'r')
    trainy=combo_h5['train'].value

    combo_h5 = h5py.File(sub1_512,'r')
    validx=combo_h5['train'].value
    validx = numpy.divide(validx, 256)

    combo_h5 = h5py.File(sub1_label_512,'r')
    validy=combo_h5['train'].value
    
    '''
    validx = trainx
    validy = trainy
    '''
    combo_h5 = h5py.File(sub3_512,'r')
    testx=combo_h5['train'].value
    testx= numpy.divide(testx, 256)

    combo_h5 = h5py.File(sub3_label_512,'r')
    testy=combo_h5['train'].value

    train = (trainx, trainy)
    valid = (validx, validy)
    test = (testx, testy)
    print "train_x", trainx.shape
    print "train_y", trainy.shape
    return (train, valid, test)

def printSet(trainx_set):
  #print len(trainx_set)
  for item in trainx_set:
      print item
def load_train(train_file, label_file):
    combo_h5 = h5py.File(train_file,'r')
    trainx=combo_h5['train'].value
    trainx = numpy.divide(trainx, 256)
    trainx = trainx.astype(numpy.float32)

    combo_h5 = h5py.File(label_file,'r')
    trainy=combo_h5['train'].value
    return (trainx, trainy)

if __name__ == '__main__':
    sets = load_data_segmentation()
    '''
    train, test, valid = sets
    trainx, trainy = train
    print trainx.shape
    print trainx[0]
    print trainy[512*256:512*258]
    '''
    #trainx, trainy = sets
    #printSet(trainy)
