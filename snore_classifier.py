#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import pyaudio
import speech_data
import numpy

# Simple spoken digit recognition demo, with 98% accuracy in under a minute

# Training Step: 544  | total loss: 0.15866
# | Adam | epoch: 034 | loss: 0.15866 - acc: 0.9818 -- iter: 0000/1000

mfcc=True

def validateWav(demo_file):
    demo = speech_data.load_wav_file(speech_data.snore_train_path + demo_file,140000,1,mfcc)
    result = model.predict([demo])
    result = numpy.argmax(result)
    print("predicted digit for %s : result = %d " % (demo_file, result))
    return result




batch=speech_data.wave_batch_snore(300,mfcc)
X,Y=next(batch)


number_classes=2 # Digits

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

if mfcc :
    width = 20;
    height = 200
    net = tflearn.input_data(shape=[None, width,height])
    net = tflearn.lstm(net, 128*4,dropout=0.5)
    net = tflearn.fully_connected(net, number_classes,activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.0001)
else :
    net = tflearn.input_data(shape=[None, 140000])
    net = tflearn.fully_connected(net, 248)
    net = tflearn.normalization.batch_normalization(net)
    net = tflearn.fully_connected(net, 64)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, number_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.0001)

model = tflearn.DNN(net)
for i in range(1,500) :
    model.fit(X, Y,n_epoch=25,show_metric=True,snapshot_step=100,)
    X, Y = next(batch)
    r1=validateWav("nosnore3.wav")
    r2 =validateWav("nosnore4.wav")
    r3 =validateWav("snore1.wav")
    r4 =validateWav("snore2.wav")
    total = r1+r2+r3+r4
    if total==2 : input("found?")

model.save("model/snore")
model.load("model/snore")

validateWav("nosnore1.wav")
validateWav("nosnore2.wav")
validateWav("nosnore3.wav")
validateWav("nosnore4.wav")
validateWav("snore1.wav")
validateWav("snore2.wav")
validateWav("snore3.wav")
validateWav("snore4.wav")