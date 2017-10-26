#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import speech_data


def validateWav(demo_file):
    demoData = speech_data.load_wav_file(speech_data.snore_train_path + demo_file,140000,1,True)
    result = model.predict([demoData])
    if (result[0][1]>0.6) :
        rc =  1
        print demo_file+":"+str((result[0][0]))+" / "+str((result[0][1]))+ ": ISSNORE"
    else :print demo_file+":"+str((result[0][0]))+" / "+str((result[0][1]))
    rc = 0

    return rc




batch=speech_data.wave_batch_snore(256,True)
testbatch=speech_data.wave_batch_snore(32,True)

learning_rate = 0.0001
number_classes=2 # Digits

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

width = 40;
height = 200
convnet = tflearn.input_data(shape=[None, width, height, 1], name='input')
convnet = tflearn.conv_2d(convnet, 32, 5, activation='relu')
convnet = tflearn.max_pool_2d(convnet, 5)
convnet = tflearn.conv_2d(convnet, 64, 5, activation='relu')
convnet = tflearn.max_pool_2d(convnet, 5)
convnet = tflearn.fully_connected(convnet, 1024, activation='relu')
convnet = tflearn.dropout(convnet, 0.8)
convnet = tflearn.fully_connected(convnet, 2, activation='softmax')
convnet = tflearn.regression(convnet, optimizer='adam',  loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
try :
    model.save("model/snore")
except :
    print "NO DATA"

for i in range(200) :
    X, Y = next(batch)
    Xtest, Ytest = next(testbatch)

    res = model.fit(X, Y,validation_set=(Xtest,Ytest),n_epoch=100,show_metric=True)

    nr1=validateWav("nosnore1.wav")
    nr2 =validateWav("nosnore2.wav")
    nr3=validateWav("nosnore3.wav")
    nr4 =validateWav("nosnore4.wav")

    r1 =validateWav("snore1.wav")
    r2 =validateWav("snore2.wav")
    r3 =validateWav("snore3.wav")
    r4 =validateWav("snore4.wav")

    model.save("model/snore")

    if (r1==1) and (r2==1) and (r3==1) and (r4==1) :input("found?")


#model.load("model/snore")

