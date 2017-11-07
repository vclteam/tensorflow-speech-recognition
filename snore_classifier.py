#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import speech_data
import tensorflow as tf
import os


def validateWav(demo_file):
    demoData = speech_data.load_wav_file(
        speech_data.snore_train_path + demo_file, 0)
    result = model.predict([demoData])
    if (result[0][1] > 0.6):
        rc = 1
        print(demo_file + ":" +
              str((result[0][0])) + " / " + str((result[0][1])) + ": ISSNORE")
    else:
        print(demo_file + ":" +
              str((result[0][0])) + " / " + str((result[0][1])))
    rc = 0

    return rc


batch = speech_data.wave_batch_snore(128, True)
testbatch = speech_data.wave_batch_snore(24, True)

learning_rate = 0.001
number_classes = 2  # Digits

# Classification
tflearn.init_graph(gpu_memory_fraction=0.9)

width = 36
height = 36
convnet = tflearn.input_data(shape=[None, 36, 36, 1], name='input')

convnet = tflearn.conv_2d(convnet, 64, 3, activation='relu')
convnet = tflearn.conv_2d(convnet, 32, 3 , activation='relu')
convnet = tflearn.max_pool_2d(convnet, 2)
convnet = tflearn.dropout(convnet, 0.50)

convnet = tflearn.conv_2d(convnet, 64, 3, activation='relu')
convnet = tflearn.conv_2d(convnet, 32, 3 , activation='relu')
convnet = tflearn.max_pool_2d(convnet, 2)
convnet = tflearn.dropout(convnet, 0.50)

convnet = tflearn.conv_2d(convnet, 64, 3, activation='relu')
convnet = tflearn.conv_2d(convnet, 32, 3 , activation='relu')
convnet = tflearn.max_pool_2d(convnet, 2)
convnet = tflearn.dropout(convnet, 0.50)


convnet = tflearn.fully_connected(convnet, 256, activation='tanh')
convnet = tflearn.fully_connected(convnet, 2, activation='softmax')
convnet = tflearn.regression(convnet, optimizer='adam', loss='categorical_crossentropy',
                             name='targets')


model = tflearn.DNN(convnet, tensorboard_verbose=0)
try:
    if len(os.listdir("model/")) > 0:
        model.load("model/snore")
except:
    print("NO DATA")


for i in range(300):
    X, Y = next(batch)
    Xtest, Ytest = next(testbatch)

    res = model.fit(X, Y, validation_set=(Xtest, Ytest),
                    n_epoch=25, show_metric=True)

    # nr1 = validateWav("nosnore1.wav")
    # nr2 = validateWav("nosnore2.wav")
    # nr3 = validateWav("nosnore3.wav")
    # nr4 = validateWav("nosnore4.wav")

    r1 = validateWav("snore1.wav")
    r2 = validateWav("snore2.wav")
    r3 = validateWav("snore3.wav")
    r4 = validateWav("snore4.wav")

    model.save("model/snore")
