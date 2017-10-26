#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python

import tflearn
import numpy
import random


id1=98
id2=109
id3=108
lines = tuple(open("data/dict.txt"))

def predict(model) :
    global id1,id2,id3,lines
    rc =  model.predict([[id1,id2,id3]])
    rc = numpy.argmax(rc)

    id1=id2
    id2=id3
    id3=rc
    return lines[rc]


classes = len(lines)
data,labels = tflearn.data_utils.load_csv("data/alllyrics3.txt",target_column=0,categorical_labels=True,n_classes=classes)

print("start learn")
net = tflearn.input_data(shape=[None,3])
net = tflearn.fully_connected(net, 43)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 32)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)


model = tflearn.DNN(net);
for i in range(1000) :
#model.load("model\lyrick.tyl")
    random.shuffle(data)
    data1 = data[:2500]
    model.fit(data1,labels,n_epoch=5,show_metric=True)

    id1=98
    id2=109
    id3=108

    line = lines[id1]+" "
    line = line  + lines[id2]+" "
    line = line  + lines[id3]+" "

    line = line + predict(model)+" "
    line = line + predict(model)+" "
    line = line + predict(model)+" "
    line = line + predict(model)+" "
    line = line + predict(model)+" "
    line = line + predict(model)+" "
    print (line)

