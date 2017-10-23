import tflearn
import numpy

lines = tuple(open("data/dict.txt"))


classes = 6000
data,labels = tflearn.data_utils.load_csv("data/alllyrics3.txt",target_column=0)


emptydt = []
for m in range(classes):
    emptydt.append(0)


def genData(v1,v2,v3):
    dt = emptydt[:]
    dt[v1] = 1
    dt[v2] = 1
    dt[v3] = 1
    return dt


for i in range(len(data)) :
    data[i] = genData(int(data[i][0]),int(data[i][1]),int(data[i][2]))

    lb = emptydt[:]
    lb[int(labels[i])] = 1
    labels[i] = lb
    if i%5000 == 0:
        print(i)



print("start learn")
net = tflearn.input_data(shape=[None,classes])
net = tflearn.fully_connected(net,256,activation="ReLU")
net = tflearn.fully_connected(net,1024,activation="ReLU")
net = tflearn.fully_connected(net,classes,activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net);
model.fit(labels,data,n_epoch=500,show_metric=True)
model.save("model\lyrick.tyl")

id1=98
id2=109
id3=108

rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
id1=id2
id2=id3
id3=rc
rc =  numpy.argmax(model.predict([genData(id1,id2,id3)]))
print(dict[rc])
