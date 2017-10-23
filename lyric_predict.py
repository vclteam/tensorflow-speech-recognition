import tflearn

lines = tuple(open("data/dict.txt"))

classes = 6000
emptydt = []
for m in range(classes):
    emptydt.append(0)


def genData(v1,v2,v3):
    dt = emptydt[:]
    dt[v1] = 1
    dt[v2] = 1
    dt[v3] = 1
    return dt


net = tflearn.input_data(shape=[None,classes])
net = tflearn.fully_connected(net,64,activation="ReLU")
net = tflearn.fully_connected(net,classes,activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net);
model.load("model\lyrick.tyl")

dt = genData(98,109,108)
rc = model.predict(dt)
print(rc)

