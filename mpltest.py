import matplotlib
from matplotlib import pyplot as plt

epochs = [1,2,3,4,5]
train_acc = [0.4,0.5,0.6,0.65,0.7]
val_acc = [0.35,0.45,0.55,0.6,0.65]

plt.plot(epochs, train_acc, label='train acc')
plt.plot(epochs, val_acc, label='validation acc') 
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('train v validation acc')
plt.legend()
plt.show()