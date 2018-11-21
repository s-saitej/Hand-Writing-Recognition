import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

data = datasets.load_digits()

print(data.data)
print(data.target)

print(data.images[0])

print(len(data.data))

model = svm.SVC(C=100, gamma = 0.001)

x = data.data[:-10]
y = data.target[:-10]

model.fit(x,y)

print("Prediction : ",model.predict(data.data[:-2]))

plt.imshow(data.images[-2], cmap = plt.cm.gray_r, interpolation = "nearest")
plt.show()

