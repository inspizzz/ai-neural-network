import numpy as np
import gzip
import matplotlib.pyplot as plt

# np.seterr(all='ignore')


class Graph:
    def __init__(self):
        plt.ion()
        self.accuracy = 1000
        self.counter = 0
        self.points = [2]*self.accuracy
        self.graph, = plt.plot(np.linspace(0, 1, self.accuracy))

    def add(self, y):
        self.points[self.counter] = y
        self.graph.set_ydata(self.points)
        plt.draw()
        plt.pause(0.00001)
        self.counter += 1
        if self.counter >= self.accuracy:
            self.counter = 0


class Dataset:
    def __init__(self):
        self.data = []
        self.classes = []
        self.size = 28

    def load(self, dir, amount):
        image_size = 28
        label_size = 1

        images = gzip.open('PATH/TO/train.gz', 'r')
        labels = gzip.open('PATH/TO/labels.gz', 'r')

        images.read(16)
        labels.read(8)

        image_data = images.read(amount * image_size * image_size)
        image_data = np.frombuffer(image_data, dtype=np.uint8).astype(float)
        image_data = image_data.reshape((amount, image_size, image_size))

        label_data = labels.read(amount)
        label_data = np.frombuffer(label_data, dtype=np.uint8).astype(float)
        label_data = label_data.reshape(amount)

        labels = []
        for i in range(0, len(label_data)):
            value = int(label_data[i])
            thing = [[0]] * 10
            thing[value] = [1]
            labels.append(thing)

        self.data = image_data
        self.classes = labels

    def Size(self):
        return len(self.data)

    def Get_random(self):
        number = np.random.randint(0, len(self.data))
        return self.data[number], self.classes[number]

    def Randomise(self):
        np.random.shuffle(self.data)

    def Get_dataset(self):
        return np.asarray(self.data), np.asarray(self.classes)


class NeuralNet2:
    def __init__(self, arch, training, labeled, weight_learning_rate=0.05, bias_learning_rate=0.05, display_update=1000,
                 epochs=1):
        self.layer_names = []
        self.layer_sizes = []

        for value, key in arch.items():
            self.layer_names.append(value)
            self.layer_sizes.append(key)

        self.weights = []
        self.biases = []
        for i in range(0, len(self.layer_sizes) - 1):
            self.weights.append(
                np.matrix(data=np.random.uniform(-1, 1, (self.layer_sizes[i + 1], self.layer_sizes[i])), dtype=float))
            self.biases.append(np.matrix(data=np.ones((self.layer_sizes[i + 1], 1))))

        self.drop_layer = []
        for i in self.weights:
            # create a mask for the weight matrix
            y, x = i.shape
            m = np.matrix(data=[[(1 if k >= 0.1 else 0) for k in np.random.random(x)] for j in np.random.random(y)], dtype=int)
            self.drop_layer.append(m)

        self.image = training
        self.label = labeled

        self.cache = []
        self.activations = []

        self.alpha = weight_learning_rate
        self.beta = bias_learning_rate
        self.update = display_update
        self.epochs = epochs

        self.graph = Graph()

    def __repr__(self):
        debug = ""
        debug += f"\nNeuralNetwork:\n"
        debug += f" - Architecture:"
        debug += f"  - {'-'.join(str(l) for l in self.layer_sizes)}\n"
        debug += f"  - {'-'.join(str(l) for l in self.layer_names)}\n"
        debug += f" - weights:\n"
        debug += f"  - {'-'.join(str(l.shape) for l in self.weights)}\n"
        debug += f" - biases:\n"
        debug += f"  - {'-'.join(str(l.shape) for l in self.biases)}\n"
        debug += f" - dataset:\n"
        debug += f"  - {self.image.shape}\n"
        debug += f"\nData\n"
        debug += f" - outputL\n"
        debug += f"  - {self.cache[-1]}"
        return debug

    def train(self):
        count = 0
        average_epoch = []
        iter_cost = []
        all_costs = []

        for i in range(self.epochs):
            for x, y in zip(self.image, self.label):
                self.forward_propagate(x.reshape((np.multiply(x.shape[0], x.shape[1]), 1)))
                cost = self.cost_function(self.cache[-1], y)
                self.backward_propagate2(y)

                average_epoch.append(cost)
                all_costs.append(cost)
                iter_cost.append(cost)

                if count % 100 == 0:
                    print(f"epoch {i} count {count} average cost is {np.average(iter_cost)}")
                    # for x1, y1 in zip(self.cache[-1], y):
                    #     print(f"got {x1}, expected {y1}")
                    self.graph.add(np.average(iter_cost))
                    iter_cost = []
                count += 1

            if i % self.update == 0:
                print(f"epoch {i} average cost of last {self.update} epochs is {np.average(average_epoch)} also saving...")
                self.save()
                average_epoch = []
                print(self)

    def forward_propagate(self, activations):
        self.cache = [activations]
        self.activations = [activations]
        for i in range(len(self.weights)):
            sum = np.dot(np.multiply(self.weights[i], self.drop_layer[i]), self.cache[-1]) + self.biases[i]
            self.activations.append(sum)
            self.cache.append(self.Sigmoid(sum))

    def backward_propagate2(self, expected):
        deltas = [np.multiply(self.cost_function_deriv(self.cache[-1], expected), self.Sigmoid_derivative(self.activations[-1]))]

        for i in np.arange(len(self.weights) - 1, 0, -1):
            delta = np.dot(self.weights[i].T, deltas[-1])
            delta = np.multiply(delta, self.Sigmoid_derivative(self.activations[i]))
            deltas.append(delta)
        deltas.reverse()
        for i in np.arange(0, len(deltas)):
            self.weights[i] += np.multiply(-self.alpha, np.dot(deltas[i], self.cache[i].T))
            self.biases[i] += np.multiply(-self.beta, deltas[i])

    def Sigmoid(self, x):
        return np.divide(1.0, (1.0 + np.exp(-x)))

    def Sigmoid_derivative(self, x):
        return np.multiply(self.Sigmoid(x), (1.0 - self.Sigmoid(x))) # <---

    def ReLu(self, x):
        return np.maximum(0, x)

    def ReLu_deriv(self, x):
        return np.array(np.array(x) > 0, dtype=int)

    def cost_function(self, predicted, expected):
        m = 10

        cost = -1 / m * np.sum(np.multiply(expected, np.log(predicted)) + np.multiply((1 - expected), np.log(1 - predicted)))
        cost = np.squeeze(cost)
        return cost

    def cost_function_deriv(self, predicted, expected):
        return -1 * (np.divide(expected, predicted) - np.divide(1-expected, 1-predicted))

    def save(self):
        # save the bias matrix
        # save the weight matrix
        # save the layer sizes
        # save the layer names

        np.save("save/weights", np.asarray(self.weights, dtype=object))
        np.save("save/biases", np.asarray(self.biases, dtype=object))
        np.save("save/layer_sizes", np.asarray(self.layer_sizes, dtype=object))
        np.save("save/layer_names", np.asarray(self.layer_names, dtype=object))
        np.save("save/drop_layer", np.asarray(self.drop_layer, dtype=object))

    def load(self):
        print("loading from file")
        # load the bias matrix
        # load the weight matrix
        # load the layer sizes
        # load the layer names

        self.weights = np.load("save/weights.npy", allow_pickle=True)
        self.biases = np.load("save/biases.npy", allow_pickle=True)
        self.layer_sizes = np.load("save/layer_sizes.npy", allow_pickle=True)
        self.layer_names = np.load("save/layer_names.npy", allow_pickle=True)
        self.drop_layer = np.load("save/drop_layer.npy", allow_pickle=True)


if __name__ == "__main__":
    dataset = Dataset()
    dataset.load("PATH/TO/train.gz", 60000)
    images, labels = dataset.Get_dataset()
    architecture = {"input": 784, "hidden1": 400, "hidden2": 300, "hidden3": 200, "hidden4": 100, "output": 10}
    net = NeuralNet2(architecture, weight_learning_rate=0.0005, bias_learning_rate=0.001, epochs=20, training=images,
                    labeled=labels, display_update=1)
    net.load()

    print("training")
    net.train()
    for i in range(10):
        net.forward_propagate(images[i].reshape(np.multiply(images[i].shape[0], images[0].shape[1]), 1))
        pred = net.cache[-1]
        for x, y in zip(pred, labels[i]):
            print(f"got {x} expected {y}")
        print()
