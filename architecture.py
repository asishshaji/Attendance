from tensorflow.keras.layers import Dense, Activation, LeakyReLU
from tensorflow.keras.models import Sequential


class NN:
    def __init__(self, noOfClasses):
        self.noOfClasses = noOfClasses
        self.model = Sequential()

    def architecture(self):
        self.model.add(Dense(64, input_dim=128))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dense(32))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dense(16))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(Dense(self.noOfClasses))
        self.model.add(Activation('softmax'))

        return self.model
