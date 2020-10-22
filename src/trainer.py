from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns

class TFTrainer:

    def __init__(self,model_fn,data,label,name,epochs=10,test_size=0.25,batch_size=8):
        self.name = name
        self.data = data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, label, test_size=test_size,
                                                                                shuffle=True)
        self.epochs = epochs
        self.test_size = test_size
        self.batch_size = batch_size
        self.model_fn = model_fn
        self.model = KerasClassifier(build_fn=self.build_fn)
        self.tested = False
        self.fitted = False

    def fit(self):
        self.history = self.model.fit(x=self.X_train,y=self.y_train,validation_data=(self.X_test,self.y_test),epochs=self.epochs,batch_size=self.batch_size)
        self.tested = False
        self.fitted = True

    def test(self):
        self.y_pred = self.model.predict(self.X_test)

        self.tested=True

    def evaluate(self):
        if not self.tested:
            self.test()
        score = accuracy_score(self.y_test, self.y_pred)

        return score

    def plot_confusion_matrix(self):
        if not self.tested:
            self.test()

        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True)
        plt.title(self.name)
        plt.show()

    def build_fn(self):
        return self.model_fn(self.data.shape[1:])

    def plot_history(self):
        plt.figure(figsize=(5,6))
        plt.subplot(211)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title(self.name+ ' model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.grid()

        plt.subplot(212)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title(self.name+' model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'Validation'], loc='upper left')
        plt.grid()
        plt.show()


class SKLTrainer:
    def __init__(self,model,data,label,name,test_size=0.15):
        self.name = name
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, label, test_size = test_size,shuffle=True)
        self.tested = False
        self.fitted = False

    def fit(self):
        self.model.fit(self.X_train,self.y_train)
        self.tested = False
        self.fitted = True

    def test(self):
        self.y_pred = self.model.predict(self.X_test)
        self.tested=True

    def evaluate(self):
        if not self.tested:
            self.test()
        score = accuracy_score(self.y_test, self.y_pred)

        return score

    def plot_confusion_matrix(self):
        if not self.tested:
            self.test()

        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True)
        plt.title(self.name)
        plt.show()



if __name__ == '__main__':
    pass