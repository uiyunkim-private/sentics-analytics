from sklearn.ensemble import RandomForestClassifier
from src.model import tf_dense,tf_lstm
from src.preprocess import load_dataset, window_sliding
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from src.trainer import SKLTrainer,TFTrainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt

def NaiveBayes_session():
    data, label = load_dataset('df')
    data = StandardScaler().fit_transform(data)
    data = Normalizer().fit_transform(data)
    trainer = SKLTrainer(model=CategoricalNB(),
                         data=data,
                         label=label,
                         name="Naive Bayes")
    trainer.fit()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()

    return score

def DecisionTree_session():
    data, label = load_dataset('df')
    data = StandardScaler().fit_transform(data)
    trainer = SKLTrainer(model=DecisionTreeClassifier(),
                         data=data,
                         label=label,
                         name="Decision Tree")
    trainer.fit()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()
    return score

def RandomForest_session():
    data, label = load_dataset('df')
    data = StandardScaler().fit_transform(data)
    trainer = SKLTrainer(model=RandomForestClassifier(max_depth=2),
                         data=data,
                         label=label,
                         name="Random Forest")
    trainer.fit()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()
    return score

def SVC_session():
    data, label = load_dataset('df')
    data = StandardScaler().fit_transform(data)
    trainer = SKLTrainer(model=SVC(gamma='auto'),
                         data=data,
                         label=label,
                         name="SVC")
    trainer.fit()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()
    return score

def Dense_session():
    data, label = load_dataset('np')
    trainer = TFTrainer(model_fn=tf_dense,
                        data=data,
                        label=label,
                        name="Dense",
                        epochs=10,
                        test_size=0.15,
                        batch_size=8)
    trainer.fit()
    trainer.plot_history()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()
    return score

def LSTM_session():
    data, label = load_dataset('np')
    data = window_sliding(data,window_size=20,stride=5)
    trainer = TFTrainer(model_fn=tf_lstm,
                        data=data,
                        label=label,
                        name="LSTM",
                        epochs=10,
                        test_size=0.15,
                        batch_size=8)
    trainer.fit()
    trainer.plot_history()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()
    return score


if __name__ =='__main__':
    test_scores = [LSTM_session(),
                   Dense_session(),
                   RandomForest_session(),
                   SVC_session(),
                   DecisionTree_session(),
                   NaiveBayes_session()]

    print(test_scores)

    plt.bar(["LSTM","Dense","RandomForest","SVC","DecisionTree","NaiveBayes"],test_scores)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Score Comparison")
    plt.show()
