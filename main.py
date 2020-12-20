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
from src.configuration import SAVE_PLOT,ROOT_DIR
from tensorflow.python.client import device_lib

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

def LSTM_session(window_size=2000,stride=10,epochs=10):
    data, label = load_dataset('np',window_size=50,stride=10)
    print(data.shape)
    trainer = TFTrainer(model_fn=tf_lstm,
                        data=data,
                        label=label,
                        name="2.5 sec per sample LSTM",
                        epochs=1000,
                        test_size=0.15,
                        batch_size=256)
    trainer.fit()
    trainer.plot_history()
    score = trainer.evaluate()
    trainer.plot_confusion_matrix()
    return score


if __name__ =='__main__':
    test_scores = [LSTM_session()]
    # test_scores = [Dense_session(),
    #                RandomForest_session(),
    #                SVC_session(),
    #                DecisionTree_session()
    #                ]

    print(test_scores)

    plt.bar(["Dense","RandomForest","SVC","DecisionTree"],test_scores)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Score Comparison")
    if SAVE_PLOT:
        plt.savefig(ROOT_DIR/"documentation"/ "Score_comparison.png",dpi=300)
    plt.show()
