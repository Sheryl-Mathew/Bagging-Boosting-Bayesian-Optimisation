import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

f = open('output.txt', 'a+')

def read_files():
    train_data = pd.read_csv("income.train.txt",header=None)
    dev_data = pd.read_csv("income.dev.txt",header=None)
    test_data = pd.read_csv("income.test.txt",header=None)
    train_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    dev_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    test_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    return train_data,dev_data,test_data

def preprocessing_sklearn(training_data,development_data,testing_data):
    train_label = training_data.iloc[:,-1]
    dev_label = development_data.iloc[:,-1]
    test_label = testing_data.iloc[:,-1]

    train_data = training_data.drop('income',1)
    dev_data = development_data.drop('income',1)
    test_data = testing_data.drop('income',1)

    combined_dataset = pd.concat([train_data,dev_data,test_data],keys=['train','dev','test'])
    one_hot_encoded_data  = pd.get_dummies(combined_dataset)
    train_data, dev_data, test_data = one_hot_encoded_data.xs('train'), one_hot_encoded_data.xs('dev'), one_hot_encoded_data.xs('test')
    
    return train_data,train_label,dev_data,dev_label,test_data,test_label

def parameters():
    parameters = {
        'size': (10,100),
        'depth': (1,10)
    }
    return parameters

def get_training_data():
    train_data,dev_data,test_data = read_files()
    train_data,train_label,_,_,_,_ = preprocessing_sklearn(train_data,dev_data,test_data)
    return train_data,train_label

def cross_validation(depth,size):
    train_data,train_label = get_training_data()
    mean_validation_score = cross_val_score(
        AdaBoostClassifier(DecisionTreeClassifier(max_depth = int(depth),random_state=1),n_estimators = int(size)).fit(train_data, train_label),
        train_data,train_label, cv=2).mean()
    return mean_validation_score

def boosting(train_data,train_label,test_data,test_label,depth_of_tree,size_of_ensemble):
    decision_tree = DecisionTreeClassifier(max_depth = int(round(depth_of_tree)),random_state=1)
    model = AdaBoostClassifier(decision_tree,n_estimators = int(round(size_of_ensemble))).fit(train_data, train_label)
    predicted = model.predict(test_data)
    accuracy = accuracy_score(test_label,predicted)*100
    return accuracy

def bayes_optimisation(number_of_iterations):
    parameters_to_search = parameters()
    bayesopt = BayesianOptimization(cross_validation,parameters_to_search)
    bayesopt.maximize(n_iter=number_of_iterations)
    best_parameters = bayesopt.res['max']['max_params']
    return best_parameters

def accuracies(train_data,train_label,test_data,test_label):
    all_hyper_parameters = []
    accuracies = []
    iteration_range = range(10)
    iterations = []
    for iteration_number in iteration_range:
        hyper_parameters = bayes_optimisation(iteration_number+1)
        accuracy = boosting(train_data,train_label,test_data,test_label,hyper_parameters['depth'],hyper_parameters['size'])
        accuracies.append(accuracy)
        all_hyper_parameters.append(hyper_parameters)
        iterations.append(iteration_number+1)
    return accuracies,all_hyper_parameters,iterations

def print_accuracies(accuracies,hyper_parameters):
    print("Accuracies: Boosting", file =f)
    accuracies = pd.DataFrame(accuracies)
    hyp = pd.DataFrame(hyper_parameters).T
    depth_of_tree = hyp.iloc[0]
    size_of_ensemble = hyp.iloc[1]
    depth_of_tree = pd.DataFrame(depth_of_tree)
    size_of_ensemble = pd.DataFrame(size_of_ensemble)
    concat_data = pd.concat([depth_of_tree,size_of_ensemble,accuracies],axis=1)
    df = pd.DataFrame(concat_data)
    df.columns = ["Depth of Tree","Ensemble Size","Validation Accuracy"]
    print(df, file =f)

def plot_curve(accuracies,hyper_parameters,iterations):
    plt.plot(iterations, accuracies, color = 'red', marker='o', linestyle='solid')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs Number of Iterations: Boosting") 

    for label, x, y in zip(hyper_parameters, iterations, accuracies):
        plt.text(x,y,label,fontsize=8,horizontalalignment='center',verticalalignment='center')

    save_file_name = "Accuracy_vs_Number_of_Iterations_Boosting.png" 
    plt.savefig(save_file_name)
    #plt.show()

def main():
    train_data,dev_data,test_data = read_files()
    train_data_sk,train_label_sk,dev_data_sk,dev_label_sk,_,_ = preprocessing_sklearn(train_data,dev_data,test_data)
    accuracy,hyper_parameters,iterations = accuracies(train_data_sk,train_label_sk,dev_data_sk,dev_label_sk)
    print(file =f)
    print("Boosting Bayesian Optimisation:", file =f)
    print(file =f)
    print_accuracies(accuracy,hyper_parameters)
    plot_curve(accuracy,hyper_parameters,iterations)

main()

f.close()