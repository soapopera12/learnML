# basic utilities 

def save_model(model, modelName):
    '''
    :type model: class model
    :type modelName: string
    :rtype: none
    '''
    saveAt = "artifacts/"
    saveat += modelName
    saveat += ".pkl"
    with open(saveAt, "wb") as f:
        pikle.dump(model, f)

def hist_plot(df, title):
    '''
    :type df: pandas dataframe
    :type title: string
    :rtype: none
    '''
    plt.figure(figsize=(10, 4))
    sns.displot(df)
    plt.title(title)
    sns.despine()
    plt.show()

def scatter_plot(df1, df2, title, xlabel, ylabel):
    '''
    :type df1: pandas dataframe
    :type df2: pandas dataframe
    :type title: string
    :type xlabel: xlabel
    :type ylabel: ylabel
    :rtype: none
    '''
    fig, ax = plt.subplot(figsize=(5, 3))
    ax.scatter(df1, df2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    plt.show()
   
def get_result(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm[np.isnan(cm)] = 0
    tp, fp, fn, tp = cm.flatten()
    tp, fp, fn, tp = tp.item(), fp.item(), fn.item(), tp.item()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * recall / precision + recall)
    fpr = fp / (fp + tn)
    return accuracy, precision, recall, f1_score

