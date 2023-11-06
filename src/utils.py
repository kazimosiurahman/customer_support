from nltk import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_tokens(strings):
    return [word_tokenize(string)
            for string in strings]


def get_tfidf_vectors(df,
                      vector_transformer,
                      type='train'):
    if type == 'train':
        return vector_transformer.fit_transform(df)
    elif type == 'test':
        return vector_transformer.transform(df)


def plot_model_performance(cv_df):
    cv = len(cv_df['fold_idx'].unique())
    plt.figure(figsize=(8,5))
    sns.boxplot(x='model_name', y='accuracy',
                data=cv_df[cv_df.accuracy > 0],
                color='lightblue',
                showmeans=True)
    plt.title(f"MEAN ACCURACY (cv = {cv})n", size=14);


def plot_confusion_matrix(y_test, y_pred, model, df):
    conf_mat = confusion_matrix(y_test, y_pred)
    _, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
                xticklabels=df.REASON.unique(),
                yticklabels=df.REASON.unique(),
                ax=ax
                )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f"CONFUSION MATRIX - {model.__class__.__name__}", size=16);
