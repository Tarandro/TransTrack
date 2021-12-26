import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def length_data(data):
    print("La base de données contient {} lignes".format(len(data)))


def info_data(data):
    print('Data information :')
    print('*' * 50)
    print(data.info(verbose=True))
    print('*' * 75)


def seperate_col_object_numeric(data):
    """ separation données numériques / données object ("object", "bool", "category", "datetime64", "timedelta")
    Args :
        data (DataFrame)
    """
    col_objects = []
    col_numerics = []

    for col in data.columns:
        type_col = data[col].dtypes

        if 'float' in str(type_col) or ('int' in str(type_col) and len(np.unique(data[col])) > 2):
            col_numerics.append(col)
        else:
            # cols = ["object", "bool", "category", "datetime64", "timedelta"]
            col_objects.append(col)

    return col_objects, col_numerics


def eda_value_counts(data, list_name_columns, n=3, pourcentage=True, alphabetic_ascending=True, bins=4):
    """ Application de la fonction value_counts() (librairie pandas) selon le type de données et affiche par groupe de n les résultats
    Args :
        data (DataFrame)
        list_name_columns (list)
        n (int) : numbre de table value_counts() par ligne
        pourcentage (Boolean) : fréquence en pourcentage
        alphabetic_ascending (Boolean)
        bins (int) : nombre de groupes à créer pour données numériques
    """

    def fct_value_counts(data, col, bins, pourcentage, alphabetic_ascending):
        """ Readjustement of value_counts() function from pandas library """

        # use option 'bins' for numeric features :
        if bins is not None:
            # trick to get better 'bins' :
            assert bins > 2, "Need bins > 2"
            out_value_counts = data[col].value_counts(bins=bins)
            # get value bins :
            new_bins = [list(out_value_counts.index)[0].left] + [t.right for t in list(out_value_counts.index)]
            list_values = list(data[col])
            if list(out_value_counts.index)[0].left < 0:
                new_bins.append(-0.0001)

            # create a range [-0.0001, 0.0001] for values '0' if frequence value '0' > 10%
            if list_values.count(0) >= len(list_values) * 0.10:
                new_bins.append(-0.0001)
                new_bins.append(0.0001)
            # create a range [0.9999, 1.0001] for values '1' if frequence value '1' > 10%
            if list_values.count(1) >= len(list_values) * 0.10:
                new_bins.append(0.9999)
                new_bins.append(1.0001)
            new_bins = list(set(new_bins))
            new_bins.sort()
        else:
            new_bins = None

        info_value_counts = data[col].value_counts(dropna=False, normalize=pourcentage, bins=new_bins).sort_index(
            ascending=alphabetic_ascending)

        if bins:
            # transform range index [-0.0001, 0.0001] -> 0 and [0.9999, 1.0001] -> 1
            new_index = list(info_value_counts.index)
            for i, j in enumerate(list(info_value_counts.index)):
                if j == pd.Interval(-0.0001, 0.0001, closed='right'):
                    new_index[i] = 0
                if j == pd.Interval(0.9999, 1.0001, closed='right'):
                    new_index[i] = 1
            info_value_counts.index = new_index
            # only show index with value_count > 0 :
            info_value_counts = info_value_counts[info_value_counts > 0]

        if pourcentage:
            return str((info_value_counts * 100).round(2).astype(str) + "%").split('\n')[:-1]
        else:
            return str((info_value_counts.astype(str).split('\n')[:-1]))

    # Return value_counts() depending on data type :
    str_value_counts = []
    for col in list_name_columns:
        type_col = data[col].dtypes
        if 'float' in str(type_col) or 'int' in str(type_col):
            try:
                if 'int' in str(type_col) and len(np.unique(data[col])) < 5:
                    str_value_counts.append(fct_value_counts(data, col, None, pourcentage, alphabetic_ascending))
                else:
                    str_value_counts.append(fct_value_counts(data, col, bins, pourcentage, alphabetic_ascending))
            except:
                str_value_counts.append(fct_value_counts(data, col, None, pourcentage, alphabetic_ascending))

        else:
            str_value_counts.append(fct_value_counts(data, col, None, pourcentage, alphabetic_ascending))

    # get the length of the largest value_counts() print
    max_length = np.max(
        [len(char) for char in list_name_columns + [item for sublist in str_value_counts for item in sublist]])

    gap = 10
    # print by group of n:
    for i in range(0, len(str_value_counts), n):
        subset_str_value_counts = []
        message = ""
        message_col = ""
        for j in range(np.max([len(t) for t in str_value_counts[i:i + n]])):
            for h in range(n):
                try:
                    text = str_value_counts[i + h][j]
                except:
                    text = ""
                message += text + " " * (max_length - len(text) + gap)
                if h == n - 1:
                    message += '\n'
                if j == 0:
                    try:
                        message_col += list_name_columns[i + h] + " :" + " " * (
                                    len(text) - len(list_name_columns[i + h] + " :") + max_length - len(text) + gap)
                    except:
                        message_col += " " * (max_length - len(text) + gap)
        message = message_col + "\n" + message

        message += '\n'
        print(message)
        print("-" * (n * max_length + n * gap))


def eda_value_counts_plot(data, name_col, pie_or_bar="bar", pourcentage=True, dropna=False, n_head=10, width=8,
                          height=8):
    """ Analyse univariée pour données object : plot de la fonction value_counts()
    Args :
        data (DataFrame)
        name_col (str) nom d'une colonne de data
        pie_or_bar (str) "pie" or "bar"
        pourcentage (Boolean) : fréquence en pourcentage
        dropna (Boolean) supprimer les NA dans le comptage
        n_head (int) limite d'une nombre de modalités à afficher
    """
    if name_col not in data.columns:
        return
    plt.figure(figsize=(width, height))
    if pie_or_bar == "bar":
        data[name_col].value_counts(dropna=dropna, normalize=pourcentage).head(n_head).plot.bar(cmap='Set3')
        plt.xlabel(name_col)
    else:
        data[name_col].value_counts(dropna=dropna, normalize=pourcentage).head(n_head).plot.pie(cmap='Set3',
                                                                                                autopct='%1.1f%%')
    if pourcentage:
        plt.ylabel("Frequency (%)")
    else:
        plt.ylabel("Frequency")
    plt.show()


def eda_hist_plot(data, name_col, width=8, height=8):
    """ Analyse univariée pour données numérique : plot de la fonction hist()
    Args :
        data (DataFrame)
        name_col (str) nom d'une colonne de data
    """
    if name_col not in data.columns:
        return
    plt.figure(figsize=(width, height))
    data[name_col].plot.hist(cmap='Set3')
    plt.xlabel(name_col)
    plt.show()


def univariate_plot(data, name_col, pie_or_bar="bar", pourcentage=True, dropna=False, n_head=10, width=8, height=8):
    """ Graph univarié selon le type de la donnée """

    if name_col not in data.columns:
        return

    type_col = data[name_col].dtypes

    if 'float' in str(type_col) or ('int' in str(type_col) and len(np.unique(data[name_col])) > 2):
        eda_hist_plot(data, name_col, width, height)
    else:
        # cols = ["object", "bool", "category", "datetime64", "timedelta"]
        eda_value_counts_plot(data, name_col, pie_or_bar, pourcentage, dropna, n_head, width, height)