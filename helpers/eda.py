import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_csv(string):
    return pd.read_csv(string)

# function summarizing all columns of a single dataFrame
# takes df type as parameter
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# function that summarizes each of multiple dataframes
def check_df_names(df_name, head=5):
    print("\nDataFrame Name: ", df_name)
    print("##################### Shape #####################")
    print(eval(df_name).shape)
    print("##################### Types #####################")
    print(eval(df_name).dtypes)
    print("##################### Head #####################")
    print(eval(df_name).head(head))
    print("##################### Tail #####################")
    print(eval(df_name).tail(head))
    print("##################### NA #####################")
    print(eval(df_name).isnull().sum())


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T, end="\n\n")

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal de??i??kenlerin isimlerini verir.
    Not: Kategorik de??i??kenlerin i??erisine numerik g??r??n??ml?? kategorik de??i??kenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                De??i??ken isimleri al??nmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan de??i??kenler i??in s??n??f e??ik de??eri
        car_th: int, optinal
                kategorik fakat kardinal de??i??kenler i??in s??n??f e??ik de??eri

    Returns
    ------
        cat_cols: list
                Kategorik de??i??ken listesi
        num_cols: list
                Numerik de??i??ken listesi
        cat_but_car: list
                Kategorik g??r??n??ml?? kardinal de??i??ken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam de??i??ken say??s??
        num_but_cat cat_cols'un i??erisinde.
        Return olan 3 liste toplam?? toplam de??i??ken say??s??na e??ittir: cat_cols + num_cols + cat_but_car = de??i??ken say??s??

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def target_correlation_matrix(dataframe, corr_th=0.5, target="Salary"):
    """
    Ba????ml?? de??i??ken ile verilen threshold de??erinin ??zerindeki korelasyona sahip de??i??kenleri getirir.
    :param dataframe:
    :param corr_th: e??ik de??eri
    :param target:  ba????ml?? de??i??ken ismi
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Y??ksek threshold de??eri, corr_th de??erinizi d??????r??n!")