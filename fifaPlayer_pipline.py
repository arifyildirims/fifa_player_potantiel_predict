'''
This dataset is ideal for data analysis, predictive modeling, and machine learning projects. It can be used for:

Player performance analysis and comparison.
Market value assessment and wage prediction.
Team composition and strategy planning.
Machine learning models to predict future player potential and career trajectories.

#######################################################################################

Bu veri kümesi veri analizi, tahmine dayalı modelleme ve makine öğrenimi projeleri için idealdir. Şu amaçlarla kullanılabilir:

Oyuncu performans analizi ve karşılaştırması.
Piyasa değeri değerlendirmesi ve ücret tahmini.
Takım kompozisyonu ve strateji planlaması.
Gelecekteki oyuncu potansiyelini ve kariyer gidişatını tahmin etmek için makine öğrenimi modelleri.


# ANALİZ : body_type değişkeninde manasız değişkenler var sayısı bir hali az 7 adet onları normal kategorsine geritmeliyim.
# ANALİZ : Bazı oyuncuların milli takımı yok en çok ingiltereye ait futbolcu var listede milli takımda ki oyuncuların büyük bir kısmı yedek pozisyonunda
# ANALİZ : Bazı milli takımları olan oyuncuların milli takımları yanlış al hilal diye bir milli takım yok oyuncunun oynadığı takımı milli takıma atamışlar
# ANALİZ : çok fazla num kolon oldugu için descirebe baksak daha anlamlı olur
# ANALİZ : görselleştirme ile çok bariz bir çarpıklık olmadığını gördüm
# national_team içerisinde ülke olmayan değerler var onların bir tanesini gözlemlemek ve doğrulamak amaçlı koşullu seçim işlemi yapıldı

# ANALİZ : national_team değişkeninde ülkeler olması gerekirken saçma sapan takımlar da var bu değişken bozuk olduğu için bu değişkeni drop edebiliriz
# ANALİZ : potansiyeli groupby'a alırsak interceptions (müdahale) durumu pontansiyel arttıkça artıyor ama potansiyeli en yüksek oyuncuların müdahalesi azalmış sebebi de
#          en yüksek potansiyelli oyuncuların hucüm oyuncuları olması olabilir
'''
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import warnings
import joblib
import joblib
import tensorflow as tf
from datetime import datetime as date
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.tree import export_graphviz, export_text, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from xgboost import XGBRFRegressor


pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

##############################################################
# 1. Load Data
##############################################################
def load_data(dataframe_url):
    df = pd.read_csv(dataframe_url)
    return df
# df = load_data('projeler/database/1_fifa_player/fifa_players.csv')

##############################################################
# 2. Exploratory Data Analysis
##############################################################
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
    print("##################### Describe #####################")
    print(dataframe.describe().T)
# check_df(df)


def general_analysis_func(dataframe):
    degisken_isimleri_tr = ['isim', 'tam_isim', 'dogum_tarihi', 'yas', 'boy_cm', 'kilo_kg', 'pozisyon', 'milliyet',
                            'genel_rating', 'potansiyel', \
                            'deger_euro', 'ucret_euro', 'tercih_edilen_ayak', 'uluslararası_itibar', 'zayif_ayak',
                            'beceri_hareketleri', 'vucut_tipi', \
                            'serbers_kalma_bedeli_euro', 'milli_takim', 'milli_takim_rating', 'milli_takim_pozisyon',
                            'milli_forma_numarası', 'gecis_pasi', 'bitiricilik', \
                            'kafa_vurusu', 'kisa_pas', 'volley', 'top_sürme', 'falso', 'freekick', 'uzun_pas',
                            'top_kontrolu', 'hizlanma', 'surat', 'ceviklik', \
                            'reaksiyon', 'denge', 'sut_gucu', 'zıplama', 'dayaniklilik', 'kuvvet', 'uzun_sut',
                            'saldirganlik', 'mudahaleler', 'pozisyon_alma', 'gorus', \
                            'penalti', 'sogukkanlilik', 'markaj', 'ayakta_mucadele', 'sürüsle_mudahale']
    dataframe.columns = degisken_isimleri_tr

    # %95.23 boş olan ve bir anlam ifade etmeyen degiskenleri silelim
    def missing_values_table(dataframe):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
        nan_columns = missing_df[missing_df['ratio'] > 90].index
        return nan_columns
    nan_columns = missing_values_table(dataframe)
    dataframe.drop(nan_columns, axis=1, inplace=True)

    dataframe['dogum_tarihi'] = pd.to_datetime(dataframe['dogum_tarihi'])
    dataframe['boy_cm'] = dataframe['boy_cm'].apply(lambda x: '%.2f' % x).astype(float)
    dataframe['kilo_kg'] = dataframe['kilo_kg'].apply(lambda x: '%.1f' % x).astype(float)
    dataframe['new_pozisyon_one'] = [unit.rsplit(',')[0] for unit in dataframe['pozisyon'].values]
    dataframe['new_pozisyon_two'] = [unit.rsplit(',')[1] if len(unit.rsplit(',')) > 1 else unit.rsplit(',')[0] \
                                     for unit in dataframe['pozisyon'].values]
    dataframe.drop('pozisyon', axis=1, inplace=True)

    #dataframe['deger_euro'] = dataframe['deger_euro'].apply(lambda x: '%.0f' % x)
    #dataframe['ucret_euro'] = dataframe['ucret_euro'].apply(lambda x: '%.0f' % x)

    # body_type
    body_types = list(dataframe['vucut_tipi'].value_counts().reset_index()['vucut_tipi'][:3])
    dataframe['vucut_tipi'] = ['Normal' if unit not in body_types else unit for unit in dataframe['vucut_tipi']]
    # release_clause_euro
    #dataframe['serbers_kalma_bedeli_euro'] = dataframe['serbers_kalma_bedeli_euro'].apply(lambda x : '%.0f' % x)
# general_analysis_func(df)


##############################################################
# 3.1 Exploratory Data Analysis
##############################################################
def grab_col_names(dataframe, cat_th=1, car_th=30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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
    num_cols = [unit for unit in num_cols if unit != 'dogum_tarihi']
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
# cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
# for cat_col in cat_cols:
#     cat_summary(df, cat_col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
# num_summary(df, num_cols)



def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: ["mean", 'count']}), end="\n\n\n")
# for num_col in num_cols[:5]:
#     target_summary_with_num(df, 'potansiyel', num_col)


def age_and_overall_rating_by_potential(dataframe, target):
    print(dataframe.groupby(target).agg({'genel_rating': ["mean", 'count'],
                                  'yas': ['mean', 'count']}))
# age_and_overall_rating_by_potential(df, 'potansiyel')


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
# for cat_col in cat_cols:
#     target_summary_with_cat(df, 'potansiyel', cat_col)



def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    temp_df[na_columns + '_NA_FLAG'] = np.where(temp_df[na_columns].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
# for na_column in na_columns:
#     missing_vs_target(df, 'potansiyel', na_columns=na_column)



def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
# missing_values_table(df)



def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w',
                      cmap='RdBu')
    plt.show(block=True)
# correlation_matrix(df, df[num_cols].corr().loc['potansiyel', :].reset_index()['index'][1:8])



def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}).sort_values(by='TARGET_MEAN',
                                                                                               ascending=False),
              end="\n\n\n")
# rare_analyser(df, 'potansiyel', cat_cols)


##############################################################
# 4. Data Preprocessing And Feature Engineering
##############################################################
def type_arrangement(dataframe, cat_but_car ):
    cat_but_car.append('dogum_tarihi')
    df_copy = dataframe.copy()
    df_copy_cat_but_car = df_copy.loc[:, cat_but_car]
    df_copy.drop(cat_but_car, axis=1, inplace=True)
    print(f'df shape: {dataframe.shape}\tdf_copy shape: {df_copy.shape}\tdf_copy_cat_but_car shape: {df_copy_cat_but_car.shape}')
    return df_copy
# df_copy = type_arrangement()

#######################
# 4.1 Encoder
#######################
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
# df_copy = one_hot_encoder(df_copy, cat_cols)



def scaler(dataframe, num_cols, target):
    scale = MinMaxScaler()
    scale_feats = scale.fit_transform(dataframe[num_cols])
    dff = pd.DataFrame(scale_feats, columns=num_cols)

    X = dff.drop([target], axis=1)
    y = dff[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    return X, y, X_train, X_test, y_train, y_test, dff
# X, y, X_train, X_test, y_train, y_test, df_copy = scaler(df_copy, df_copy.columns,'potansiyel')


#######################
# 4.2 Missing Values
#######################

def filling_missing_value_with_knn(dataframe, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    filled_data = imputer.fit_transform(dataframe)
    df_copy = pd.DataFrame(columns=dataframe.columns, data=filled_data)
    return df_copy
# df_copy = filling_missing_value_with_knn(df_copy)


#####################
# 4.3 Outliers
#####################

def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# for col_name in df_copy.columns:
#     outlier_thresholds(df_copy, col_name)


def check_outlier(dataframe, col_name, q1=0.1, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
# check_outlier(df_copy, df_copy.columns)



def outlier_list(dataframe, num_cols, plot=False):
    null_list = []
    outliers_for_list = []
    for num_col in num_cols:
        outlier_true = check_outlier(dataframe, num_col)
        null_list.append(outlier_true)

    new_outlier_list = list(zip(num_cols, null_list))

    for a, b in new_outlier_list:
        if b:
            outliers_for_list.append(a)
            if plot:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=dataframe[a], data=dataframe[a])
                plt.title(a.upper() + ' Grafigi')
                plt.show()
    print(new_outlier_list)
    print(outliers_for_list)
    return outliers_for_list

# outliers_for_list = outlier_list(df_copy, df_copy.columns)
# outliers_for_list = outlier_list(df_copy, [col_name for col_name in outliers_for_list if len(df_copy[col_name].value_counts()) > 2])
# outlier_thresholds(df_copy, outliers_for_list)



def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
# for variable in outliers_for_list:
#     replace_with_thresholds(df_copy, variable)



'''def float_conversion(df_copy_2):
    for col_name in df_copy_2.columns:
        if len(df_copy_2[col_name].value_counts()) <= 5:
            df_copy_2[col_name] = df_copy_2[col_name].apply(lambda x: '%.0f' % x)

        elif col_name in ['boy_cm', 'kilo_kg']:
            df_copy_2[col_name] = df_copy_2[col_name].apply(lambda x: '%.2f' % x)

        else:
            df_copy_2[col_name] = df_copy_2[col_name].apply(lambda x: '%.0f' % x)

    df_copy = df_copy_2.copy()
    return df_copy
df_copy = float_conversion(df_copy_2)
'''

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        #print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe)}).sort_values(by='RATIO',
                                                                                               ascending=False),
              end="\n\n\n")
# rare_analyser(df_copy, 'potansiyel', df_copy.columns)


##############################################################
# 5. Based Model
##############################################################
def based_model(X_train, X_test, y_train, y_test, scoring='neg_mean_squared_error', scoring2='r2'):
    best_model = {}
    regressors = [('LR_model', LinearRegression()),
                  ('KNN_model', KNeighborsRegressor()),
                  ("CART_model", DecisionTreeRegressor()),
                  ("RF_model", RandomForestRegressor()),
                  ('Adaboost_model', AdaBoostRegressor()),
                  ('GBM_model', GradientBoostingRegressor()),
                  ('XGBoost_model', XGBRFRegressor())]

    for regressors_model_name, regressor_model in regressors:
        regressors_model_name = regressor_model
        regressors_model_name.fit(X_train, y_train)
        cv_mse = np.mean(-cross_val_score(regressor_model, X_test, y_test, cv=5, scoring='neg_mean_squared_error'))
        cv_r2 = np.mean(cross_val_score(regressor_model, X_test, y_test, cv=5, scoring='r2'))
        print(f"{scoring}: {round(cv_mse, 13)} ({regressors_model_name})\n{scoring2}: {round(cv_r2, 13)} ({regressors_model_name})")
        best_model[regressors_model_name] = cv_mse

    print(best_model)
# X, y, X_train, X_test, y_train, y_test, df_copy = scaler(df_copy, df_copy.columns,'potansiyel')
# based_model(X_train, X_test, y_train, y_test)


########################################################
# 5. Automated Hyperparameter Optimization
########################################################


'''cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

gbm_params = {'n_estimators': [100, 200, 500],
              'learning_rate': [0.1, 0.01],
              'max_depth': [3, 4, 5],
              'min_samples_split': [2, 5]}



regressors = [("CART_model", DecisionTreeRegressor(), cart_params),
              ("RF_model", RandomForestRegressor(), rf_params),
              ('GBM_model', GradientBoostingRegressor(), gbm_params)]
'''
def hyperparameter_optimization(X_train, X_test, y_train, y_test, cv=5, scoring="neg_mean_squared_error", scoring2='r2'):
    cart_params = {'max_depth': range(1, 20),
                   "min_samples_split": range(2, 30)}

    '''rf_params = {"max_depth": [8, 15, None],
                 "max_features": [5, 7, "auto"],
                 "min_samples_split": [15, 20],
                 "n_estimators": [200, 300]}'''
    gbm_params = {'n_estimators': [100, 200, 500],
                  'learning_rate': [0.1, 0.01],
                  'max_depth': [3, 4, 5],
                  'min_samples_split': [2, 5]}

    regressors = [("CART_model", DecisionTreeRegressor(), cart_params),
                  ('GBM_model', GradientBoostingRegressor(), gbm_params)]

    # başlangıc
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        name = regressor.fit(X_train, y_train)
        cv_mse = np.mean(-cross_val_score(name, X_test, y_test, cv=5, scoring=scoring))
        print(f"{scoring} (Before): {round(cv_mse, 13)}")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X_train, y_train)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_mse = np.mean(-cross_val_score(final_model, X_test, y_test, cv=5, scoring=scoring))

        print(f"{scoring} (After): {round(cv_mse, 13)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

# best_models = hyperparameter_optimization(X_train, X_test, y_train, y_test)


##################################################################
# 6. Hyperparameter Optimization Of The Best Model
##################################################################

def best_model_hyperparam(X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
    gbm_paramss = {'n_estimators': [700, 1000, 1500],
                  'learning_rate': [0.1],
                  'max_depth': [5, 7],
                  'min_samples_split': [7, 9, 12, 15]}


    model_is_final_gbm = GradientBoostingRegressor()
    model_is_final_gbm.fit(X_train, y_train)
    # gridSearch
    gs_gbm_final = GridSearchCV(model_is_final_gbm, gbm_paramss, cv=cv, n_jobs=-1, verbose=False).fit(X_train, y_train)

    final_model = GradientBoostingRegressor().set_params(**gs_gbm_final.best_params_)
    mse = np.mean(-cross_val_score(final_model, X_test, y_test, cv=5, scoring=scoring))
    print(f'final model mean_squared_error: {mse}')
    return final_model

# final_model = best_model_hyperparam(X_train, y_train)
# 0.0008095220585930425
# {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 9, 'n_estimators': 1000}


'''
    tahmin değerleri ve gerçek test değerleri arasında ki grafik 
y_pred = final_model.predict(X_test)
# Gerçek ve tahmin değerlerini karşılaştıran bir grafik çizelim
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Gerçek Değerler', color='blue')
plt.plot(y_pred, label='Tahmin Değerler', color='red', linestyle='--')
plt.legend()
plt.title('Gerçek vs Tahmin Değerleri')
plt.xlabel('Örnekler')
plt.ylabel('Değerler')
plt.show()'''
##################################################################
# 7. Model testing with Tensorflow Keras
##################################################################
def model_with_keras(X_train, y_train, X_test, y_test, activation='relu', optimizer='adam', loss='mean_squared_error', epochs=100, validation_split= 0.2):

    # Modeli oluşturun
    model_keras = Sequential([
        Dense(64, activation=activation, input_shape=(X_train.shape[1],)),  # Giriş katmanı ve ilk gizli katman
        Dense(64, activation=activation),  # İkinci gizli katman
        Dense(1)  # Çıkış katmanı (regresyon için tek bir çıkış)
    ])

    # Modeli derleyin
    model_keras.compile(optimizer=optimizer, loss=loss)

    # Modeli eğitin
    history = model_keras.fit(X_train, y_train, epochs=epochs, validation_split=validation_split, verbose=1)

    # Modeli test verisi üzerinde değerlendirin
    test_loss = model_keras.evaluate(X_test, y_test)
    print('Test kaybı (MSE):', test_loss)

    # Tahminler yapın
    # predictions = model_keras.predict(X_test)

    # İlk birkaç tahmini ve gerçek değeri gösterin
    # for i in range(5):
    #     print(f"Tahmin edilen değer: {predictions[i][0]}, Gerçek değer: {y_test[i]}")
    return model_keras


##############################################################
# 8. Pipeline Main Function
##############################################################

def main():
    df = load_data('projeler/database/1_fifa_player/fifa_players.csv')
    general_analysis_func(df)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    #
    df_copy = type_arrangement(df, cat_but_car)
    df_copy = one_hot_encoder(df_copy, cat_cols)
    X, y, X_train, X_test, y_train, y_test, df_copy = scaler(df_copy, df_copy.columns, 'potansiyel')
    df_copy = filling_missing_value_with_knn(df_copy)

    # outlier
    for col_name in df_copy.columns:
        outlier_thresholds(df_copy, col_name)

    outliers_for_list = outlier_list(df_copy, df_copy.columns)
    outliers_for_list = outlier_list(df_copy, [col_name for col_name in outliers_for_list if
                                               len(df_copy[col_name].value_counts()) > 2])

    outlier_thresholds(df_copy, outliers_for_list)
    # outlierı baskılamak
    for variable in outliers_for_list:
        replace_with_thresholds(df_copy, variable)

    # model
    X, y, X_train, X_test, y_train, y_test, df_copy = scaler(df_copy, df_copy.columns, 'potansiyel')
    based_model(X_train, X_test, y_train, y_test)
    best_models = hyperparameter_optimization(X_train, X_test, y_train, y_test)

    #
    final_model = best_model_hyperparam(X_train, y_train, scoring='neg_mean_squared_error')
    joblib.dump(final_model, "final_model.pkl")
    return final_model


if __name__ == "__main__":
    print("İşlem başladı")
    main()































































































































































































































































































































































































