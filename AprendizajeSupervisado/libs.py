#Resto de libs
import os, gc
import psutil
import sidetable
import unidecode
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
from seaborn import heatmap
from random import randrange
from scipy.stats import norm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from IPython.display import display_html
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import (MinMaxScaler,
                                   OneHotEncoder,
                                   LabelBinarizer,
                                   StandardScaler)
from sklearn.tree import (plot_tree,
                          DecisionTreeClassifier)
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import (Perceptron,
                                  SGDClassifier,
                                  LogisticRegression)
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     RandomizedSearchCV)
from sklearn.metrics import (f1_score,
                             roc_curve,
                             recall_score,
                             roc_auc_score,
                             accuracy_score,
                             precision_score,
                             confusion_matrix,
                             classification_report, 
                             precision_recall_curve,
                             balanced_accuracy_score)


#Reproducibilidad
np.random.seed(0)

#Aesthetic
sns.set()
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["legend.shadow"]    = True

BLUE   = '#5DADE2'
RED    = '#ff7043'
ORANGE = '#F5B041'
GREEN  = '#58D68D'
YELLOW = '#F4D03F'

pd.options.display.max_columns = 30
pd.options.display.max_rows    = 20

def CantidadNulos(df, col=None):
    '''Imprime la cantidad de nulos en cada columna de df.
    Parámetros
    ----------
    df : DataFrame
    \t Dataframe a analizar nulos.
    col: string
    \t Columna específica a analizar.'''
    if col in df.columns:
        Cur_DropNa(df, col, exc=False)
        return
    print('\tCantidad de nulos por columna:')
    for cols in df.columns:
        Cur_DropNa(df, cols, exc=False)
    return

def get_null_fraction(df, col):
    """Fracción nula."""
    return df[col].isnull().sum() / float(df.shape[0])

def get_null_percent(df, col):
    """Porcentaje nulo."""
    return 'NULL = {:.2f}%'.format(100*get_null_fraction(df, col))

class display_side_by_side(object):
    """Display HTML representation of multiple objects."""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
                           
def Memory():
    """Imprime la cantidad de memoria RAM utilizada por el proceso 
    hasta el momento, en GB."""
    gc.collect()
    process = psutil.Process(os.getpid())
    print('Memoria actual utilizada: %.2f GB' 
          %(process.memory_full_info().rss*1e-9))
    return

def Ploteo(Tot, Dif, name=None, h: bool=False, size=(20,5), row: bool=True, order=None, yticks: bool=False):
    """Función para plotear las tablas 'Tot' y 'Dif' (Provenientes de 'Tablas')
    Parámetros
    ----------
    Tot : DataFrame
    \t Dataframe con los porcentajes totales.
    Dif : DataFrame
    \t Dataframe con los porcentajes diferenciados.
    name : string, optional (Default = str(col))
    \t Nombre a colocar en el eje principal.
    h : bool, optional (Default = False)
    \t Si el barplot es horizontal o no.
    size : tuple, optional (Default = (20,5))
    \t Tamaño de la figura.
    row : bool, optional(Default = True)
    \t Si la figura posee 3 columnas (True) o 3 filas (False).
    order : list, optional (Default=np.sort(values))
    \t El orden de ploteo de los valores.
    yticks : bool, optional (Default=False)
    \t Si plotear o no los ticks en caso de que h==True.
    """
    col = Tot.index.name
    if not name:
        name = str(col)
    if not order:
        order = Tot.index
    elif order==True:
        order = np.sort(Tot.index)
    kind  = 'bar'
    if h:
        kind  = 'barh'
        order = order[::-1]
    if row:
        fig, ax = plt.subplots(1,3, figsize=size)
    else:
        fig, ax = plt.subplots(3,1, figsize=size)
    Tot.Porcentaje[order].plot(kind=kind, color=BLUE, ax=ax[0], label='Todos')
    Dif.Porcentaje.unstack(level=1).loc[order].plot(kind=kind, subplots=False, color=[RED, ORANGE], ax=ax[1])
    Tot.Ausentismo[order].plot(kind=kind, color=GREEN, ax=ax[2])
    if not h:
        plt.setp(ax[0], ylabel='Frecuencia')
        plt.setp([ax[0],ax[1],ax[2]], xlabel=name)
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
        ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=0)
    else:
        plt.setp([ax[0], ax[1], ax[2]], xlabel='Frecuencia')
        plt.setp(ax[0], ylabel=name)
        plt.setp([ax[1], ax[2]], ylabel='')
        if not yticks:
            ax[1].set_yticklabels('')
            ax[2].set_yticklabels('')
    ax[0].legend(loc=(1,1), bbox_to_anchor=(0.4, 1.05))
    ax[1].legend(loc=(1,1), bbox_to_anchor=(0.2, 1.05),  ncol=2)
    ax[2].legend(loc=(1,1), bbox_to_anchor=(0.35, 1.05))
    plt.show()

def Porcent(x):
    """Devuelve un string pasando de ratio a porcentaje."""
    return "%.2f%%"%(x*100)
                        
def Tablas(df, col, order=None, name=None):
    """Función para generar tablas 'Total' y 'Diferenciado'.
    Parámetros
    ----------
    df  : DataFrame
    \t Dataframe con los datos.
    col : string
    \t Nombre de la columna a operar.
    order : list, optional (Default="Mayor a menor frecuencia")
    \t Lista con el order de los índices de las tablas.
    name : string, optional (Default=str(col))
    \t Nombre a introducir en como título de las tablas.
    """
    if not name:
        name = str(col)
    Values = df[col].value_counts()
    if not order:
        order = Values.index
    elif order==True:
        order = np.sort(df[col].unique())
    Diferen = pd.DataFrame(df[[col, 'EstadoDelTurno']].groupby([col, 'EstadoDelTurno']).size()).rename(
        columns={0: 'Cantidad'}).reindex(order, level=0)
    Diferen['Porcentaje'] = Diferen.Cantidad.div(Diferen.Cantidad.sum(level=0), level=0)
    Ausent  = Diferen.unstack().Cantidad.Ausente.div(Diferen.unstack().Cantidad.Atendido)
    Values  = Values.loc[order]
    ValuesN = df[col].value_counts(normalize=True).loc[order]
    Total   = pd.concat([Values.rename('Cantidad'), 
                         ValuesN.rename('Porcentaje'), 
                         Values.cumsum().rename('Cant_Acum'), 
                         ValuesN.cumsum().rename('Porcent_Acumu'), 
                         Ausent.rename('Ausentismo')], axis=1).rename_axis(name)
    return Total, Diferen

def get_metrics_train_test(y_true_train, y_pred_train, y_true_test, y_pred_test):
    print('Conjunto de entrenamiento:')
    display(get_metrics(y_true_train, y_pred_train).T.applymap("{0:.2%}".format))
    print('\nConjunto de evaluación:')
    display(get_metrics(y_true_test, y_pred_test).T.applymap("{0:.2%}".format))
    return
    
def get_metrics(y_true, y_pred, index='Porcentaje'):
    """Devuelve un DataFrame con accuracy, precision, recall y f1-score."""
    accuracy     = accuracy_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision    = precision_score(y_true, y_pred)
    recall       = recall_score(y_true, y_pred)
    f1           = f1_score(y_true, y_pred)
    df           = pd.DataFrame(data=[[accuracy, bal_accuracy, precision, recall, f1]],
                    columns=['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    index=[index])
    return df

def get_metrics_comparison(y_true, y_predictions, models):
    metrics_list = []
    for y_pred, model in zip(y_predictions, models):
        df_y       = get_metrics(y_true, y_pred)
        df_y.index = [model]
        metrics_list.append(df_y)
    df_metrics = pd.concat(metrics_list)
    display( df_metrics.style.background_gradient(low=0, high=1, cmap='Greens', axis=0))
    return

def get_confusion_matrix(y_true, y_pred):
    """Devuelve un DataFrame con la matriz de confusión."""    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).transpose()
    df = pd.DataFrame(
        cm, 
        columns=['Actual - Atendido', 'Actual - Ausente'],
        index=['Predicted - Atendido', 'Predicted - Ausente'])
    df.loc['Total'] = df.sum()  # Agrego fila de totales
    df['Total']     = df.sum(axis=1)  # Agrego columna de totales
    df              = df.applymap("{:,}".format)
    return df

def Cur_Lim(df, col=None, Min=None, Max=None, inplace: bool=False,
           exc: bool=True, ret: bool=None):
    '''Función para remover valores numéricos por encima de "Min",
    y por encima de "Max".
    Parámetros:
    -----------
    Min    : Valor límite inferior. (float/int - Default:18)
    Max    : Valor límite superior. (float/int - Default:150)
    col    : Columna a trabajar. (string - Default:None)
    inplace: Sobreescribir DataFrame. (bool - Default:False)
    exc    : Elevar exception. [Opcional] (bool - Default:True)
    ret    : Devolver DataFrame sin eliminados (0) o 
             solamente ellos (1). [Opcional] (bool - Default:None).
    '''
    if not col:
        verb = 'Especifique columna.'
        if exc:
            raise Exception(verb)
        return print(verb)
    if col not in df.columns:
        verb = 'La columna {} no es válida.'.format(col)
        if exc:
            raise Exception(verb)
        return print(verb)
    if (not Min) and (not Max):
        print('No se introdujeron límites para {}.'.
              format(col))
        print('Límites:'
              '\n Mín: {}'
              '\n Máx: {}'.format(
              df[col].min(), df[col].max()))
        verb = 'Especifique al menos un límite.'
        if exc:
            raise Exception(verb)
        return print(verb)
    if ret not in [None, 1, 0]:
        verb = 'Especifique un parámetro de devolución' \
               ' permitido. [None, 1, 0]'
        if exc:
            raise Exception(verb)
        return print(verb)
    a   = len(df)
    Min = df[col].min() if not Min else Min
    Max = df[col].max() if not Max else Max
    Rem = df[(df[col]<Min) | (df[col]>Max)]
    if len(Rem)==0:
        verb = 'No hay valores a eliminar en la columnas'\
               ' {} fuera de límites: [{}, {}]'.format(
                   col, Min, Max)
        if exc:
            raise Exception(verb)
        return print(verb)
    b   = df.drop(Rem.index, inplace=inplace)
    print('Curación de columna {} aplicada con límites:' \
          '[{:d}, {:d}]'.format(col, Min, Max))
    if not inplace:
        print('\tA remover: {:d} filas. (=={:.2f}%)'.format(
            a-len(b), (1-len(b)/a)*100))
        if ret==None:
            return 
        return Rem if ret else b
    print('\tRemovidas: {:d} filas. (=={:.2f}%)'.format(
        a-len(df), (1-len(df)/a)*100))
    if ret==None:
        return 
    return Rem if ret else b

def Cur_Drop(df, col=None, val=None, re: bool=True, inplace: bool=False,
           exc: bool=True, ret: bool=None):
    '''Función para remover entradas con valor "val" en la columna "col".
    Parámetros:
    -----------
    val    : Valor a intercambiar. (string - Default:None)
    col    : Columna a trabajar. (string - Default:None)
    re     : Búsqueda por regex. (bool  - Default:True)
    inplace: Sobreescribir DataFrame. (bool - Default:False)
    exc    : Elevar exception. [Opcional] (bool - Default:True)
    ret    : Devolver DataFrame sin eliminados (0) o 
             solamente ellos (1). [Opcional] (bool - Default:None).
    '''
    if not col:
        verb = 'Especifique columna.'
        if exc:
            raise Exception(verb)
        return print(verb)
    if col not in df.columns:
        verb = 'La columna {} no es válida.'.format(col)
        if exc:
            raise Exception(verb)
        return print(verb)
    if not val:
        verb = 'Especifique valor.'
        if exc:
            raise Exception(verb)
        return print(verb)
    if ret not in [None, 1, 0]:
        verb = 'Especifique un parámetro de devolución' \
               ' permitido. [1, 0]'
        if exc:
            raise Exception(verb)
        return print(verb)
    a = len(df)
    if not re:
        VAL = [val in s for s in df[col]]
        Rem = df[VAL]
        if len(Rem)==0:
            verb = 'No hay aciertos de {} en {}.'.format(val, col)
            if exc:
                raise Exception(verb)
            return print(verb)
        b   = df.drop(Rem.index, inplace=inplace)
    else:
        if val not in df[col].unique():
            verb = '{} no se encuentra en' \
                   ' la columna {}.'.format(val, col)
            if exc:
                raise Exception(verb)
            return print(verb)
        Rem = df[df[col]==val]
        b   = df.drop(Rem.index, inplace=inplace)
    print('Curación {} valor {} en columna {:s} aplicada.'
          .format(('de' if re else 'incluyendo'), val, col))
    if not inplace:
        print('\tA remover: {:d} filas. (=={:.2f}%)'.format(
            a-len(b), (1-len(b)/a)*100))
        if ret==None:
            return 
        return Rem if ret else b
    print('\tRemovidas: {:d} filas. (=={:.2f}%)'.format(
        a-len(df), (1-len(df)/a)*100))
    if ret==None:
        return 
    return Rem if ret else b

def Cur_DropNa(df, col=None, inplace: bool=False, exc: bool=True, 
              ND='No Definido', ret: bool=None):
    '''Función para remover entradas con valor nulo en la columna "col".
    Parámetros:
    -----------
    col    : Columna a trabajar. (string - Default:None)
    inplace: Sobreescribir DataFrame. (bool - Default:False)
    exc    : Elevar exception. [Opcional] (bool - Default:True)
    ND     : Valor referido a Nulo en cuanto a columna tipo "Objeto".
             [Opcional] (string/int/float - Default:'No Definido')
    ret    : Devolver DataFrame sin eliminados (0) o 
             solamente ellos (1). [Opcional] (bool - Default:None).
    '''
    if not col:
        verb = 'Especifique columna.'
        if exc:
            raise Exception(verb)
        return print(verb)
    if col not in df.columns:
        verb = 'La columna {} no es válida.'.format(col)
        if exc:
            raise Exception(verb)
        return print(verb)
    n = df[col].isnull()
    if df[col].dtype == 'O': n = n & (df[col] == ND)
    if n.sum()==0:
        verb = 'La columna {} no posee valores nulos.'.format(col)
        if exc:
            raise Exception(verb)
        return print(verb)
    a   = len(df)
    Rem = df[n]
    b   = df.drop(Rem.index, inplace=inplace)
    print('Curación de Nulos en columna {:s} aplicada.'.format(col))
    if not inplace:
        print('\tA remover: {:d} filas. (=={:.2f}%)'.format(
            a-len(b), (1-len(b)/a)*100))
        if ret==None:
            return 
        return Rem if ret else b
    print('\tRemovidas: {:d} filas. (=={:.2f}%)'.format(
        a-len(df), (1-len(df)/a)*100))
    if ret==None:
        return 
    return Rem if ret else b

def Binarizer(df, col=None, unique=None, re: bool=True, exc: bool=True,
              merge: bool=True, text: bool=True, name=None, drop: bool=False):
    '''Función para binarizar en [0, 1] las entradas 
    de la columna "col". Si se especifica un valor (unique),
    se reemplazará por 0, y al resto 1; de lo contrario, se 
    binarizará en [0, 1] a los 2 valores pre-existentes de "col".
    Parámetros:
    -----------
    col    : Columna a binarizar. (str - Default:None)
    unique : Valor a setear en 1. (str/float/int - Default:None)
    re     : Búsqueda por regex. (bool  - Default:True)
    exc    : Elevar exception. [Opcional] (bool - Default:True)
    merge  : Devolver DataFrame/Series binario creado (0) o el merge
             con el DataFrame original (1). 
             [Opcional] (bool - Default:True).
    text   : Imprimir el proceso realizado.
    name   : Nombre a asignar a la columna creada.
    drop   : Eliminar la columna fuente utilizada para binarizar.
    '''
    if not col:
        verb = 'Especifique columna.'
        if exc:
            raise Exception(verb)
        return print(verb)
    if col not in df.columns:
        verb = 'La columna {} no es válida.'.format(col)
        if exc:
            raise Exception(verb)
        return print(verb)
    vals = df[col].unique()
    if not unique:
        if len(vals)!=2:
            verb = 'La columna introducida posee {} clases (!=2)'.format(
                len(vals))
            if exc:
                raise Exception(verb)
            return print(verb)
        if name is None: name = col+'_'+str(vals[0])
        if name in df.columns:
            verb = 'La columna {} ya existe'.format(name)
            if exc:
                raise Exception(verb)
            return print(verb)
        new  = df[col].replace(
            {vals[0]: 1, vals[1]: 0}).rename(name)
        if text:
            print('Reemplazo:'
                  '\n 1 = {} ({:.2f}%)'
                  '\n 0 = {}'.
                  format(vals[0], new.sum()*100/len(df),
                         vals[1]))
    else:
        if name is None: name = col+'_'+str(unique)
        if name in df.columns:
            verb = 'La columna {} ya existe'.format(name)
            if exc:
                raise Exception(verb)
            return print(verb)
        if re:
            if unique not in vals:
                verb = '{} no se encuentra en la columna {}.'.format(
                    unique, col)
                if exc:
                    raise Exception(verb)
                return print(verb)
            val = df[col] == unique
            can = sum(val)
        else:
            val = [unique in s for s in df[col]]
            can = sum(val)
            if can==0:
                verb = 'La expresión {} no se encuentra en ningúna' \
                        ' entrada de la columna {}.'.format(unique, col)
                if exc:
                    raise Exception(verb)
                return print(verb)
        new = df[col].copy().rename(name)
        new.loc[np.asarray(val)]  = 1
        new.loc[~np.asarray(val)] = 0
        if text:
            print('Reemplazo:'
                  '\n 1 = {} ({:.2f}%)'
                  '\n 0 = Resto'.format(
                      unique, can*100/len(df)))
    print('Columna {} creada'.format(name))
    if not merge: return new.astype('uint8')
    if drop: 
        print('Se elimina columna {}'.format(col))
        return df.merge(new.astype('uint8'), left_index=True,
                        right_index=True).drop(col, axis=1)
    return df.merge(new.astype('uint8'),
           left_index=True,right_index=True)

def BinarAll(df, cols=None, exc: bool=True, merge: bool=False,
             text: bool=True, fast: bool=True, nan: bool=False):
    '''Función para binarizar en [0, 1] TODAS las entradas 
    de la columna "col".
    Parámetros:
    -----------
    cols   : Columna(s) a binarizar. (str/list - Default:None)
    exc    : Elevar exception. [Opcional] (bool - Default:True)
    merge  : Devolver DataFrame/Series binario creado (1) o el merge
             con el DataFrame original (0). 
             [Opcional] (bool - Default:None).
    text   : Imprimir el proceso realizado.
    fast   : Método rápido. [Opcional] (bool - Default:True)
    nan    : Incluir (True) o no los NaN al binarizar.
    '''
    if not cols:
        verb = 'Especifique columna(s).'
        if exc:
            raise Exception(verb)
        return print(verb)
    if isinstance(cols, str): cols = [cols]
    for col in cols:
        if col not in df.columns:
            verb = 'La columna {} no es válida.'.format(col)
            if exc:
                raise Exception(verb)
            return print(verb)
    if merge:
        return pd.get_dummies(df, columns=cols, dummy_na=nan)
    if fast:
        extra = [cl for cl in df.columns if cl not in cols]
        return pd.get_dummies(
            df, columns=cols, dummy_na=nan).drop(columns=extra, axis=1)
    for col in cols:
        vals = df[col].unique()
        if 'sr' not in locals():
            print('First')
            sr = Binarizer(df, col=col, unique=vals[0], exc=exc,
                           text=text).astype('uint8')
        for val in vals[1:]:
            sr = pd.concat([sr, Binarizer(
                    df, col=col, unique=val, exc=exc,
                    text=text).astype('uint8')], axis=1)
    return sr

def Metrics(y_real, y_pred, name='Entrenamiento'):
    '''Imprime las métricas comunes (Ac, Pr, Re, F1, CM)
    dados los Target y sus predicciones.'''
    print('Métricas {}:'
          '\n {}'
          '\n Confusion Matrix: \n{}'.format(
              name, classification_report(y_real, y_pred, digits=3), 
              confusion_matrix(y_real, y_pred)))

def HM(y_real, y_pred, ax=None, cmap='Greys', normalize=None):
    '''Plot simple de heatmap.'''
    kwargs = dict(annot=True, linewidths=1.2, square=True,
                   linecolor='black', cbar=False, cmap=cmap,
                   annot_kws={"weight":'bold', "size":'15'}, ax=ax)
    if normalize: return heatmap(confusion_matrix(y_real, y_pred,
                            normalize=normalize), fmt='.3f', **kwargs)
    return heatmap(confusion_matrix(y_real, y_pred,
                            normalize=normalize), fmt='d', **kwargs)

def PlotHM(y_train, y_train_pred, y_test, y_test_pred, size=(10,6), cmap='Greys', 
           Title='Título', normalize=None):
    '''Plotea las 2 matrices de confusión (Entrenamiento y Prueba),
    dados los Target y sus predicciones (Train y Test).'''
    plt.figure(figsize=size)
    plt.suptitle(Title, y=.93)
    plt.subplot(1,2,1)
    plt.title('Entrenamiento')
    ax1 = HM(y_train, y_train_pred, cmap=cmap, normalize=normalize)
    plt.ylabel('Real', fontsize=15)
    plt.subplot(1,2,2)
    plt.title('Prueba')
    ax2  = HM(y_test, y_test_pred, cmap=cmap, normalize=normalize)
    plt.setp([ax1.get_xticklabels(), ax2.get_xticklabels()], rotation=0, weight='bold')
    plt.setp([ax1.get_yticklabels(), ax2.get_yticklabels()], rotation=0, weight='bold')
    plt.setp([ax1.set_xlabel('Predecido'), ax2.set_xlabel('Predecido')], fontsize=15)
    if normalize: 
        [t.set_text(t.get_text() + " %") for t in ax1.texts]
        [t.set_text(t.get_text() + " %") for t in ax2.texts]
    plt.show()

def PlotROC(y_real, y_pred, ls='.--', ms=15, color=RED, label=None,
            x=0.55, y =0.3, fsize=12, ecolor='white'):
    """Plot simple de curva ROC, y el texto de área debajo de ella."""
    plt.text(x, y, 'Área bajo curva: %.3f' %roc_auc_score(y_real, y_pred), weight='bold',
             size=fsize, bbox=dict(boxstyle="round", ec=ecolor, fc=color))
    roc = roc_curve(y_real, y_pred)
    return plt.plot(roc[0], roc[1], ls, ms=ms, color=color, label=label)

def PlotPR(y_real, y_pred, ls='.--', ms=15, color=RED, label=None):
    """Plot simple de curva PR."""
    PR  = precision_recall_curve(y_real, y_pred)
    return plt.plot(PR[0], PR[1], ls, ms=ms, color=color, label=label)

def PlotCurves(y_real, y_pred, Title='Title',
               x_text=0.4, y_text=0.2, size_text=12, Ret_Vals: bool=False, size=(14,6),
               ms=15, colorR=RED, colorP=BLUE):
    '''Plotea las 2 curvas (ROC y PR), dados los Target y sus predicciones.
    -> Ret_Vals (bool): Devolver métricas calculadas.'''
    plt.figure(figsize=size)
    plt.suptitle(Title)
    ax1 = plt.subplot(1,2,1)
    plt.title('ROC')
    PlotROC(y_real, y_pred, ms=ms, color=colorR, x=x_text, y=y_text,
            ecolor=(1., 0.5, 0.5), fsize=size_text)
    ax2 = plt.subplot(1,2,2)
    plt.title('PR')
    PlotPR(y_real, y_pred, ms=ms, color=colorP)
    plt.setp([ax1.set_xlabel('False Positive Rate'), ax1.set_ylabel('True Positive Rate'),
              ax2.set_xlabel('Recall'), ax2.set_ylabel('Precision')], fontsize=15)
    plt.show()
    if Ret_Vals: 
        print('Salida: ROC == [FPR, TPR, Thresh], PR == [Prec, Recall, Thresh]')
        return roc_curve(y_real, y_pred), precision_recall_curve(y_real, y_pred)