import io
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import plotly.tools as tls
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, plot,iplot

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

sns.set()
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_columns',100)

BLUE  = '#3498DB'
RED   = '#C0392B'
GREEN = '#52BE80'

def InerciaSilueta(df, max_iner=10, max_sil=6, 
                   skill_1='attacking_finishing', skill_2='defending_sliding_tackle',
                   random_state=10):
    #Inercia
    print('Análisis de Inercia')
    fits   = []
    scores = []
    for i in range(max_iner):
        fits.append(KMeans(n_clusters=i+2, random_state=random_state).fit(df))
        scores.append(fits[i].inertia_)    
    plt.figure(dpi=150)
    plt.title('Inercia de K-Means vs. Número de Clusters')
    plt.plot(np.arange(2, max_iner+2), scores)
    plt.xlabel('Número de clusters')
    plt.ylabel("Inercia")
    plt.show()
    print('')
    print('')
    print('Análisis de Coeficiente de Silueta')
    #Silueta    
    if isinstance(skill_1, str): skill_1 = int(np.where(df.columns==skill_1)[0])
    if isinstance(skill_2, str): skill_2 = int(np.where(df.columns==skill_2)[0])
    for n_clusters in range(2, max_sil+1):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
        cluster_labels = fits[n_clusters-2].predict(df)
        silhouette_avg = silhouette_score(df, cluster_labels)
        print("Para n_clusters =", n_clusters,
              "El coeficiente de silueta promedio es de:", silhouette_avg)
        sample_silhouette_values = silhouette_samples(df, cluster_labels)
        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper        = y_lower + size_cluster_i
            color          = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        ax1.set_title('Plot de Silueta para varios clustering')
        ax1.set_xlabel('Valores del coeficiente de silueta')
        ax1.set_ylabel('Cluster label')
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(df.iloc[:, skill_1], df.iloc[:, skill_2],
                    marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
        centers = fits[n_clusters-2].cluster_centers_
        ax2.scatter(centers[:, skill_1], centers[:, skill_2], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centers):
            ax2.scatter(c[skill_1], c[skill_2], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title('Visualización de Datos en cluster')
        ax2.set_xlabel(df.columns[skill_1])
        ax2.set_ylabel(df.columns[skill_2])

        plt.suptitle(('Análisis de silueta para clustering K-Means con' \
                      'n_clusters = %d' % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()
    return 

def PlotScatter(df, skill_1='attacking_finishing', skill_2='defending_sliding_tackle',
                   cluster_col='kmeans', title='Clustering K means'):
    data=[]
    bool_crack    = df["overall"] > 84
    bool_no_crack = df["overall"] < 85
    if isinstance(skill_1, int): skill_1 = df.columns[skill_1]
    if isinstance(skill_2, int): skill_2 = df.columns[skill_2]
    k_clus  = go.Scatter(x=df[skill_1], y=df[skill_2], mode='markers',
                         name='Todos los Jugadores', text=df.loc[:,'short_name'],
                         marker=dict(
                              size=5, color=df[cluster_col].astype(np.float), 
                              colorscale='portland', showscale=False))
    crack   = go.Scatter(x=df.loc[bool_crack, skill_1], y=df.loc[bool_crack, skill_2],
                         name='Mejores Jugadores', text=df.loc[bool_crack,'short_name'],
                          textfont=dict(family='sans serif', size=10, color='black'),
                          opacity=0.9, mode='text')
    data    = [k_clus, crack]
    layout  = go.Layout(title=title, titlefont=dict(size=20),
                    xaxis=dict(title=skill_1),
                    yaxis=dict(title=skill_2),
                    autosize=False, width=1000, height=1000)
    fig     = go.Figure(data=data, layout=layout)
    iplot(fig)
    return
    
def PlotsClusters(df, cluster_col='kmeans', compare_col='POS',
                 first=True, second=False, ax=False, **CPargs):
    if ax: return sns.countplot(df[cluster_col], ax=ax, **CPargs)
    #Cantidad x Cluster
    if first:
        plt.figure(dpi=100)
        sns.countplot(df[cluster_col], **CPargs)
        plt.title('Cantidad de jugadores \npor cluster en {}'.format(cluster_col))
        plt.xlabel('Cluster')
        plt.ylabel('Cantidad de \n jugadores')
        plt.show()
    # Comparación x Cluster
    if second:
        n_clust  = df[cluster_col].nunique()
        fig, axs = plt.subplots((n_clust+1)//2, 2)
        fig.set_size_inches((25,10))
        fig.suptitle('Cantidad de jugadores por cluster, discriminando por: \n"{}"'.format(compare_col),
                        size=20)
        for clust in range(n_clust):
            ax = axs[clust//2,int(clust%2)] if n_clust>2 else axs[clust]
            sns.countplot(df[compare_col][df[cluster_col]==clust], ax=ax)
            ax.set_ylabel('Cantidad de jugadores')
            ax.set_xlabel('')
            ax.set_title('Cluster {}'.format(clust), fontweight='bold')
        plt.show()
    return

def HeatMap(df, col1='kmeans', col2='POS', table=False, ax=False, 
            normed=True, **HMargs):
    cruz = pd.crosstab(index=df[col1], columns=df[col2])
    norm = cruz/cruz.sum(axis=0)
    if ax: return sns.heatmap(norm if normed else cruz, annot=True, 
                              fmt= ".2f" if normed else "d",
                              square=True, linewidths=.5, ax=ax, **HMargs)
    if table:
        print('Tabla Cruzada Normalizada de {} respecto a {}'.format(col1, col2))
        with pd.option_context('display.precision', 3):
            display(norm)
    plt.figure(dpi=100)
    plt.title('Mapa de calor de {} respecto a \n {}'.format(col1, col2), fontsize=12)
    ax = sns.heatmap(norm if normed else cruz, annot=True, 
                              fmt= ".2f" if normed else "d",
                              square=True, linewidths=.5, **HMargs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()
    return