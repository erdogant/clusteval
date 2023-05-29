# Libraries
import numpy as np
import pandas as pd

import flameplot as flameplot
from clusteval import clusteval
from scatterd import scatterd
import datazets as dz
from df2onehot import df2onehot

# %%
df = dz.get('ds_salaries.zip')

# %%
countries_europe = ['SM', 'DE', 'GB', 'ES', 'FR', 'RU', 'IT', 'NL', 'CH', 'CF', 'FI', 'UA', 'IE', 'GR', 'MK', 'RO', 'AL', 'LT', 'BA', 'LV', 'EE', 'AM', 'HR', 'SI', 'PT', 'HU', 'AT', 'SK', 'CZ', 'DK', 'BE', 'MD', 'MT']
df['europe'] = np.isin(df['company_location'], countries_europe)
# df = df.loc[df['europe'], :]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# fig, ax = plt.subplots(1, 2, figsize=(30, 15))
n = 25
top_n = df['job_title'].value_counts().head(n)
plt.figure(figsize=(12, 12))
sns.barplot(x=top_n.values, y=top_n.index)
plt.title('Top Job Titles Europe')
plt.xlabel('Frequency')
plt.ylabel('Job Title')
plt.grid(True)
# ax[0].show()


top_n = df.groupby('job_title')['salary_in_usd'].mean().nlargest(n)
plt.figure(figsize=(12, 12))
sns.barplot(x=top_n.values, y=top_n.index)
plt.title('Top Job Titles with Highest Salary Europe')
plt.xlabel('Average Salary (USD)')
plt.ylabel('Job Title')
plt.grid(True)
# plt.show()


# %%
# Remove columns
y = df['salary_in_usd']
df.drop(labels=['salary_currency', 'salary', 'salary_in_usd'], inplace=True, axis=1)

# Replace pattern of string using regular expression.
df['experience_level'] = df['experience_level'].replace({'EN':'Entry-level', 'MI':'Junior Mid-level', 'SE':'Intermediate Senior-level', 'EX':'Expert Executive-level / Director'}, regex=True)
df['employment_type'] = df['employment_type'].replace({'PT':'Part-time', 'FT':'Full-time', 'CT':'Contract', 'FL':'Freelance'}, regex=True)
df['company_size'] = df['company_size'].replace({'S':'Small (less than 50)', 'M':'Medium (50 to 250)', 'L':'Large (>250)'}, regex=True)
df['remote_ratio'] = df['remote_ratio'].replace({0:'No remote', 50:'Partially remote', 100:'>80% remote'}, regex=True)
df['work_year'] = df['work_year'].astype(str)


# %%
dfhot = df2onehot(df,
                  remove_multicollinearity=True,
                  y_min=5,
                  verbose=4)['onehot']

# %%
from pca import pca
model = pca(normalize=False)
model.fit_transform(dfhot)
model.plot()

# %%
model.biplot(labels=df['job_title'],
             s=y/100,
             marker=df['experience_level'],
             n_feat=10,
             density=True,
             fontsize=0,
             jitter=0.05,
             alpha=0.8,
             color_arrow='#000000',
             arrowdict={'color_text': '#000000', 'fontsize': 28},
             figsize=(40, 30),
             verbose=4,
             )

# %%
model.scatter(labels=df['experience_level'],
             s=y/100,
             marker=df['job_title'],
             density=True,
             fontsize=40,
             jitter=0.05,
             alpha=0.8,
             figsize=(40, 30),
             verbose=4,
             grid=True,
             legend=False,
             )

# %% Remove work year
df.drop(labels=['work_year'], inplace=True, axis=1)
dfhot = df2onehot(df, remove_multicollinearity=True, y_min=5, verbose=4)['onehot']

# %%
from sklearn.manifold import TSNE
# X = TSNE(n_components=2, init='pca', perplexity=50, metric='hamming').fit_transform(dfhot.values)
X = TSNE(n_components=2, init='pca', perplexity=100).fit_transform(dfhot.values)

# %%
fig, ax = scatterd(X[:, 0],
                   X[:, 1],
                   marker=df['experience_level'],
                   s=y/100,
                   labels=df['job_title'],
                   fontweight='normal',
                   fontsize=0,
                   density=True,
                   density_on_top=False,
                   args_density={'alpha': 0.4},
                   gradient='opaque',
                   edgecolor='#000000',
                   jitter=1,
                   grid=True,
                   legend=False,
                   figsize=(40, 30),
                   )


# %%
from scatterd import scatterd, jitter_func
from d3blocks import D3Blocks, normalize
import numpy as np

# Initialize
d3 = D3Blocks()

tooltip = []
for i in range(0, df.shape[0]):
    tip = '<br>'.join(list(map(lambda x: x[0].replace('_', ' ').title()+': '+x[1], np.array(list(zip(df.columns, df.iloc[i,:].values))))))
    tip = tip + '<br>' + 'Salary: $' + str(y[i])
    tooltip.append(tip)

# Set all propreties
d3.scatter(jitter_func(X[:,0], jitter=1),      # PC1 x-coordinates
           jitter_func(X[:,1], jitter=1),      # PC2 y-coordinates
           x1=jitter_func(model.results['PC']['PC1'].values, jitter=0.05),
           y1=jitter_func(model.results['PC']['PC2'].values, jitter=0.05),
           color=df['job_title'].values,       # Hex-colors or classlabels
           tooltip=tooltip,                    # Tooltip
           size=normalize(y.values,
                          minscale=1,
                          maxscale=25),        # Node size
           opacity='opaque',                   # Opacity
           stroke='#000000',                   # Edge color
           cmap='tab20',                       # Colormap
           scale=True,                         # Scale the datapoints
           label_radio=['tSNE', 'PCA'],
           figsize=[1024, 768],
           filepath='c://temp//data_science_landscape.html',
           )


# %%
ce = clusteval(cluster='dbscan', metric='euclidean', linkage='complete', min_clust=3, normalize=True, verbose='info')
results = ce.fit(X)
results = ce.enrichment(df)

# %%
ce.plot(figsize=(12,5))
ce.plot_silhouette(jitter=0.05)

# %%
ce.scatter(n_feat=4,
           s=y/100,
           jitter=0.05,
           fontsize=18,
           density=True,
           params_scatterd={'marker':df['experience_level'], 'gradient':'opaque', 'dpi':200, 'alpha': 0.2},
           figsize=(40,30),
           )

ce.scatter(n_feat=4,
           s=0, jitter=0.05,
           fontsize=18,
           density=True,
           params_scatterd={'marker':df['experience_level'], 'gradient':'opaque', 'dpi':200, 'alpha': 0.2},
           figsize=(40,30))

