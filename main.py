import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import DPRContextEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE 
import plotly.graph_objects as go
import pandas as pd

encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
df = pd.read_csv('output.csv')

df['Speciality'] = df['Speciality'].apply(lambda x: x.split(', '))
merged_list = list(set([item for sublist in df['Speciality'] for item in sublist]))
text = merged_list
vectors = encoder.encode(text)
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

query_text = "Human herpesvirus 6 infection"

# Encode the query text
query_vector = encoder.encode([query_text])

# Search for similar text
k = 5  # Number of nearest neighbors to retrieve
distance, indices = index.search(query_vector, k=10)
print(distance)
# Retrieve the similar text based on the indices
similar_text = [merged_list[i] for i in indices[0]]

very_similar = similar_text[0:5]
less_similar = similar_text[6:10]

final_vector = np.append(vectors, query_vector, axis=0)

tsne = TSNE(n_components=3, perplexity=2)
embeddings = tsne.fit_transform(final_vector)
text.append(query_text)

option_1 = False
option_2 = False
option_3 = True

plot_df = pd.DataFrame({'X': embeddings[:, 0], 'Y': embeddings[:, 1], 'Z': embeddings[:, 2], 'Element': text})
plot_df['Color'] = plot_df['Element'].apply(lambda x: 'blue' if x in very_similar else 'red' if x == query_text else 'green' if x in less_similar else 'grey')

if option_1 == True:
  option_df = plot_df[(plot_df['Element'].isin(very_similar)) | (plot_df['Element'] == query_text)]

if option_2 == True:
  option_df = plot_df[(plot_df['Element'].isin(very_similar)) | (plot_df['Element'] == query_text) | (plot_df['Element'].isin(less_similar))]

if option_3 == True:
  option_df = plot_df

fig = go.Figure()

# Add scatter plot
fig.add_trace(go.Scatter3d(
    x=option_df['X'],
    y=option_df['Y'],
    z=option_df['Z'],
    mode='markers',
    marker=dict(
        color=option_df['Color'],
        opacity=0.7
    ),
    text=option_df['Element'],
    hoverinfo='text'
))

# Connect black element with all red elements
black_element = option_df[option_df['Color'] == 'red']
red_elements = option_df[option_df['Color'] == 'blue']

for index, row in black_element.iterrows():
    for red_index, red_row in red_elements.iterrows():
        fig.add_trace(go.Scatter3d(
            x=[row['X'], red_row['X']],
            y=[row['Y'], red_row['Y']],
            z=[row['Z'], red_row['Z']],
            mode='lines',
            line=dict(
                color='black',
                width=2
            ),
            showlegend=True
        ))

# Update layoutx
fig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)

st.plotly_chart(fig)


