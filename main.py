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
from streamlit_folium import folium_static
from helper import create_map, get_similar
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
df = pd.read_csv('updated_output.csv')

st.title("Lets get you treated !")
query_text = st.text_input("Enter disease name")
option = st.radio(
        "What do you want to visualise",
        ('Top Similar Diseases', 'More Similar Diseases', 'All'))

df['Speciality'] = df['Speciality'].apply(lambda x: x.split(', '))
merged_list = list(set([item for sublist in df['Speciality'] for item in sublist]))

similar_text, vectors, query_vector = get_similar(merged_list, encoder, faiss, query_text)

if query_text != '':
    st.success("Similar Diseases : " + str(similar_text))
    very_similar = similar_text[0:5]
    less_similar = similar_text[6:10]

    final_vector = np.append(vectors, query_vector, axis=0)

    tsne = TSNE(n_components=3, perplexity=2)
    embeddings = tsne.fit_transform(final_vector)
    merged_list.append(query_text)

    plot_df = pd.DataFrame({'X': embeddings[:, 0], 'Y': embeddings[:, 1], 'Z': embeddings[:, 2], 'Element': merged_list})
    plot_df['Color'] = plot_df['Element'].apply(lambda x: 'blue' if x in very_similar else 'red' if x == query_text else 'green' if x in less_similar else 'grey')

    if option == 'Only Similar Diseases':
      option_df = plot_df[(plot_df['Element'].isin(very_similar)) | (plot_df['Element'] == query_text)]

    if option == 'Less Similar Diseases':
      option_df = plot_df[(plot_df['Element'].isin(very_similar)) | (plot_df['Element'] == query_text) | (plot_df['Element'].isin(less_similar))]

    if option == 'All':
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

    #creating empty df
    final_df = pd.DataFrame(columns=['Doctor', 'url', 'Speciality', 'Address','Distance(miles)','Timings','Insurance'])
    #filter based on similar diseases
    for i in similar_text:
      temp_df = df[df['Speciality'].apply(lambda x: i in x)]
      final_df = final_df.append(temp_df)

    final_df = final_df[0:5]

    st.title('List of Doctors')
    for i, row in final_df.iterrows():
        doctor_name = row['Doctor']
        speciality = row['Speciality']
        clinic_distance = row['Distance(miles)']

        # Convert the list of specialities to a string
        speciality_str = ', '.join(speciality)
        # Remove the square brackets and single quotes
        # speciality_str = speciality_str.replace("[", "").replace("]", "").replace("'", "")


        # Highlight keywords in the speciality
        for keyword in similar_text:
            if keyword in speciality_str:
                speciality_str = speciality_str.replace(keyword, f"**<span style='background-color: lightgreen;'>{keyword}</span>**")

        # Display the doctor's details
        st.markdown(f"**{doctor_name}**")
        st.markdown(f"**Speciality:** {speciality_str}", unsafe_allow_html=True)
        st.write('**Distance (miles):**', clinic_distance)
        st.write('---')

     # Display the map in Streamlit
    st.title('Doctor Map')
    st.write('Map showing Santa Clara University and clinic locations')

    map = create_map(final_df)
    folium_static(map)

