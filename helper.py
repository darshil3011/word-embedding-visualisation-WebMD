import streamlit as st
import folium
from geopy.distance import geodesic
from streamlit_folium import folium_static

def create_map():
    # Create a map centered at Santa Clara University
    map_center = (37.349, -121.939)
    m = folium.Map(location=map_center, zoom_start=8)

    # Add Santa Clara University marker with two icons
    santa_clara_marker = folium.Marker(
        location=(37.349, -121.939),
        popup='Santa Clara University',
        icon=folium.Icon(color='red', icon='university'),
    )
    santa_clara_marker.add_child(folium.Icon(color='red', icon='star'))
    santa_clara_marker.add_to(m)

    # Add clinic markers in blue color and connect with lines
    for i,doctor in final_df.iterrows():
        # Create HTML for doctor name with clickable link
        doctor_name = f"<a href='{doctor['url']}' target='_blank' style='text-decoration: underline; color: #333; font-weight: bold;'>{doctor['Doctor']}</a>"

        clinic_marker = folium.Marker(
            location=(doctor['Latitude'], doctor['Longitude']),
            popup=doctor_name,
            icon=folium.Icon(color='blue', icon='clinic-medical', prefix='fa'),
        ).add_to(m)

        distance = calculate_distance(map_center, (doctor['Latitude'], doctor['Longitude']))

        folium.PolyLine(
            locations=[map_center, (doctor['Latitude'], doctor['Longitude'])],
            color='blue',
            weight=1.5,
            opacity=1,
            tooltip=f"Distance: {distance} miles",
        ).add_to(m)

    return m

def calculate_distance(point1, point2):
    # Calculate distance between two points in miles using geodesic distance
    return round(geodesic(point1, point2).miles, 2)
