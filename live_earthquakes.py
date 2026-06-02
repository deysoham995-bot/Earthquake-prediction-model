"""
Live earthquake data fetcher from USGS API
Provides real-time earthquake information and visualization
"""

import requests
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class LiveEarthquakeFetcher:
    """Fetch real-time earthquake data from USGS Earthquake Hazards Program"""
    
    # USGS API endpoints
    USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
    
    @staticmethod
    def fetch_earthquakes(time_period='day', min_magnitude=0):
        """
        Fetch earthquake data from USGS API
        
        Args:
            time_period: 'hour', 'day', 'week', 'month'
            min_magnitude: Minimum magnitude to include
        
        Returns:
            pd.DataFrame with earthquake data
        """
        try:
            url = f"{LiveEarthquakeFetcher.USGS_API_URL}/all_{time_period}.geojson"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            earthquakes = []
            
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                eq = {
                    'Longitude': coords[0],
                    'Latitude': coords[1],
                    'Depth_km': coords[2],
                    'Magnitude': props['mag'],
                    'Location': props.get('place', 'Unknown'),
                    'Time': datetime.utcfromtimestamp(props['time'] / 1000),
                    'Time_UTC': datetime.utcfromtimestamp(props['time'] / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'Type': props.get('type', 'earthquake'),
                    'ID': props.get('id', ''),
                    'Status': props.get('status', 'automatic'),
                    'URL': props.get('url', ''),
                }
                
                if eq['Magnitude'] >= min_magnitude:
                    earthquakes.append(eq)
            
            df = pd.DataFrame(earthquakes)
            
            # Sort by time (newest first)
            if not df.empty:
                df = df.sort_values('Time', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching earthquake data: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def create_map(df, center_lat=20, center_lon=0, zoom=2):
        """
        Create interactive Folium map with earthquake markers
        
        Args:
            df: DataFrame with earthquake data
            center_lat, center_lon: Map center
            zoom: Map zoom level
        
        Returns:
            folium.Map object
        """
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        if df.empty:
            return m
        
        # Add marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in df.iterrows():
            # Magnitude color coding
            magnitude = row['Magnitude']
            if magnitude >= 7:
                color = 'red'
                intensity = '🔴 Major'
            elif magnitude >= 6:
                color = 'darkred'
                intensity = '🟠 Strong'
            elif magnitude >= 5:
                color = 'orange'
                intensity = '🟡 Moderate'
            elif magnitude >= 4:
                color = 'yellow'
                intensity = '🟢 Light'
            else:
                color = 'green'
                intensity = '🔵 Minor'
            
            # Marker size based on magnitude
            radius = max(5, magnitude * 2)
            
            # Create popup with details
            popup_text = f"""
            <b>Earthquake Details</b><br>
            <b>Magnitude:</b> {magnitude:.1f} {intensity}<br>
            <b>Location:</b> {row['Location']}<br>
            <b>Depth:</b> {row['Depth_km']:.1f} km<br>
            <b>Time (UTC):</b> {row['Time_UTC']}<br>
            <b>Type:</b> {row['Type']}<br>
            <b>Status:</b> {row['Status']}
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=300),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        return m
    
    @staticmethod
    def get_statistics(df):
        """Calculate earthquake statistics"""
        if df.empty:
            return {
                'total': 0,
                'major': 0,
                'strong': 0,
                'moderate': 0,
                'avg_magnitude': 0,
                'max_magnitude': 0,
                'avg_depth': 0
            }
        
        return {
            'total': len(df),
            'major': len(df[df['Magnitude'] >= 7]),
            'strong': len(df[(df['Magnitude'] >= 6) & (df['Magnitude'] < 7)]),
            'moderate': len(df[(df['Magnitude'] >= 5) & (df['Magnitude'] < 6)]),
            'avg_magnitude': df['Magnitude'].mean(),
            'max_magnitude': df['Magnitude'].max(),
            'avg_depth': df['Depth_km'].mean()
        }
    
    @staticmethod
    def get_top_regions(df, limit=10):
        """Get top earthquake regions"""
        if df.empty or 'Place' not in df.columns:
            return pd.DataFrame()
        # Extract region names
        df['Region'] = df['Place'].apply(
            lambda x: x.split(',')[-1].strip() if ',' in str(x) else str(x)
        )
        # Group data
        top_regions = (
            df.groupby('Region')
            .agg(
                earthquake_count=('Magnitude', 'count'),
                avg_magnitude=('Magnitude', 'mean'),
                max_magnitude=('Magnitude', 'max')
            )
            .reset_index()
            .sort_values('earthquake_count', ascending=False)
            .head(limit)
        )
        return top_regions