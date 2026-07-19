import os
import time
from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib
# Use the non-interactive Agg backend to prevent crashes in web environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import predict_severity

app = Flask(__name__)

# Load dataset and prepare lists
DATASET_PATH = "indian_roads_dataset.csv"
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
else:
    # Fallback empty DataFrame in case dataset is missing
    df = pd.DataFrame(columns=[
        'accident_id', 'city', 'state', 'latitude', 'longitude', 'date', 'time',
        'hour', 'day_of_week', 'is_weekend', 'road_type', 'lanes', 'traffic_signal',
        'weather', 'visibility', 'temperature', 'traffic_density', 'cause',
        'accident_severity', 'vehicles_involved', 'casualties', 'is_peak_hour',
        'festival', 'risk_score'
    ])

# Extract unique filters for dropdowns
UNIQUE_STATES = sorted(df['state'].dropna().unique()) if not df.empty else []
UNIQUE_CITIES = sorted(df['city'].dropna().unique()) if not df.empty else []
UNIQUE_WEATHER = sorted(df['weather'].dropna().unique()) if not df.empty else []
UNIQUE_ROAD_TYPES = sorted(df['road_type'].dropna().unique()) if not df.empty else []
UNIQUE_SEVERITIES = sorted(df['accident_severity'].dropna().unique()) if not df.empty else []

def get_road_name(city, road_type):
    """Generate realistic highway/road names for India based on city and road type."""
    roads = {
        'Pune': {
            'highway': 'NH 48 - Mumbai Pune Expressway',
            'urban': 'Senapati Bapat Road',
            'rural': 'Mulshi Road'
        },
        'Bengaluru': {
            'highway': 'NH 44 - Hosur Road',
            'urban': 'Outer Ring Road',
            'rural': 'Sarjapur Road'
        },
        'Lucknow': {
            'highway': 'NH 44 - Delhi to Lucknow',
            'urban': 'Hazratganj Road',
            'rural': 'Malihabad Road'
        },
        'Mumbai': {
            'highway': 'Eastern Express Highway',
            'urban': 'Western Express Highway',
            'rural': 'Kalyan-Murbad Road'
        },
        'Chennai': {
            'highway': 'NH 48 - Chennai Bypass',
            'urban': 'Mount Road (Anna Salai)',
            'rural': 'ECR (East Coast Road)'
        },
        'Kolkata': {
            'highway': 'NH 12 - Jessore Road',
            'urban': 'Park Street',
            'rural': 'Kona Expressway'
        },
        'Delhi': {
            'highway': 'NH 44 - Grand Trunk Road',
            'urban': 'Ring Road',
            'rural': 'Narela Road'
        },
        'Chandigarh': {
            'highway': 'Himalayan Expressway',
            'urban': 'Madhya Marg',
            'rural': 'Mullanpur Road'
        },
        'Ahmedabad': {
            'highway': 'NE 1 - Ahmedabad Vadodara Expressway',
            'urban': 'Ashram Road',
            'rural': 'Sarkhej Highway'
        },
        'Jaipur': {
            'highway': 'NH 48 - Jaipur Bypass',
            'urban': 'Ajmer Road',
            'rural': 'Tonk Road'
        },
        'Hyderabad': {
            'highway': 'Nehru Outer Ring Road',
            'urban': 'Gachibowli Road',
            'rural': 'Warangal Highway'
        },
        'Indore': {
            'highway': 'NH 52 - Agra Bombay Road',
            'urban': 'MG Road',
            'rural': 'Dhar Road'
        },
        'Patna': {
            'highway': 'NH 31 - Patna Bypass',
            'urban': 'Bailey Road',
            'rural': 'Digha Road'
        },
        'Bhopal': {
            'highway': 'NH 46 - Bhopal Bypass',
            'urban': 'Hoshangabad Road',
            'rural': 'Kolar Road'
        },
        'Nagpur': {
            'highway': 'NH 44 - Wardha Road',
            'urban': 'Amravati Road',
            'rural': 'Katol Road'
        }
    }
    city_roads = roads.get(city, {})
    return city_roads.get(road_type, f"{city} {road_type.capitalize()} Main Road")

def ensure_folders():
    """Ensure that the static folders exist before writing files."""
    os.makedirs(os.path.join(app.root_path, 'static', 'graphs'), exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'maps'), exist_ok=True)

def generate_matplotlib_graphs(filtered_df, graphs_dir, total_accidents):
    """Generate the four Matplotlib graphs stylized like the mockup and save as PNG."""
    # Set global font family
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Plus Jakarta Sans', 'DejaVu Sans', 'Arial']

    # 1. ACCIDENTS BY CITY (Vertical Bar Chart)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not filtered_df.empty:
        city_counts = filtered_df['city'].value_counts().head(5)
        # Deep blue color
        bars = ax.bar(city_counts.index, city_counts.values, color='#1e56a0', edgecolor='none', width=0.45)
        ax.bar_label(bars, fmt='%d', padding=3, fontsize=9, fontweight='bold', color='#475569')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(left=False)
    ax.grid(axis='y', linestyle='-', alpha=0.2, color='#cbd5e1')
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, fontsize=9, fontweight='semibold', color='#475569')
    plt.yticks(fontsize=9, color='#64748b')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'accidents_by_city.png'), dpi=150)
    plt.close()

    # 2. SEVERITY DISTRIBUTION (Donut Chart)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not filtered_df.empty:
        severity_counts = filtered_df['accident_severity'].value_counts()
        colors_map = {'fatal': '#ef4444', 'major': '#f59e0b', 'minor': '#10b981'}
        colors = [colors_map.get(x, '#3b82f6') for x in severity_counts.index]
        
        # Donut Chart
        wedges, texts, autotexts = ax.pie(
            severity_counts, 
            labels=None, # Labels will be in the legend on the mockup
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors, 
            wedgeprops=dict(width=0.35, edgecolor='w', linewidth=2),
            pctdistance=0.78
        )
        # Style percentages text
        for autotext in autotexts:
            autotext.set_fontsize(8.5)
            autotext.set_weight('bold')
            autotext.set_color('#ffffff')

        # Add total number of accidents in the center hole
        ax.text(
            0, 0, f"{total_accidents:,}\nTotal", 
            ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#0f172a'
        )
    else:
        ax.text(0, 0, 'No Data', ha='center', va='center', fontsize=12, color='#64748b')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'accidents_by_severity.png'), dpi=150)
    plt.close()

    # 3. ACCIDENTS BY WEATHER (Donut/Pie Chart)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not filtered_df.empty:
        weather_counts = filtered_df['weather'].value_counts()
        # Mockup colors: clear (blue #3b82f6), rain (cyan #06b6d4), cloudy/fog (slate #64748b)
        colors_map = {'clear': '#2563eb', 'rain': '#0ea5e9', 'fog': '#64748b'}
        colors = [colors_map.get(x, '#3b82f6') for x in weather_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            weather_counts, 
            labels=None, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=colors, 
            wedgeprops=dict(width=0.35, edgecolor='w', linewidth=2),
            pctdistance=0.78
        )
        for autotext in autotexts:
            autotext.set_fontsize(8.5)
            autotext.set_weight('bold')
            autotext.set_color('#ffffff')
            
        ax.text(
            0, 0, 'Weather', 
            ha='center', va='center', 
            fontsize=11, fontweight='bold', color='#475569'
        )
    else:
        ax.text(0, 0, 'No Data', ha='center', va='center', fontsize=12, color='#64748b')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'accidents_by_weather.png'), dpi=150)
    plt.close()

    # 4. ACCIDENTS BY HOUR (Line Chart)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not filtered_df.empty:
        hour_counts = filtered_df['hour'].value_counts().sort_index()
        hour_counts = hour_counts.reindex(range(24), fill_value=0)
        
        ax.plot(
            hour_counts.index, hour_counts.values, 
            marker='o', markersize=3, linewidth=2, color='#2563eb'
        )
        ax.fill_between(hour_counts.index, hour_counts.values, color='#3b82f6', alpha=0.1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.grid(True, linestyle='-', alpha=0.15, color='#cbd5e1')
    ax.set_xlabel('Hour of the Day', fontsize=8.5, color='#64748b')
    plt.xticks(range(0, 24, 4), fontsize=8.5, color='#64748b')
    plt.yticks(fontsize=8.5, color='#64748b')
    plt.xlim(0, 23)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'accidents_by_hour.png'), dpi=150)
    plt.close()

def generate_folium_map(filtered_df, maps_dir):
    """Generate and save an interactive Folium Map based on filtered coordinates."""
    map_file = os.path.join(maps_dir, 'hotspot_map.html')
    
    if filtered_df.empty:
        # Default center on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron", zoom_control=True)
        m.save(map_file)
        return
        
    center_lat = filtered_df["latitude"].mean()
    center_lon = filtered_df["longitude"].mean()
    
    # Adjust zoom level based on whether a specific city is selected
    is_single_city = filtered_df["city"].nunique() == 1
    zoom_start = 11 if is_single_city else 5
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, control_scale=True, tiles="CartoDB positron")
    
    # Create HeatMap layer
    heat_data = filtered_df[["latitude", "longitude"]].dropna().values.tolist()
    if heat_data:
        HeatMap(
            heat_data, 
            name="Hotspot Density Heatmap", 
            min_opacity=0.45, 
            radius=15, 
            blur=10
        ).add_to(m)
        
    # Create MarkerCluster layer (limit to 1000 markers for performance)
    marker_df = filtered_df.copy()
    if len(marker_df) > 1000:
        marker_df = marker_df.sample(1000, random_state=42)
        
    marker_cluster = MarkerCluster(name="Accident Details Cluster", show=False)
    
    color_map = {
        "fatal": "#ef4444",
        "major": "#f59e0b",
        "minor": "#10b981"
    }
    
    for idx, row in marker_df.iterrows():
        road_name = get_road_name(row['city'], row['road_type'])
        popup_html = f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 12px; width: 220px; line-height: 1.4; color: #1e293b;">
            <div style="font-weight: bold; border-bottom: 2px solid {color_map.get(row['accident_severity'], '#3b82f6')}; padding-bottom: 4px; margin-bottom: 8px; font-size: 13px;">
                🚨 Severity: {row['accident_severity'].upper()}
            </div>
            <b>Location:</b> {road_name}<br>
            <b>City/State:</b> {row['city']}, {row['state']}<br>
            <b>Date/Time:</b> {row['date']} {row['time']}<br>
            <b>Weather:</b> {row['weather']} ({row['visibility']})<br>
            <b>Casualties:</b> {row['casualties']}<br>
            <b>Risk Score:</b> <span style="color: #ef4444; font-weight: bold;">{row['risk_score']*10:.1f}/10</span>
        </div>
        """
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            popup=folium.Popup(popup_html, max_width=260),
            color=color_map.get(row["accident_severity"], "blue"),
            fill=True,
            fill_color=color_map.get(row["accident_severity"], "blue"),
            fill_opacity=0.7,
            weight=1
        ).add_to(marker_cluster)
        
    marker_cluster.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(map_file)

@app.route("/")
def landing():
    return render_template("landing.html")

def render_dashboard_view(active_tab):
    # Retrieve filters
    state_filter = request.args.get("state", "All")
    city_filter = request.args.get("city", "All")
    weather_filter = request.args.get("weather", "All")
    road_filter = request.args.get("road_type", "All")
    severity_filter = request.args.get("severity", "All")
    start_date = request.args.get("start_date", "2022-01-01")
    end_date = request.args.get("end_date", "2025-04-15")
    if state_filter != "All":
        # Get top 5 accident-prone cities in that selected state based on accident count
        state_df = df[df["state"] == state_filter]
        top_cities = state_df["city"].value_counts().head(5).index.tolist()
        city_options = sorted(top_cities)
        # Reset city filter if it's not in the top 5 cities of this state
        if city_filter != "All" and city_filter not in city_options:
            city_filter = "All"
    else:
        city_options = sorted(df["city"].dropna().unique())

    # Apply filters to DataFrame
    filtered_df = df.copy()
    if state_filter != "All" and not df.empty:
        filtered_df = filtered_df[filtered_df['state'] == state_filter]
    if city_filter != "All" and not df.empty:
        filtered_df = filtered_df[filtered_df['city'] == city_filter]
    if weather_filter != "All" and not df.empty:
        filtered_df = filtered_df[filtered_df['weather'] == weather_filter]
    if road_filter != "All" and not df.empty:
        filtered_df = filtered_df[filtered_df['road_type'] == road_filter]
    if severity_filter != "All" and not df.empty:
        filtered_df = filtered_df[filtered_df['accident_severity'] == severity_filter]
    if 'date' in filtered_df.columns and not df.empty:
        filtered_df = filtered_df[(filtered_df['date'] >= start_date) & (filtered_df['date'] <= end_date)]

    # Calculate statistics
    total_accidents = len(filtered_df)
    total_cities = filtered_df['city'].nunique() if not filtered_df.empty else 0
    total_states = filtered_df['state'].nunique() if not filtered_df.empty else 0

    # Ensure folders exist
    ensure_folders()

    # Generate Matplotlib graphs
    static_graphs_dir = os.path.join(app.root_path, 'static', 'graphs')
    generate_matplotlib_graphs(filtered_df, static_graphs_dir, total_accidents)

    # Generate Folium map
    static_maps_dir = os.path.join(app.root_path, 'static', 'maps')
    generate_folium_map(filtered_df, static_maps_dir)

    # Compute Grouped Locations for Table (Top 5 High Risk Locations)
    top_locations_list = []
    high_risk_loc_count = 0
    
    if not filtered_df.empty:
        loc_groups = filtered_df.groupby(['city', 'state', 'road_type']).agg(
            avg_risk=('risk_score', 'mean'),
            total_accidents=('accident_id', 'count'),
            fatal_accidents=('accident_severity', lambda x: (x == 'fatal').sum())
        ).reset_index()
        print("Average Risk Values:")
        print(loc_groups[['city', 'avg_risk']].sort_values(by='avg_risk', ascending=False).head(10))
        # High Risk Location count (where average risk >= 0.50)
        high_risk_loc_count = len(loc_groups[loc_groups['avg_risk'] >= 0.50])
        

        # Sort and take top 5
        top_loc_df = loc_groups.sort_values(by='avg_risk', ascending=False).head(5)
        
        for idx, row in top_loc_df.iterrows():
            top_locations_list.append({
                'rank': len(top_locations_list) + 1,
                'location': get_road_name(row['city'], row['road_type']),
                'city': row['city'],
                'state': row['state'],
                'risk_score': round(row['avg_risk'] * 10, 1),
                'accidents': int(row['total_accidents']),
                'fatal_accidents': int(row['fatal_accidents'])
            })
    else:
        # Fallback to general top dangerous roads when dataset is filtered to 0 matches
        pass

    # Compute Key Insights
    insights = {}
    if not filtered_df.empty:
        # Highest Accident City
        city_counts = filtered_df['city'].value_counts()
        insights['highest_city'] = city_counts.index[0] if not city_counts.empty else "N/A"
        insights['highest_city_accidents'] = city_counts.values[0] if not city_counts.empty else 0
        
        # Most Common Severity
        severity_counts = filtered_df['accident_severity'].value_counts()
        insights['common_severity'] = severity_counts.index[0].capitalize() if not severity_counts.empty else "N/A"
        insights['common_severity_pct'] = round((severity_counts.values[0] / total_accidents) * 100, 1) if not severity_counts.empty else 0
        
        # Highest Risk Location
        if top_locations_list:
            insights['highest_risk_road'] = top_locations_list[0]['location']
            insights['highest_risk_score'] = top_locations_list[0]['risk_score']
        else:
            insights['highest_risk_road'] = "N/A"
            insights['highest_risk_score'] = 0.0
            
        # Peak Hour
        hour_counts = filtered_df['hour'].value_counts()
        peak_hour = hour_counts.index[0] if not hour_counts.empty else 18
        peak_hour_cnt = hour_counts.values[0] if not hour_counts.empty else 0
        insights['peak_hour_pct'] = round((peak_hour_cnt / total_accidents) * 100, 1)
        h_start = peak_hour
        h_end = (peak_hour + 1) % 24
        period_start = "AM" if h_start < 12 else "PM"
        period_end = "AM" if h_end < 12 else "PM"
        h_start_fmt = h_start if h_start <= 12 else h_start - 12
        if h_start_fmt == 0: h_start_fmt = 12
        h_end_fmt = h_end if h_end <= 12 else h_end - 12
        if h_end_fmt == 0: h_end_fmt = 12
        insights['peak_hour_text'] = f"{h_start_fmt}:00 {period_start} - {h_end_fmt}:00 {period_end}"
        
        # Most Affected Weather
        weather_counts = filtered_df['weather'].value_counts()
        insights['common_weather'] = weather_counts.index[0].capitalize() if not weather_counts.empty else "N/A"
        insights['common_weather_pct'] = round((weather_counts.values[0] / total_accidents) * 100, 1) if not weather_counts.empty else 0
    else:
        # Fallback values to prevent N/A displaying when filtered to 0 matches
        insights = {
            'highest_city': "N/A",
            'highest_city_accidents': 0,
            'common_severity': "N/A",
            'common_severity_pct': 0.0,
            'highest_risk_road': "N/A",
            'highest_risk_score': 0.0,
            'peak_hour_text': "N/A",
            'peak_hour_pct': 0.0,
            'common_weather': "N/A",
            'common_weather_pct': 0.0
        }

    # Severity analysis percentages for KPIs and insights page
    fatal_cnt = len(filtered_df[filtered_df['accident_severity'] == 'fatal'])
    major_cnt = len(filtered_df[filtered_df['accident_severity'] == 'major'])
    minor_cnt = len(filtered_df[filtered_df['accident_severity'] == 'minor'])
    
    fatal_pct = round((fatal_cnt / total_accidents) * 100, 1) if total_accidents > 0 else 0
    major_pct = round((major_cnt / total_accidents) * 100, 1) if total_accidents > 0 else 0
    minor_pct = round((minor_cnt / total_accidents) * 100, 1) if total_accidents > 0 else 0

    # Timestamp to force reload of static images (cache busting)
    ts = int(time.time())

    return render_template(
        "index.html",
        total_accidents="{:,}".format(total_accidents),
        total_cities=total_cities,
        total_states=total_states,
        fatal_count="{:,}".format(fatal_cnt),
        fatal_pct=fatal_pct,
        major_pct=major_pct,
        minor_pct=minor_pct,
        high_risk_loc_count=high_risk_loc_count,
        
        states=UNIQUE_STATES,
        cities=city_options, # Pass full list so dropdown is always complete
        weathers=UNIQUE_WEATHER,
        road_types=UNIQUE_ROAD_TYPES,
        severities=UNIQUE_SEVERITIES,
        
        selected_state=state_filter,
        selected_city=city_filter,
        selected_weather=weather_filter,
        selected_road_type=road_filter,
        selected_severity=severity_filter,
        selected_start_date=start_date,
        selected_end_date=end_date,
        
        top_locations=top_locations_list,
        insights=insights,
        timestamp=ts,
        active_tab=active_tab
    )

@app.route("/dashboard")
def dashboard():
    active_tab = request.args.get("tab", "dashboard")

    if active_tab not in ["dashboard", "insights"]:
        active_tab = "dashboard"

    return render_dashboard_view(active_tab)
@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    if request.method == "POST":
        try:
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
                
            weather = data.get("weather", "clear").lower()
            visibility = data.get("visibility", "high").lower()
            traffic_density = data.get("traffic_density", "low").lower()
            try:
                temperature = float(data.get("temperature", 25))
            except ValueError:
                temperature = 25.0
                
            prediction = predict_severity(weather, visibility, traffic_density, temperature)
            
            # Calculate risk score percentage based on model factors
            w_score = 2 if weather == "fog" else (1 if weather == "rain" else 0)
            v_score = 2 if visibility == "low" else (1 if visibility == "medium" else 0)
            t_score = 2 if traffic_density == "high" else (1 if traffic_density == "medium" else 0)
            temp_score = 1 if temperature >= 32 else 0
            
            total_score = w_score + v_score + t_score + temp_score
            risk_percent = int((total_score / 7) * 100)
            
            return jsonify({
                'status': 'success',
                'prediction': prediction,
                'risk_percent': risk_percent,
                'score': total_score,
                'breakdown': {
                    'weather': w_score,
                    'visibility': v_score,
                    'traffic_density': t_score,
                    'temperature': temp_score
                }
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
    else:
        return render_dashboard_view("predictor")
@app.route("/insights")
def insights():
    return render_dashboard_view("insights")
@app.route("/map")
@app.route("/analytics")
@app.route("/weather")
@app.route("/road")
@app.route("/city")
def redirect_to_dashboard():
    from flask import redirect
    return redirect("/dashboard", code=302)

if __name__ == "__main__":
    ensure_folders()
    app.run(debug=True)