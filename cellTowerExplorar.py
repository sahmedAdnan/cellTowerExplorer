import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Output, Input, no_update
from flask_caching import Cache
import dash_bootstrap_components as dbc
import warnings

# Suppress runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ----------------------------------------
# Logging configuration
# ----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------
# Configuration / Parameters
# ----------------------------------------
CELL_TOWER_FILE_PATH = 'sampled_dataset.csv'
POPULATION_FILE_PATH = r"C:\Users\Adnan\Downloads\population.csv"  # Ensure this file exists in your directory
MAX_POINTS = 50000  # Adjust as needed for performance

# Specify actual column names in population_data.csv
POPULATION_COUNTRY_COLUMN = 'Country Name'       
POPULATION_VALUE_COLUMN = 'Population'           
POPULATION_YEAR_COLUMN = 'Year'                  

# ----------------------------------------
# Data Loading and Preprocessing
# ----------------------------------------
logger.info("Loading cell tower dataset...")
try:
    df = pd.read_csv(CELL_TOWER_FILE_PATH)
    logger.info(f"Cell tower dataset loaded successfully with {len(df)} records.")
except FileNotFoundError:
    logger.error(f"File {CELL_TOWER_FILE_PATH} not found.")
    df = pd.DataFrame()
except pd.errors.ParserError as e:
    logger.error(f"Error parsing {CELL_TOWER_FILE_PATH}: {e}")
    df = pd.DataFrame()
except Exception as e:
    logger.exception("Unexpected error while loading cell tower data:")
    df = pd.DataFrame()

if not df.empty:
    # Drop rows with missing latitude or longitude
    df = df.dropna(subset=['lat', 'lon'])

    # Convert 'created' and 'updated' timestamps to datetime
    df['created'] = pd.to_datetime(df['created'], unit='s', errors='coerce')
    df['updated'] = pd.to_datetime(df['updated'], unit='s', errors='coerce')
    df = df.dropna(subset=['created', 'updated'])

    # Fill missing 'Operator' values
    df['Operator'] = df['Operator'].fillna('Unknown')

    # Convert 'range' to numeric
    df['range'] = pd.to_numeric(df['range'], errors='coerce')

    logger.info("Cell tower data preprocessing completed.")
else:
    logger.warning("Empty DataFrame. Please check the CSV file.")

# ----------------------------------------
# Population Data Loading and Preprocessing
# ----------------------------------------
logger.info("Loading population dataset...")
try:
    population_df = pd.read_csv(POPULATION_FILE_PATH)
    logger.info(f"Population dataset loaded successfully with {len(population_df)} records.")
except FileNotFoundError:
    logger.error(f"File {POPULATION_FILE_PATH} not found.")
    population_df = pd.DataFrame()
except pd.errors.ParserError as e:
    logger.error(f"Error parsing {POPULATION_FILE_PATH}: {e}")
    population_df = pd.DataFrame()
except Exception as e:
    logger.exception("Unexpected error while loading population data:")
    population_df = pd.DataFrame()

if not population_df.empty:
    # Check if the specified columns exist
    required_columns = [POPULATION_COUNTRY_COLUMN, POPULATION_VALUE_COLUMN, POPULATION_YEAR_COLUMN]
    if not all(col in population_df.columns for col in required_columns):
        logger.error(
            f"Population data must contain '{POPULATION_COUNTRY_COLUMN}', '{POPULATION_VALUE_COLUMN}', and '{POPULATION_YEAR_COLUMN}' columns.")
    else:
        # Select the latest population data for each country
        population_df_sorted = population_df.sort_values(by=[POPULATION_YEAR_COLUMN], ascending=False)
        latest_population_df = population_df_sorted.drop_duplicates(subset=[POPULATION_COUNTRY_COLUMN], keep='first')

        # Rename columns for consistency
        latest_population_df = latest_population_df.rename(columns={
            POPULATION_COUNTRY_COLUMN: 'Country',
            POPULATION_VALUE_COLUMN: 'Population'
        })

        # Standardize country names to match cell tower data
        latest_population_df['Country'] = latest_population_df['Country'].str.strip().str.title()

        # Handle mismatched country names using a mapping dictionary
        country_mapping = {
            'United States': 'United States of America',
            'Turkiye': 'Turkey'
            # Add more mappings as needed
        }

        latest_population_df['Country'] = latest_population_df['Country'].replace(country_mapping)

        # Merge with cell tower data
        df = pd.merge(df, latest_population_df[['Country', 'Population']], on='Country', how='left')

        # Handle missing population data
        df['Population'] = df['Population'].fillna(0)

        logger.info("Population data merged successfully with cell tower data.")
else:
    logger.warning("Population DataFrame is empty. Skipping population merge.")

# ----------------------------------------
# Initialize the Dash App
# ----------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 300})


# ----------------------------------------
# Utility Functions
# ----------------------------------------
def downsample_data(data, max_points=MAX_POINTS):
    """Downsample the data if it exceeds a certain number of points."""
    if len(data) > max_points:
        return data.sample(n=max_points, random_state=42)
    return data


@cache.memoize(timeout=300)
def get_filtered_data(selected_countries, selected_operators, selected_radios):
    """Filter the global DataFrame based on user selections."""
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
    if selected_operators:
        filtered_df = filtered_df[filtered_df['Operator'].isin(selected_operators)]
    if selected_radios:
        filtered_df = filtered_df[filtered_df['radio'].isin(selected_radios)]
    filtered_df = filtered_df.dropna(subset=['created', 'range'])
    return filtered_df


# ----------------------------------------
# Layout
# ----------------------------------------
app.layout = dbc.Container(
    fluid=True,
    children=[
        # Navbar
        dbc.NavbarSimple(
            children=[],
            brand=html.Strong("Cell Tower Explorer"),
            brand_href="#",
            color="light",
            dark=False,
            fluid=True,
        ),
        html.Br(),
        # Main Content
        dbc.Row(
            [
                # Sidebar with Filters
                dbc.Col(
                    [
                        html.H4("Filter Data"),  # Font size and color set via global CSS
                        html.Hr(),
                        dbc.Form(
                            [
                                dbc.Row([
                                    dbc.Label("Select Country:", className="me-2"),
                                    dcc.Dropdown(
                                        id='country_dropdown',
                                        options=[{'label': c, 'value': c} for c in
                                                 sorted(df['Country'].dropna().unique())],
                                        multi=True,
                                        placeholder="Select countries",
                                        style={'margin-bottom': '10px', 'width': '100%'}
                                    ),
                                ]),
                                dbc.Row([
                                    dbc.Label("Select Operator:", className="me-2"),
                                    dcc.Dropdown(
                                        id='operator_dropdown',
                                        options=[{'label': o, 'value': o} for o in
                                                 sorted(df['Operator'].dropna().unique())],
                                        multi=True,
                                        placeholder="Select operators",
                                        style={'margin-bottom': '10px', 'width': '100%'}
                                    ),
                                ]),
                                dbc.Row([
                                    dbc.Label("Select Radio Type:", className="me-2"),
                                    dcc.Dropdown(
                                        id='radio_dropdown',
                                        options=[{'label': r, 'value': r} for r in
                                                 sorted(df['radio'].dropna().unique())],
                                        multi=True,
                                        placeholder="Select radio types",
                                        style={'margin-bottom': '10px', 'width': '100%'}
                                    ),
                                ]),
                                dbc.Button("Reset Filters", id="reset_filters", color="secondary", className="mt-2"),
                            ],
                            className="g-2",
                        ),
                        html.Br(),
                        # Single Section for Displaying Counts
                        dbc.Row([
                            dbc.Col(
                                dbc.Alert(id='record_count', color='info', is_open=True),
                                width=12
                            ),
                        ]),
                    ],
                    md=4,
                    style={
                        'backgroundColor': '#e9ecef',  # hoverColor
                        'padding': '15px',
                        'border-radius': '5px'
                    }
                ),
                # Main Visualization Area
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Map Visualizations", tab_id="tab-maps"),
                                dbc.Tab(label="Data Insights", tab_id="tab-insights"),
                                dbc.Tab(label="Trends & Analysis", tab_id="tab-trends"),
                            ],
                            id="tabs",
                            active_tab="tab-maps",
                            className="mt-2",
                            # Removed active_tab_style and inactive_tab_style
                        ),
                        html.Div(id="tab-content", className="p-4"),
                    ],
                    md=8,
                ),
            ],
            className="mt-2",
        ),
    ],
    style={'backgroundColor': '#f8f9fa', 'fontFamily': 'Arial'},
)

# ----------------------------------------
# Figure Generation Functions
# ----------------------------------------
def generate_choropleth_map(filtered_df):
    """Generates a choropleth map showing the number of cell towers per 100,000 people by country."""
    if filtered_df.empty or 'Population' not in filtered_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Cell Towers per 100,000 People by Country",
                font=dict(size=16, family="Arial"),
                x=0.5  # Center align the title
            ),
            template='plotly_white',
            annotations=[
                dict(
                    text="No data available to generate the choropleth map.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                )
            ]
        )
        return fig

    # Ensure 'Country' is standardized
    filtered_df['Country'] = filtered_df['Country'].str.strip().str.title()

    # Calculate Tower Count per Country
    country_tower_counts = filtered_df.groupby(['Country']).size().reset_index(name='Tower Count')

    # Merge with Population Data
    if 'Population' in filtered_df.columns:
        country_tower_counts = pd.merge(
            country_tower_counts,
            filtered_df[['Country', 'Population']].drop_duplicates(),
            on='Country',
            how='left'
        )
    else:
        logger.warning("'Population' column not found in filtered data. Proceeding without population data.")
        country_tower_counts['Population'] = 0

    # Handle Missing Population Data
    country_tower_counts['Population'] = country_tower_counts['Population'].fillna(0)

    # Calculate towers per 100,000 people
    country_tower_counts['Towers_per_100k'] = (
        (country_tower_counts['Tower Count'] / country_tower_counts['Population']) * 100000
    )

    # Replace infinite values and NaNs resulting from division by zero
    country_tower_counts.replace([np.inf, -np.inf], np.nan, inplace=True)
    country_tower_counts['Towers_per_100k'] = country_tower_counts['Towers_per_100k'].fillna(0)

    # Create Choropleth Map
    fig = px.choropleth(
        country_tower_counts,
        locations='Country',
        locationmode='country names',
        color='Towers_per_100k',
        hover_name='Country',
        color_continuous_scale=px.colors.sequential.Blues,
        title="Cell Towers per 100,000 People by Country",
        labels={'Towers_per_100k': 'Cell Towers per 100k People'},
        template='plotly_white'
    )

    # Update Geo Layout to Fit Bounds Based on Selected Data
    fig.update_geos(
        fitbounds="locations",  # Automatically adjust the map to fit the selected locations
        visible=False  # Hide the default geographic frame
    )

    # Update Additional Layout Properties
    fig.update_layout(
        title=dict(
            text="Cell Towers per 100,000 People by Country",
            font=dict(size=16, family="Arial"),
            x=0.5  
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text="Cell Towers<br>per 100k People",
                font=dict(size=14, family="Arial")
            ),
        ),
    )

    return fig



def generate_scatter_map_box(filtered_df):
    """Generates a scatter map box showing individual cell tower locations."""
    if filtered_df.empty or 'lat' not in filtered_df or 'lon' not in filtered_df:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Geospatial Distribution by Radio Type and Operator",
                font=dict(size=20, family="Arial", color="blue"),
                x=0.5  # Center align the title
            ),
            template='plotly_white',
            annotations=[
                dict(
                    text="No data available to generate the map.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            ]
        )
        return fig

    # Ensure valid coordinates
    filtered_df = filtered_df.dropna(subset=['lat', 'lon'])
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Geospatial Distribution by Radio Type and Operator",
                font=dict(size=14, family="Arial", color="black"),
                x=0.5
            ),
            template='plotly_white',
            annotations=[
                dict(
                    text="No valid latitude and longitude data available.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            ]
        )
        return fig

    # Dynamically calculate the center and zoom level
    map_center = {"lat": filtered_df['lat'].mean(), "lon": filtered_df['lon'].mean()}

    fig = px.scatter_mapbox(
        filtered_df,
        lat='lat',
        lon='lon',
        color='radio',
        size='range',
        hover_name='Operator',
        hover_data={
            'range': True,
            'lat': ':.4f',  # Limit decimal points for lat/lon
            'lon': ':.4f',
            'radio': True,
            'cell': True,
            'area': True,
            'unit': True,
            'changeable': True,
            'created': True,
            'updated': True
        },
        zoom=3,
        center=map_center,
        height=500,
        title="Geospatial Distribution by Radio Type and Operator",
        mapbox_style="carto-positron",  
        color_discrete_sequence=px.colors.qualitative.G10,  
        labels={
            'range': 'Signal Range (m)',
            'lat': 'Latitude',
            'lon': 'Longitude',
            'radio': 'Radio Type',
            'cell': 'Cell ID',
            'created': 'Created',
            'updated': 'Updated',
            'area': 'Area',
            'unit': 'Unit',
            'changeable': 'Changeable'
        }
    )

    fig.update_traces(marker=dict(opacity=0.7))  # Slight transparency for better visibility

    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title=dict(
            text="Geospatial Distribution by Radio Type and Operator",
            font=dict(size=16, family="Arial"),
            x=0.5  # Center align the title
        ),
    )
    return fig



def generate_radio_distribution_bar_chart(filtered_df):
    """Generates a bar chart showing the distribution of radio types."""
    if 'radio' not in filtered_df.columns or filtered_df['radio'].isnull().all():
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text="Distribution of Radio Types",
                font=dict(size=16, family="Arial", color="black"),
                x=0.5  # Center-align the title
            ),
            template='plotly_white',
            annotations=[
                dict(
                    text="No radio data available.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="red")
                )
            ]
        )
        return fig

    # Calculate the distribution of radio types
    radio_counts = filtered_df['radio'].value_counts().reset_index()
    radio_counts.columns = ['Radio Type', 'Count']

    # Create bar chart
    fig = px.bar(
        radio_counts,
        x='Radio Type',
        y='Count',
        title="Distribution of Radio Types",
        template='plotly_white',
        color='Radio Type',
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    # Update layout properties
    fig.update_layout(
        title=dict(
            text="Distribution of Radio Types",
            font=dict(size=16, family="Arial"),
            x=0.5  # Center-align the title
        ),
        margin=dict(l=40, r=40, t=60, b=40)  # Adjust margins for better aesthetics
    )

    return fig



def generate_bar_chart(filtered_df):
    """Generates a horizontal bar chart showing the top 10 countries by number of cell towers."""
    if 'Country' not in filtered_df.columns or filtered_df['Country'].isnull().all():
        fig = go.Figure()
        fig.update_layout(
            title="Number of Cell Towers per Country",
            template='plotly_white',
            annotations=[dict(text="No country data available.", xref="paper", yref="paper", showarrow=False)]
        )
        return fig
    country_counts = filtered_df['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Tower Count']
    # Select top 10 countries
    top_countries = country_counts.head(10).sort_values(by='Tower Count', ascending=True)  # For horizontal bar chart
    fig = px.bar(
        top_countries,
        x='Tower Count',
        y='Country',
        orientation='h',
        title="Top 10 Countries by Number of Cell Towers",
        template='plotly_white',
    )
    fig.update_layout(
        title=dict(
            font=dict(size=16, family="Arial"),
            x=0.5
        ),
        yaxis={'categoryorder': 'total ascending'},  # Highest count at top
        xaxis_title="Number of Cell Towers",
        yaxis_title="Country",
        margin=dict(l=0, r=10, t=50, b=50),  # Adjust margins for better spacing
    )
    return fig


def generate_treemap_chart(filtered_df):
    """Generates a treemap showing the hierarchical distribution of towers."""
    required_cols = ['Country', 'Operator', 'radio']
    if not all(col in filtered_df.columns for col in required_cols):
        fig = go.Figure()
        fig.update_layout(
            title="Hierarchical Distribution of Towers",
            template='plotly_white',
            annotations=[dict(text="Data columns missing for treemap.", xref="paper", yref="paper", showarrow=False)]
        )
        return fig
    filtered_df['Tower Count'] = 1
    treemap_data = filtered_df.groupby(['Country', 'Operator', 'radio'], observed=True).size().reset_index(
        name='Tower Count')
    fig = px.treemap(
        treemap_data,
        path=['Country', 'Operator', 'radio'],
        values='Tower Count',
        title="Hierarchical Distribution of Towers",
        template='plotly_white',
        color='radio',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data={'Tower Count': True}
    )
    fig.update_layout(
        title=dict(
            text="Hierarchical Distribution of Towers",
            font=dict(size=16, family="Arial"),
            x=0.5  # Center-align the title
        ),
        margin=dict(l=40, r=40, t=60, b=40),  # Adjust margins for better spacing
    )
    return fig


def generate_cumulative_growth_trend(filtered_df):
    """Generates an area chart showing the cumulative growth of cell towers over time."""
    if filtered_df.empty or 'created' not in filtered_df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="Cumulative Tower Growth Over Time",
            template='plotly_white',
            annotations=[
                dict(text="No data available to generate growth trend.", xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # Ensure 'created' is datetime
    filtered_df['created'] = pd.to_datetime(filtered_df['created'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['created'])
    if filtered_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Cumulative Tower Growth Over Time",
            template='plotly_white',
            annotations=[dict(text="No valid creation dates available.", xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # Sort by 'created' date
    filtered_df = filtered_df.sort_values('created')

    # Calculate cumulative count
    growth_trend_df = filtered_df.resample('ME', on='created').size().cumsum().reset_index(
        name='Cumulative Tower Count')
    if growth_trend_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Cumulative Tower Growth Over Time",
            template='plotly_white',
            annotations=[
                dict(text="No data available to generate growth trend.", xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    fig = px.area(
        growth_trend_df,
        x='created',
        y='Cumulative Tower Count',
        title="Cumulative Tower Growth Over Time",
        template='plotly_white',
        color_discrete_sequence=['#58508d']
    )
    fig.update_layout(
        xaxis_rangeslider_visible=True,
        hovermode='x unified',
        title=dict(
            font=dict(size=16, family="Arial"),
            x=0.5  # Center-align the title
        ),
        xaxis_title="Date",
        yaxis_title="Cumulative Number of Towers",
        height=500,  # Adjusted height for better aspect ratio
    )
    return fig


def generate_operator_growth_trend(filtered_df):
    """
    Generates a multi-line chart showing cumulative tower growth for the top 10 operators over time.
    """
    if filtered_df.empty or 'created' not in filtered_df.columns or 'Operator' not in filtered_df.columns:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="Operator-Specific Tower Growth Over Time",
            template='plotly_white',
            annotations=[dict(text="Insufficient data to generate operator growth trend.",
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # Ensure 'created' is datetime
    filtered_df['created'] = pd.to_datetime(filtered_df['created'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['created', 'Operator'])

    # Calculate total towers per operator
    operator_counts = filtered_df['Operator'].value_counts().nlargest(10).index
    top_10_df = filtered_df[filtered_df['Operator'].isin(operator_counts)]

    if top_10_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Operator-Specific Tower Growth Over Time",
            template='plotly_white',
            annotations=[dict(text="No data available for the top 10 operators.",
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    # Group by Operator and resample
    growth_trend_df = top_10_df.groupby('Operator').resample('ME', on='created').size().groupby(
        'Operator').cumsum().reset_index(name='Cumulative Tower Count')

    if growth_trend_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Operator-Specific Tower Growth Over Time",
            template='plotly_white',
            annotations=[dict(text="No data available to generate operator growth trend.",
                              xref="paper", yref="paper", showarrow=False)]
        )
        return fig

    fig = px.line(
        growth_trend_df,
        x='created',
        y='Cumulative Tower Count',
        color='Operator',
        title="Operator-Specific Tower Growth Over Time (Top 10 Operators)",
        template='plotly_white',
        labels={'created': 'Date', 'Cumulative Tower Count': 'Number of Towers'},
        color_discrete_sequence=px.colors.qualitative.Dark24  # Enhanced color palette for distinction
    )

    # Adjust legend to be outside the plot area
    fig.update_layout(
        legend=dict(
            title="Operator",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),  
        ),
        margin=dict(l=50, r=150, t=100, b=50),  # Increased right margin to accommodate legend
        height=600,  # Adjusted height for better aspect ratio
        title=dict(
            font=dict(size=16, family="Arial"),
            x=0.5  # Center-align the title
        ),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    return fig
# ----------------------------------------
# Callbacks
# ----------------------------------------
@app.callback(
    Output("tab-content", "children"),
    [
        Input("tabs", "active_tab"),
        Input('country_dropdown', 'value'),
        Input('operator_dropdown', 'value'),
        Input('radio_dropdown', 'value')
    ]
)
def render_tab_content(active_tab, selected_countries, selected_operators, selected_radios):
    """
    Render tab content based on current tab and selected filters.
    """
    try:
        logger.info(f"Active tab: {active_tab}")
        logger.info(f"Selected countries: {selected_countries}")
        logger.info(f"Selected operators: {selected_operators}")
        logger.info(f"Selected radios: {selected_radios}")

        filtered_df = get_filtered_data(selected_countries, selected_operators, selected_radios)
        filtered_df = downsample_data(filtered_df)

        logger.info(f"Filtered data contains {len(filtered_df)} records.")

        if filtered_df.empty:
            return dbc.Alert("No data available for the selected filters.", color="warning")

        if active_tab == "tab-maps":
            return [
                dcc.Loading(
                    id="loading-maps",
                    type="circle",
                    children=[
                        dbc.Row(
                            [dbc.Col(dcc.Graph(figure=generate_choropleth_map(filtered_df)), width=12)],
                            className="mb-4",
                        ),
                        dbc.Row(
                            [dbc.Col(dcc.Graph(figure=generate_scatter_map_box(filtered_df)), width=12)],
                            className="mb-4",
                        ),
                    ],
                )
            ]
        elif active_tab == "tab-insights":
            return [
                dcc.Loading(
                    id="loading-insights",
                    type="circle",
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(figure=generate_radio_distribution_bar_chart(filtered_df)), width=6),
                                dbc.Col(dcc.Graph(figure=generate_bar_chart(filtered_df)), width=6),
                            ],
                            className="mb-4",
                        ),
                        dbc.Row(
                            [dbc.Col(dcc.Graph(figure=generate_treemap_chart(filtered_df)), width=12)],
                            className="mb-4",
                        ),
                    ],
                )
            ]
        elif active_tab == "tab-trends":
            return [
                dcc.Loading(
                    id="loading-trends",
                    type="circle",
                    children=[
                        # Operator-Specific Growth Trend (Upper Section)
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(figure=generate_operator_growth_trend(filtered_df)), width=12),
                            ],
                            className="mb-4",
                        ),
                        # Cumulative Growth Trend (Lower Section)
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(figure=generate_cumulative_growth_trend(filtered_df)), width=12),
                            ],
                            className="mb-4",
                        ),
                    ],
                )
            ]
        else:
            return dbc.Alert("This tab is not available.", color="danger")
    except Exception as e:
        logger.exception("Error in render_tab_content:")
        return dbc.Alert("An error occurred while generating the graphs.", color="danger")


@app.callback(
    Output('record_count', 'children'),
    [Input('country_dropdown', 'value'),
     Input('operator_dropdown', 'value'),
     Input('radio_dropdown', 'value')]
)
def update_record_count(selected_countries, selected_operators, selected_radios):
    """
    Update the single record count section based on selected filters.
    """
    total = len(df)
    if df.empty:
        filtered = 0
    else:
        filtered_df = get_filtered_data(selected_countries, selected_operators, selected_radios)
        filtered = len(filtered_df)
    record_text = f"Total Records: {total} | Filtered Records: {filtered}"
    return record_text


@app.callback(
    [Output('operator_dropdown', 'options'),
     Output('radio_dropdown', 'options')],
    [Input('country_dropdown', 'value'),
     Input('operator_dropdown', 'value')]
)
def update_dropdown_options(selected_countries, selected_operators):
    filtered = df
    if selected_countries:
        filtered = filtered[filtered['Country'].isin(selected_countries)]
    if selected_operators:
        filtered = filtered[filtered['Operator'].isin(selected_operators)]

    operators = sorted(filtered['Operator'].dropna().unique())
    radios = sorted(filtered['radio'].dropna().unique())

    operator_options = [{'label': o, 'value': o} for o in operators]
    radio_options = [{'label': r, 'value': r} for r in radios]

    return operator_options, radio_options


@app.callback(
    [Output('country_dropdown', 'value'),
     Output('operator_dropdown', 'value'),
     Output('radio_dropdown', 'value')],
    [Input('reset_filters', 'n_clicks')],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if n_clicks:
        logger.info("Reset Filters button clicked. Resetting dropdowns.")
        return [], [], []
    return no_update, no_update, no_update


# ----------------------------------------
# Run Server
# ----------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
