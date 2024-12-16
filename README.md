# Cell Tower Visualization Dashboard

![Project Logo](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/Overview.png)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)

## Introduction

The **Cell Tower Visualization Dashboard** is an interactive web application designed to visualize and analyze the distribution of cell towers in relation to population metrics across different countries. Built using [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/), this dashboard empowers telecom operators, policymakers, urban planners, and researchers to explore cellular infrastructure data, identify coverage gaps, and make informed decisions for network optimization and expansion.

By integrating cell tower data with population statistics, the dashboard provides a comprehensive view of how telecom infrastructure aligns with demographic trends, highlighting areas that may require further investment or technological upgrades.

## Features

- **Interactive Filtering:** Easily filter cell tower data by country, operator, and radio type to focus on specific regions or technologies.
- **Geospatial Visualizations:** 
  - **Choropleth Map:** Displays the density of cell towers per 100,000 people by country.
  - **Scatter Map:** Shows individual cell tower locations with color-coded radio types and adjustable marker sizes based on signal range.
- **Data Insights:**
  - **Radio Distribution Bar Chart:** Compares the distribution of different radio technologies across selected regions.
  - **Top Countries Bar Chart:** Highlights the top countries with the highest number of cell towers.
  - **Treemap:** Visualizes the hierarchical distribution of towers by country, operator, and radio type.
- **Trends & Analysis:**
  - **Cumulative Growth Trend:** Illustrates the growth of cell towers over time.
  - **Operator-Specific Growth Trend:** Tracks the expansion of cell towers for the top operators over time.
- **Performance Optimization:** Implements data downsampling and caching to ensure smooth and responsive interactions even with large datasets.

## Data Sources

- **Cell Tower Data:** [OpenCellID](https://www.opencellid.org/) - A collaborative project aimed at creating a global database of cell towers.
- **Population Data:** [Office for National Statistics (ONS)](https://www.ons.gov.uk/) - Provides population estimates by country and year.
- **Additional Reference:** [Plotly Dash World Cell Towers](https://github.com/plotly/dash-world-cell-towers) - Inspiring examples of interactive cell tower visualizations.

## Installation

### Prerequisites

- **Python 3.7 or higher**  
  Ensure you have Python installed on your system. You can download it from the [official website](https://www.python.org/downloads/).

## Usage
   Once the application is running, you can:

- **Filter Data:**
Use the dropdown menus on the sidebar to filter cell towers by country, operator, and radio type.

- **Explore Maps:**
Navigate to the Map Visualizations tab to view choropleth and scatter maps illustrating tower density and distribution.

- **Analyze Insights:**
Switch to the Data Insights tab to compare radio technology distributions, identify top countries, and explore hierarchical relationships through treemaps.

- **Examine Trends:**
Visit the Trends & Analysis tab to observe growth patterns over time, both cumulatively and operator-specific.

- **View Record Counts:**
The dashboard displays total and filtered record counts, providing context to your selections.

## screenshots
Choropleth map showing cell tower density per 100,000 people by country.
![Choropleth map](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_1.png)

Scatter map displaying individual cell tower locations with color-coded radio types.
![Scatter map](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_2.png)

Bar chart illustrating radio distribution.
![charts](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_3.png)



Bar charts illustrating radio distribution and top countries by cell tower count.
![Bar](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_4.png)

Treemap representing hierarchical distribution of cell towers.
![Tree map](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_5.png)

Growth trend for operator
![operator](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_6.png)

Area and line charts showcasing cumulative and operator-specific growth trends over time.
![growth trend](https://github.com/sahmedAdnan/cellTowerExplorer/blob/main/pic/BD_7.png)

## Presentation
![](https://drive.google.com/file/d/1tgNQUW9lVwP-5DJfPhgNvZlFIiP8dIOu/view?usp=sharing)
