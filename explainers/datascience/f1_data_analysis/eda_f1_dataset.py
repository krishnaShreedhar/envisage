import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')


class F1DataAnalyzer:
    """
    Comprehensive Formula 1 Data Analysis and Visualization Tool

    This class provides methods for:
    - Data loading and preprocessing
    - Exploratory data analysis
    - Statistical analysis
    - Machine learning insights
    - Interactive visualizations
    """

    def __init__(self, data_path=None):
        """
        Initialize the F1 Data Analyzer

        Args:
            data_path (str): Path to the F1 dataset directory
        """
        self.data_path = data_path
        self.circuits = None
        self.constructors = None
        self.drivers = None
        self.races = None
        self.results = None
        self.constructor_results = None
        self.driver_standings = None
        self.constructor_standings = None
        self.lap_times = None
        self.qualifying = None

        # Color palette for visualizations
        self.colors = px.colors.qualitative.Set3

        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def load_data(self):
        """
        Load all F1 dataset files

        Expected CSV files:
        - circuits.csv, constructors.csv, drivers.csv, races.csv
        - results.csv, constructor_results.csv, driver_standings.csv
        - constructor_standings.csv, lap_times.csv, qualifying.csv
        """
        try:
            if self.data_path:
                self.circuits = pd.read_csv(f"{self.data_path}/circuits.csv")
                self.constructors = pd.read_csv(f"{self.data_path}/constructors.csv")
                self.drivers = pd.read_csv(f"{self.data_path}/drivers.csv")
                self.races = pd.read_csv(f"{self.data_path}/races.csv")
                self.results = pd.read_csv(f"{self.data_path}/results.csv")
                self.constructor_results = pd.read_csv(f"{self.data_path}/constructor_results.csv")
                self.driver_standings = pd.read_csv(f"{self.data_path}/driver_standings.csv")
                self.constructor_standings = pd.read_csv(f"{self.data_path}/constructor_standings.csv")

                # Optional files that might not exist
                try:
                    self.lap_times = pd.read_csv(f"{self.data_path}/lap_times.csv")
                    self.qualifying = pd.read_csv(f"{self.data_path}/qualifying.csv")
                except FileNotFoundError:
                    print("Optional files (lap_times.csv, qualifying.csv) not found. Continuing without them.")

                print("✅ Data loaded successfully!")
                self._preprocess_data()
            else:
                print("⚠️  No data path provided. Generating sample data for demonstration.")
                self._generate_sample_data()

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("Generating sample data for demonstration.")
            self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate sample F1 data for demonstration purposes"""
        np.random.seed(42)

        # Sample circuits
        circuits_data = {
            'circuitId': range(1, 21),
            'circuitRef': [f'circuit_{i}' for i in range(1, 21)],
            'name': [f'Circuit {i}' for i in range(1, 21)],
            'location': [f'Location {i}' for i in range(1, 21)],
            'country': ['Monaco', 'UK', 'Italy', 'Germany', 'Brazil', 'Australia', 'Japan', 'USA', 'Canada', 'Spain',
                        'France', 'Belgium', 'Hungary', 'Netherlands', 'Austria', 'Singapore', 'Russia', 'Mexico',
                        'UAE', 'Turkey'],
            'lat': np.random.uniform(-60, 60, 20),
            'lng': np.random.uniform(-180, 180, 20),
            'alt': np.random.uniform(0, 1000, 20)
        }
        self.circuits = pd.DataFrame(circuits_data)

        # Sample constructors
        constructors_data = {
            'constructorId': range(1, 16),
            'constructorRef': ['ferrari', 'mercedes', 'red_bull', 'mclaren', 'williams', 'alpine', 'alpha_tauri',
                               'aston_martin', 'alfa_romeo', 'haas', 'lotus', 'force_india', 'toro_rosso', 'sauber',
                               'manor'],
            'name': ['Ferrari', 'Mercedes', 'Red Bull Racing', 'McLaren', 'Williams', 'Alpine', 'AlphaTauri',
                     'Aston Martin', 'Alfa Romeo', 'Haas', 'Lotus', 'Force India', 'Toro Rosso', 'Sauber', 'Manor'],
            'nationality': ['Italian', 'German', 'Austrian', 'British', 'British', 'French', 'Italian',
                            'British', 'Swiss', 'American', 'British', 'Indian', 'Italian', 'Swiss', 'British']
        }
        self.constructors = pd.DataFrame(constructors_data)

        # Sample drivers
        drivers_data = {
            'driverId': range(1, 51),
            'driverRef': [f'driver_{i}' for i in range(1, 51)],
            'forename': [f'Driver{i}' for i in range(1, 51)],
            'surname': [f'Surname{i}' for i in range(1, 51)],
            'dob': pd.date_range('1980-01-01', periods=50, freq='30D'),
            'nationality': np.random.choice(
                ['British', 'German', 'Italian', 'French', 'Spanish', 'Brazilian', 'Finnish', 'Dutch'], 50)
        }
        self.drivers = pd.DataFrame(drivers_data)

        # Sample races
        years = range(1950, 2021)
        races_data = []
        race_id = 1

        for year in years:
            for round_num in range(1, np.random.randint(15, 23)):  # 15-22 races per year
                races_data.append({
                    'raceId': race_id,
                    'year': year,
                    'round': round_num,
                    'circuitId': np.random.randint(1, 21),
                    'name': f'{year} Race {round_num}',
                    'date': f'{year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
                    'time': f'{np.random.randint(10, 18):02d}:00:00'
                })
                race_id += 1

        self.races = pd.DataFrame(races_data)

        # Sample results
        results_data = []
        result_id = 1

        for _, race in self.races.iterrows():
            drivers_in_race = np.random.choice(self.drivers['driverId'],
                                               size=np.random.randint(18, 25), replace=False)

            for pos, driver_id in enumerate(drivers_in_race, 1):
                constructor_id = np.random.randint(1, 16)
                points = max(0, 26 - pos) if pos <= 10 else 0

                results_data.append({
                    'resultId': result_id,
                    'raceId': race['raceId'],
                    'driverId': driver_id,
                    'constructorId': constructor_id,
                    'position': pos if pos <= 20 else None,
                    'points': points,
                    'laps': np.random.randint(50, 71),
                    'time': f'{np.random.randint(90, 120)}:{np.random.randint(0, 60):02d}.{np.random.randint(0, 999):03d}',
                    'milliseconds': np.random.randint(5400000, 7200000),
                    'fastestLap': np.random.randint(1, 71),
                    'rank': np.random.randint(1, 21),
                    'fastestLapTime': f'1:{np.random.randint(10, 30)}.{np.random.randint(0, 999):03d}',
                    'fastestLapSpeed': np.random.uniform(200, 230),
                    'statusId': 1
                })
                result_id += 1

        self.results = pd.DataFrame(results_data)

        # Generate driver standings
        standings_data = []
        standing_id = 1

        for year in years:
            year_races = self.races[self.races['year'] == year]
            drivers_points = {}

            for _, race in year_races.iterrows():
                race_results = self.results[self.results['raceId'] == race['raceId']]
                for _, result in race_results.iterrows():
                    if result['driverId'] not in drivers_points:
                        drivers_points[result['driverId']] = 0
                    drivers_points[result['driverId']] += result['points']

            sorted_drivers = sorted(drivers_points.items(), key=lambda x: x[1], reverse=True)

            for pos, (driver_id, points) in enumerate(sorted_drivers, 1):
                standings_data.append({
                    'driverStandingsId': standing_id,
                    'raceId': year_races['raceId'].iloc[-1],  # Last race of the year
                    'driverId': driver_id,
                    'points': points,
                    'position': pos,
                    'wins': np.random.randint(0, 8) if pos <= 5 else 0
                })
                standing_id += 1

        self.driver_standings = pd.DataFrame(standings_data)

        print("✅ Sample data generated successfully!")
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess and clean the data"""
        # Convert date columns
        if 'date' in self.races.columns:
            self.races['date'] = pd.to_datetime(self.races['date'])

        # Add driver ages
        if 'dob' in self.drivers.columns:
            self.drivers['dob'] = pd.to_datetime(self.drivers['dob'])
            current_year = 2020
            self.drivers['age'] = current_year - self.drivers['dob'].dt.year

        # Create merged datasets for analysis
        self.merged_results = self.results.merge(self.races, on='raceId') \
            .merge(self.drivers, on='driverId') \
            .merge(self.constructors, on='constructorId')

        print("✅ Data preprocessing completed!")

    def data_overview(self):
        """Display comprehensive data overview"""
        print("=" * 60)
        print("📊 FORMULA 1 DATASET OVERVIEW")
        print("=" * 60)

        datasets = {
            'Circuits': self.circuits,
            'Constructors': self.constructors,
            'Drivers': self.drivers,
            'Races': self.races,
            'Results': self.results,
            'Driver Standings': self.driver_standings
        }

        for name, df in datasets.items():
            if df is not None:
                print(f"\n🏁 {name}:")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Year range analysis
        if self.races is not None:
            print(f"\n📅 Year Range: {self.races['year'].min()} - {self.races['year'].max()}")
            print(f"🏆 Total Races: {len(self.races)}")
            print(f"🏎️  Total Drivers: {len(self.drivers)}")
            print(f"🏭 Total Constructors: {len(self.constructors)}")

    def championship_analysis(self):
        """Analyze championship statistics by year"""
        print("\n" + "=" * 60)
        print("🏆 CHAMPIONSHIP ANALYSIS")
        print("=" * 60)

        # World Champions by year
        champions_by_year = []

        for year in sorted(self.races['year'].unique()):
            year_races = self.races[self.races['year'] == year]
            if len(year_races) == 0:
                continue

            # Get final standings for the year
            last_race_id = year_races['raceId'].max()
            year_standings = self.driver_standings[self.driver_standings['raceId'] == last_race_id]

            if len(year_standings) > 0:
                champion = year_standings.loc[year_standings['position'] == 1]
                if len(champion) > 0:
                    champion_info = champion.merge(self.drivers, on='driverId').iloc[0]
                    champions_by_year.append({
                        'year': year,
                        'champion': f"{champion_info['forename']} {champion_info['surname']}",
                        'points': champion_info['points'],
                        'nationality': champion_info['nationality']
                    })

        champions_df = pd.DataFrame(champions_by_year)

        if len(champions_df) > 0:
            print(f"📊 Championship Winners ({len(champions_df)} years analyzed):")
            print(champions_df.tail(10).to_string(index=False))

            # Most successful drivers
            champion_counts = champions_df['champion'].value_counts()
            print(f"\n🏅 Most Championships Won:")
            print(champion_counts.head(10).to_string())

        return champions_df

    def visualize_championship_trends(self, champions_df):
        """Create interactive championship trend visualizations"""
        if len(champions_df) == 0:
            print("No championship data available for visualization.")
            return

        # Points evolution over time
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Championship Points Over Time', 'Championships by Nationality',
                            'Era Analysis', 'Dominant Periods'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.1
        )

        # Championship points trend
        fig.add_trace(
            go.Scatter(x=champions_df['year'], y=champions_df['points'],
                       mode='lines+markers', name='Championship Points',
                       line=dict(color='red', width=3)),
            row=1, col=1
        )

        # Nationality distribution
        nationality_counts = champions_df['nationality'].value_counts()
        fig.add_trace(
            go.Pie(labels=nationality_counts.index, values=nationality_counts.values,
                   name="Nationality Distribution"),
            row=1, col=2
        )

        # Era analysis - group by decades
        champions_df['decade'] = (champions_df['year'] // 10) * 10
        era_analysis = champions_df.groupby('decade').agg({
            'points': 'mean',
            'champion': 'count'
        }).reset_index()

        fig.add_trace(
            go.Bar(x=era_analysis['decade'], y=era_analysis['champion'],
                   name='Championships per Decade', marker_color='lightblue'),
            row=2, col=1
        )

        fig.update_layout(
            title_text="Formula 1 Championship Analysis Dashboard",
            showlegend=True,
            height=800
        )

        fig.show()

    def driver_performance_analysis(self):
        """Comprehensive driver performance analysis"""
        print("\n" + "=" * 60)
        print("🏎️  DRIVER PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Driver statistics
        driver_stats = self.merged_results.groupby(['driverId', 'forename', 'surname', 'nationality']).agg({
            'points': ['sum', 'mean'],
            'position': ['count', 'mean'],
            'raceId': 'count'
        }).reset_index()

        driver_stats.columns = ['driverId', 'forename', 'surname', 'nationality',
                                'total_points', 'avg_points', 'total_races', 'avg_position', 'race_count']

        driver_stats = driver_stats.sort_values('total_points', ascending=False)

        print("🏆 Top 10 Drivers by Total Points:")
        print(driver_stats.head(10)[['forename', 'surname', 'total_points', 'total_races', 'avg_position']].to_string(
            index=False))

        return driver_stats

    def constructor_analysis(self):
        """Analyze constructor performance"""
        print("\n" + "=" * 60)
        print("🏭 CONSTRUCTOR ANALYSIS")
        print("=" * 60)

        constructor_stats = self.merged_results.groupby(['constructorId', 'name', 'nationality']).agg({
            'points': ['sum', 'mean'],
            'position': ['count', 'mean'],
            'raceId': 'count'
        }).reset_index()

        constructor_stats.columns = ['constructorId', 'name', 'nationality',
                                     'total_points', 'avg_points', 'total_races', 'avg_position', 'race_count']

        constructor_stats = constructor_stats.sort_values('total_points', ascending=False)

        print("🏆 Top 10 Constructors by Total Points:")
        print(
            constructor_stats.head(10)[['name', 'total_points', 'total_races', 'avg_position']].to_string(index=False))

        return constructor_stats

    def create_interactive_dashboard(self):
        """Create comprehensive interactive dashboard"""
        print("\n" + "=" * 60)
        print("📊 CREATING INTERACTIVE DASHBOARD")
        print("=" * 60)

        # Championship analysis
        champions_df = self.championship_analysis()

        # Driver performance
        driver_stats = self.driver_performance_analysis()

        # Constructor analysis
        constructor_stats = self.constructor_analysis()

        # Create visualizations
        self.visualize_championship_trends(champions_df)
        self.visualize_driver_performance(driver_stats)
        self.visualize_constructor_performance(constructor_stats)
        self.visualize_geographical_analysis()

        # Machine learning insights
        self.ml_insights()

    def visualize_driver_performance(self, driver_stats):
        """Create driver performance visualizations"""
        top_drivers = driver_stats.head(20)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top Drivers by Points', 'Points vs Races',
                            'Average Position Distribution', 'Nationality Distribution'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "pie"}]]
        )

        # Top drivers bar chart
        fig.add_trace(
            go.Bar(x=top_drivers['forename'] + ' ' + top_drivers['surname'],
                   y=top_drivers['total_points'],
                   name='Total Points',
                   marker_color='red'),
            row=1, col=1
        )

        # Points vs races scatter
        fig.add_trace(
            go.Scatter(x=top_drivers['total_races'],
                       y=top_drivers['total_points'],
                       mode='markers',
                       name='Points vs Races',
                       text=top_drivers['forename'] + ' ' + top_drivers['surname'],
                       marker=dict(size=10, color='blue')),
            row=1, col=2
        )

        # Average position histogram
        fig.add_trace(
            go.Histogram(x=top_drivers['avg_position'],
                         name='Avg Position Distribution',
                         marker_color='green'),
            row=2, col=1
        )

        # Nationality pie chart
        nationality_counts = top_drivers['nationality'].value_counts()
        fig.add_trace(
            go.Pie(labels=nationality_counts.index,
                   values=nationality_counts.values,
                   name="Driver Nationalities"),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Driver Performance Analysis Dashboard",
            showlegend=True,
            height=800
        )

        fig.show()

    def visualize_constructor_performance(self, constructor_stats):
        """Create constructor performance visualizations"""
        top_constructors = constructor_stats.head(15)

        fig = go.Figure()

        # Constructor performance over time
        fig.add_trace(
            go.Bar(x=top_constructors['name'],
                   y=top_constructors['total_points'],
                   name='Total Points',
                   marker_color='orange')
        )

        fig.update_layout(
            title='Constructor Performance - Total Points',
            xaxis_title='Constructor',
            yaxis_title='Total Points',
            xaxis_tickangle=-45
        )

        fig.show()

    def visualize_geographical_analysis(self):
        """Create geographical analysis of circuits"""
        if self.circuits is None:
            return

        # World map of circuits
        fig = go.Figure(data=go.Scattermapbox(
            lat=self.circuits['lat'],
            lon=self.circuits['lng'],
            mode='markers',
            marker=dict(size=12, color='red'),
            text=self.circuits['name'] + ', ' + self.circuits['country'],
            hoverinfo='text'
        ))

        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=20, lon=0),
                zoom=1
            ),
            title='Formula 1 Circuits Around the World',
            height=600
        )

        fig.show()

    def ml_insights(self):
        """Apply machine learning for insights"""
        print("\n" + "=" * 60)
        print("🤖 MACHINE LEARNING INSIGHTS")
        print("=" * 60)

        # Prepare data for ML
        ml_data = self.merged_results.groupby('driverId').agg({
            'points': ['sum', 'mean', 'std'],
            'position': ['mean', 'std'],
            'raceId': 'count',
            'age': 'first'
        }).reset_index()

        ml_data.columns = ['driverId', 'total_points', 'avg_points', 'points_std',
                           'avg_position', 'position_std', 'race_count', 'age']

        # Fill NaN values
        ml_data = ml_data.fillna(0)

        # Feature selection for clustering
        features = ['total_points', 'avg_points', 'avg_position', 'race_count', 'age']
        X = ml_data[features]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        ml_data['cluster'] = kmeans.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Plot clusters
        fig = go.Figure()

        for cluster in range(4):
            cluster_data = ml_data[ml_data['cluster'] == cluster]
            cluster_pca = X_pca[ml_data['cluster'] == cluster]

            fig.add_trace(
                go.Scatter(x=cluster_pca[:, 0], y=cluster_pca[:, 1],
                           mode='markers',
                           name=f'Cluster {cluster}',
                           marker=dict(size=8),
                           text=f'Cluster {cluster}')
            )

        fig.update_layout(
            title='Driver Performance Clusters (PCA Visualization)',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component'
        )

        fig.show()

        # Cluster analysis
        cluster_analysis = ml_data.groupby('cluster')[features].mean()
        print("\n🎯 Driver Performance Clusters:")
        print(cluster_analysis.round(2))

        # Predictive modeling - predict points based on other features
        feature_cols = ['avg_position', 'race_count', 'age']
        X_pred = ml_data[feature_cols].fillna(0)
        y_pred = ml_data['total_points']

        # Remove outliers for better model performance
        from scipy import stats
        z_scores = np.abs(stats.zscore(X_pred))
        mask = (z_scores < 3).all(axis=1)
        X_pred_clean = X_pred[mask]
        y_pred_clean = y_pred[mask]

        # Linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_pred_clean, y_pred_clean)

        y_pred_lr = lr_model.predict(X_pred_clean)
        r2 = r2_score(y_pred_clean, y_pred_lr)

        print(f"\n📈 Predictive Model Performance:")
        print(f"R² Score: {r2:.3f}")
        print(f"Features: {feature_cols}")
        print(f"Coefficients: {lr_model.coef_.round(3)}")

    def yearly_statistics(self, year=None):
        """Generate detailed statistics for a specific year"""
        if year is None:
            year = self.races['year'].max()

        print(f"\n" + "=" * 60)
        print(f"📊 DETAILED STATISTICS FOR {year}")
        print("=" * 60)

        year_races = self.races[self.races['year'] == year]
        year_results = self.merged_results[self.merged_results['year'] == year]

        if len(year_races) == 0:
            print(f"No data available for year {year}")
            return

        print(f"🏁 Total Races: {len(year_races)}")
        print(f"🏎️  Participating Drivers: {year_results['driverId'].nunique()}")
        print(f"🏭 Participating Constructors: {year_results['constructorId'].nunique()}")

        # Race winners
        race_winners = year_results[year_results['position'] == 1]
        if len(race_winners) > 0:
            winner_counts = race_winners['forename'].str.cat(race_winners['surname'], sep=' ').value_counts()
            print(f"\n🏆 Race Winners in {year}:")
            print(winner_counts.to_string())

        # Constructor performance
        constructor_points = year_results.groupby('name')['points'].sum().sort_values(ascending=False)
        print(f"\n🏭 Constructor Championship Standings {year}:")
        print(constructor_points.head(10).to_string())

        # Driver championship
        driver_points = year_results.groupby(['forename', 'surname'])['points'].sum().sort_values(ascending=False)
        print(f"\n🏎️  Driver Championship Standings {year}:")
        print(driver_points.head(10).to_string())

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🏁 Starting Formula 1 Complete Data Analysis")
        print("=" * 60)

        # Load data
        self.load_data()

        # Data overview
        self.data_overview()

        # Create interactive dashboard
        self.create_interactive_dashboard()

        # Year-specific analysis
        recent_years = sorted(self.races['year'].unique())[-5:]  # Last 5 years
        for year in recent_years:
            self.yearly_statistics(year)

        print("\n" + "=" * 60)
        print("✅ Analysis Complete!")
        print("=" * 60)

        # Usage instructions
        print("\n📋 Usage Instructions:")
        print("1. Load your F1 dataset: analyzer = F1DataAnalyzer('/path/to/your/dataset')")
        print("2. Run analysis: analyzer.run_complete_analysis()")
        print("3. View specific year: analyzer.yearly_statistics(2020)")
        print("4. Custom analysis: Use individual methods for specific insights")


def main():
    """
    Main function to demonstrate the F1 Analysis Tool
    """
    print("🏁 Formula 1 Interactive Data Visualization Tool")
    print("=" * 60)

    # Initialize analyzer
    analyzer = F1DataAnalyzer()

    # Run complete analysis
    analyzer.run_complete_analysis()

    # Interactive menu
    while True:
        print("\n" + "=" * 40)
        print("🎮 Interactive Menu")
        print("=" * 40)
        print("1. Championship Analysis")
        print("2. Driver Performance")
        print("3. Constructor Analysis")
        print("4. Year-specific Stats")
        print("5. Machine Learning Insights")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ")

        if choice == '1':
            analyzer.championship_