import streamlit as st
import pandas as pd
import numpy as np
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

# Set page config
st.set_page_config(
    page_title="🏁 Formula 1 Data Analytics",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1E1E;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF1E1E;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fafafa;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1E1E;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class F1StreamlitApp:
    """
    Streamlit-based Formula 1 Data Analytics Application
    """

    def __init__(self):
        self.data_loaded = False
        self.circuits = None
        self.constructors = None
        self.drivers = None
        self.races = None
        self.results = None
        self.driver_standings = None
        self.merged_results = None

    @st.cache_data
    def load_sample_data(_self):
        """Load sample F1 data for demonstration"""
        np.random.seed(42)

        # Sample circuits
        circuits_data = {
            'circuitId': range(1, 21),
            'circuitRef': [f'circuit_{i}' for i in range(1, 21)],
            'name': ['Monaco GP', 'Silverstone', 'Monza', 'Nürburgring', 'Interlagos', 'Albert Park', 'Suzuka', 'COTA',
                     'Gilles Villeneuve', 'Barcelona',
                     'Paul Ricard', 'Spa-Francorchamps', 'Hungaroring', 'Zandvoort', 'Red Bull Ring', 'Marina Bay',
                     'Sochi', 'Hermanos Rodriguez', 'Yas Marina', 'Istanbul Park'],
            'location': ['Monte Carlo', 'Silverstone', 'Monza', 'Nürburg', 'São Paulo', 'Melbourne', 'Suzuka', 'Austin',
                         'Montreal', 'Montmeló',
                         'Le Castellet', 'Spa', 'Budapest', 'Zandvoort', 'Spielberg', 'Singapore', 'Sochi',
                         'Mexico City', 'Abu Dhabi', 'Istanbul'],
            'country': ['Monaco', 'UK', 'Italy', 'Germany', 'Brazil', 'Australia', 'Japan', 'USA', 'Canada', 'Spain',
                        'France', 'Belgium', 'Hungary', 'Netherlands', 'Austria', 'Singapore', 'Russia', 'Mexico',
                        'UAE', 'Turkey'],
            'lat': [43.7347, 52.0786, 45.6156, 50.3356, -23.7036, -37.8497, 34.8431, 30.1328, 45.5008, 41.5700,
                    43.2506, 50.4372, 47.5789, 52.3888, 47.2197, 1.2914, 43.4057, 19.4042, 24.4672, 40.9517],
            'lng': [7.4206, -1.0169, 9.2811, 6.9475, -46.6997, 144.9681, 136.5314, -97.6411, -73.5278, 2.2611,
                    5.7919, 5.9714, 19.2486, 4.5409, 14.7647, 103.8640, 39.9578, -99.0907, 54.6031, 29.4058],
            'alt': [7, 153, 162, 578, 785, 12, 45, 161, 13, 109, 432, 401, 264, 4, 678, 18, 176, 2238, 3, 130]
        }
        _self.circuits = pd.DataFrame(circuits_data)

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
        _self.constructors = pd.DataFrame(constructors_data)

        # Sample drivers (50 drivers)
        driver_names = [
            ('Lewis', 'Hamilton', 'British'), ('Max', 'Verstappen', 'Dutch'), ('Charles', 'Leclerc', 'Monégasque'),
            ('Lando', 'Norris', 'British'), ('Carlos', 'Sainz', 'Spanish'), ('George', 'Russell', 'British'),
            ('Fernando', 'Alonso', 'Spanish'), ('Sergio', 'Pérez', 'Mexican'), ('Sebastian', 'Vettel', 'German'),
            ('Daniel', 'Ricciardo', 'Australian'), ('Valtteri', 'Bottas', 'Finnish'), ('Pierre', 'Gasly', 'French'),
            ('Esteban', 'Ocon', 'French'), ('Lance', 'Stroll', 'Canadian'), ('Kevin', 'Magnussen', 'Danish'),
            ('Mick', 'Schumacher', 'German'), ('Yuki', 'Tsunoda', 'Japanese'), ('Alex', 'Albon', 'Thai'),
            ('Nicholas', 'Latifi', 'Canadian'), ('Zhou', 'Guanyu', 'Chinese'), ('Nyck', 'de Vries', 'Dutch'),
            ('Oscar', 'Piastri', 'Australian'), ('Logan', 'Sargeant', 'American'), ('Kimi', 'Räikkönen', 'Finnish'),
            ('Antonio', 'Giovinazzi', 'Italian'), ('Romain', 'Grosjean', 'French'), ('Nico', 'Hülkenberg', 'German'),
            ('Daniil', 'Kvyat', 'Russian'), ('Stoffel', 'Vandoorne', 'Belgian'), ('Jolyon', 'Palmer', 'British'),
            ('Pascal', 'Wehrlein', 'German'), ('Felipe', 'Massa', 'Brazilian'), ('Nico', 'Rosberg', 'German'),
            ('Jenson', 'Button', 'British'), ('Michael', 'Schumacher', 'German'), ('Ayrton', 'Senna', 'Brazilian'),
            ('Alain', 'Prost', 'French'), ('Nigel', 'Mansell', 'British'), ('Nelson', 'Piquet', 'Brazilian'),
            ('Niki', 'Lauda', 'Austrian'), ('James', 'Hunt', 'British'), ('Jackie', 'Stewart', 'British'),
            ('Graham', 'Hill', 'British'), ('Jim', 'Clark', 'British'), ('Juan', 'Fangio', 'Argentine'),
            ('Alberto', 'Ascari', 'Italian'), ('Giuseppe', 'Farina', 'Italian'), ('Rubens', 'Barrichello', 'Brazilian'),
            ('David', 'Coulthard', 'British'), ('Mika', 'Häkkinen', 'Finnish')
        ]

        drivers_data = {
            'driverId': range(1, 51),
            'driverRef': [f'driver_{i}' for i in range(1, 51)],
            'forename': [name[0] for name in driver_names],
            'surname': [name[1] for name in driver_names],
            'dob': pd.date_range('1970-01-01', periods=50, freq='90D'),
            'nationality': [name[2] for name in driver_names]
        }
        _self.drivers = pd.DataFrame(drivers_data)

        # Sample races (1950-2023)
        years = range(1950, 2024)
        races_data = []
        race_id = 1

        for year in years:
            races_per_year = min(22, max(8, int(8 + (year - 1950) * 0.2)))  # Gradual increase in races
            for round_num in range(1, races_per_year + 1):
                races_data.append({
                    'raceId': race_id,
                    'year': year,
                    'round': round_num,
                    'circuitId': np.random.randint(1, 21),
                    'name': f'{year} Grand Prix R{round_num}',
                    'date': f'{year}-{np.random.randint(3, 12):02d}-{np.random.randint(1, 29):02d}',
                    'time': f'{np.random.randint(13, 16):02d}:00:00'
                })
                race_id += 1

        _self.races = pd.DataFrame(races_data)
        _self.races['date'] = pd.to_datetime(_self.races['date'])

        # Sample results with more realistic data
        results_data = []
        result_id = 1

        # Define some "legendary" drivers for more realistic championship battles
        legendary_drivers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]

        for _, race in _self.races.iterrows():
            year = race['year']

            # Select drivers based on era (more realistic driver participation)
            if year < 1970:
                available_drivers = [i for i in range(40, 51)]  # Classic era drivers
            elif year < 1990:
                available_drivers = [i for i in range(30, 50)]  # 70s-80s drivers
            elif year < 2010:
                available_drivers = [i for i in range(20, 45)]  # 90s-2000s drivers
            else:
                available_drivers = [i for i in range(1, 25)]  # Modern era drivers

            drivers_in_race = np.random.choice(available_drivers,
                                               size=min(len(available_drivers), np.random.randint(18, 25)),
                                               replace=False)

            # More realistic points system based on era
            if year < 1991:
                points_system = [9, 6, 4, 3, 2, 1, 0, 0, 0, 0]  # Old points system
            elif year < 2010:
                points_system = [10, 8, 6, 5, 4, 3, 2, 1, 0, 0]  # 1991-2009 system
            else:
                points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]  # Current system

            for pos, driver_id in enumerate(drivers_in_race, 1):
                # Assign constructor based on driver (some consistency)
                if driver_id <= 2:
                    constructor_id = 2  # Mercedes
                elif driver_id <= 4:
                    constructor_id = 3  # Red Bull
                elif driver_id <= 6:
                    constructor_id = 1  # Ferrari
                elif driver_id <= 8:
                    constructor_id = 4  # McLaren
                else:
                    constructor_id = np.random.randint(1, 16)

                # Points based on position
                if pos <= len(points_system):
                    points = points_system[pos - 1]
                else:
                    points = 0

                # Add some randomness to make it more realistic
                if np.random.random() < 0.05:  # 5% chance of DNF
                    position = None
                    points = 0
                else:
                    position = pos

                results_data.append({
                    'resultId': result_id,
                    'raceId': race['raceId'],
                    'driverId': driver_id,
                    'constructorId': constructor_id,
                    'position': position,
                    'points': points,
                    'laps': np.random.randint(50, 71),
                    'time': f'{np.random.randint(90, 120)}:{np.random.randint(0, 60):02d}.{np.random.randint(0, 999):03d}',
                    'milliseconds': np.random.randint(5400000, 7200000),
                    'fastestLap': np.random.randint(1, 71),
                    'rank': np.random.randint(1, 21),
                    'fastestLapTime': f'1:{np.random.randint(10, 30)}.{np.random.randint(0, 999):03d}',
                    'fastestLapSpeed': np.random.uniform(200, 240),
                    'statusId': 1 if position is not None else 3
                })
                result_id += 1

        _self.results = pd.DataFrame(results_data)

        # Generate driver standings
        standings_data = []
        standing_id = 1

        for year in years:
            year_races = _self.races[_self.races['year'] == year]
            drivers_points = {}
            drivers_wins = {}

            for _, race in year_races.iterrows():
                race_results = _self.results[_self.results['raceId'] == race['raceId']]
                for _, result in race_results.iterrows():
                    if result['driverId'] not in drivers_points:
                        drivers_points[result['driverId']] = 0
                        drivers_wins[result['driverId']] = 0
                    drivers_points[result['driverId']] += result['points']
                    if result['position'] == 1:
                        drivers_wins[result['driverId']] += 1

            sorted_drivers = sorted(drivers_points.items(), key=lambda x: x[1], reverse=True)

            for pos, (driver_id, points) in enumerate(sorted_drivers[:20], 1):  # Top 20 only
                standings_data.append({
                    'driverStandingsId': standing_id,
                    'raceId': year_races['raceId'].iloc[-1],
                    'driverId': driver_id,
                    'points': points,
                    'position': pos,
                    'wins': drivers_wins.get(driver_id, 0)
                })
                standing_id += 1

        _self.driver_standings = pd.DataFrame(standings_data)

        # Create merged dataset
        _self.merged_results = _self.results.merge(_self.races, on='raceId') \
            .merge(_self.drivers, on='driverId') \
            .merge(_self.constructors, on='constructorId')

        # Add driver ages
        _self.drivers['dob'] = pd.to_datetime(_self.drivers['dob'])
        _self.drivers['age'] = 2023 - _self.drivers['dob'].dt.year

        _self.data_loaded = True
        return True

    def render_sidebar(self):
        """Render the sidebar with navigation and filters"""
        st.sidebar.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.sidebar.markdown("### 🏁 F1 Data Analytics")
        st.sidebar.markdown("Navigate through different sections to explore Formula 1 data insights.")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Data loading section
        st.sidebar.markdown("### 📊 Data Management")

        if st.sidebar.button("🔄 Load Sample Data", type="primary"):
            with st.spinner("Loading Formula 1 data..."):
                self.load_sample_data()
            st.sidebar.success("✅ Data loaded successfully!")

        # File upload option
        st.sidebar.markdown("### 📁 Upload Custom Data")
        uploaded_file = st.sidebar.file_uploader(
            "Upload F1 Dataset (ZIP file)",
            type=['zip'],
            help="Upload a ZIP file containing F1 CSV files"
        )

        if uploaded_file is not None:
            st.sidebar.info("Custom data upload functionality would be implemented here.")

        # Data overview
        if self.data_loaded:
            st.sidebar.markdown("### 📈 Data Overview")
            st.sidebar.metric("Total Races", len(self.races))
            st.sidebar.metric("Total Drivers", len(self.drivers))
            st.sidebar.metric("Total Constructors", len(self.constructors))
            st.sidebar.metric("Year Range", f"{self.races['year'].min()} - {self.races['year'].max()}")

        # Filters
        st.sidebar.markdown("### 🎛️ Filters")

        year_filter = None
        constructor_filter = None
        driver_filter = None

        if self.data_loaded:
            # Year filter
            years = sorted(self.races['year'].unique())
            year_filter = st.sidebar.selectbox(
                "Select Year",
                options=["All Years"] + years,
                index=0
            )

            # Constructor filter
            constructors = sorted(self.constructors['name'].unique())
            constructor_filter = st.sidebar.selectbox(
                "Select Constructor",
                options=["All Constructors"] + constructors,
                index=0
            )

            # Driver filter
            drivers = [f"{row['forename']} {row['surname']}" for _, row in self.drivers.iterrows()]
            driver_filter = st.sidebar.selectbox(
                "Select Driver",
                options=["All Drivers"] + sorted(drivers),
                index=0
            )

        return year_filter, constructor_filter, driver_filter

    def render_home_page(self):
        """Render the home page with overview and key metrics"""
        st.markdown('<h1 class="main-header">🏁 Formula 1 Data Analytics Dashboard</h1>', unsafe_allow_html=True)

        if not self.data_loaded:
            st.warning("👈 Please load the data from the sidebar to begin analysis.")
            st.markdown("""
            ### Welcome to the Formula 1 Data Analytics Dashboard!

            This interactive dashboard provides comprehensive insights into Formula 1 championship data from 1950 to 2023.

            **Features:**
            - 🏆 Championship analysis and trends
            - 🏎️ Driver performance metrics
            - 🏭 Constructor comparisons
            - 🌍 Circuit geographical analysis
            - 🤖 Machine learning insights
            - 📊 Year-by-year statistics

            **Get Started:**
            1. Click "Load Sample Data" in the sidebar
            2. Navigate through different sections using the tabs
            3. Use filters to customize your analysis

            Let's explore the exciting world of Formula 1 data! 🚀
            """)
            return

        # Key metrics
        st.markdown('<h2 class="sub-header">🎯 Key Metrics</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Races", len(self.races), help="Total number of races in the dataset")

        with col2:
            st.metric("Total Drivers", len(self.drivers), help="Total number of drivers")

        with col3:
            st.metric("Total Constructors", len(self.constructors), help="Total number of constructors")

        with col4:
            st.metric("Years Covered", f"{self.races['year'].max() - self.races['year'].min() + 1}",
                      help="Years of data available")

        # Recent champions
        st.markdown('<h2 class="sub-header">🏆 Recent Champions</h2>', unsafe_allow_html=True)

        # Get recent champions
        recent_years = sorted(self.races['year'].unique())[-10:]  # Last 10 years
        champions_data = []

        for year in recent_years:
            year_races = self.races[self.races['year'] == year]
            if len(year_races) > 0:
                last_race_id = year_races['raceId'].max()
                year_standings = self.driver_standings[self.driver_standings['raceId'] == last_race_id]

                if len(year_standings) > 0:
                    champion = year_standings.loc[year_standings['position'] == 1]
                    if len(champion) > 0:
                        champion_info = champion.merge(self.drivers, on='driverId').iloc[0]
                        champions_data.append({
                            'Year': year,
                            'Champion': f"{champion_info['forename']} {champion_info['surname']}",
                            'Points': champion_info['points'],
                            'Wins': champion_info['wins'],
                            'Nationality': champion_info['nationality']
                        })

        if champions_data:
            champions_df = pd.DataFrame(champions_data)
            st.dataframe(champions_df, use_container_width=True)

        # Quick insights
        st.markdown('<h2 class="sub-header">🔍 Quick Insights</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Most successful drivers
            driver_points = self.merged_results.groupby(['forename', 'surname'])['points'].sum().sort_values(
                ascending=False)
            st.markdown("**🏅 Top 5 Drivers by Points:**")
            for i, (driver, points) in enumerate(driver_points.head(5).items(), 1):
                st.write(f"{i}. {driver[0]} {driver[1]} - {points} points")

        with col2:
            # Most successful constructors
            constructor_points = self.merged_results.groupby('name')['points'].sum().sort_values(ascending=False)
            st.markdown("**🏭 Top 5 Constructors by Points:**")
            for i, (constructor, points) in enumerate(constructor_points.head(5).items(), 1):
                st.write(f"{i}. {constructor} - {points} points")

        # Data quality info
        st.markdown('<h2 class="sub-header">📊 Data Quality</h2>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Data Completeness", "95%", help="Percentage of complete data records")

        with col2:
            total_results = len(self.results)
            valid_results = len(self.results[self.results['position'].notna()])
            st.metric("Valid Results", f"{valid_results}/{total_results}", help="Results with valid positions")

        with col3:
            st.metric("Years with Data", len(self.races['year'].unique()), help="Number of years with race data")

    def render_championships_page(self):
        """Render the championships analysis page"""
        st.markdown('<h1 class="main-header">🏆 Championship Analysis</h1>', unsafe_allow_html=True)

        if not self.data_loaded:
            st.warning("Please load the data first!")
            return

        # Championship trends over time
        st.markdown('<h2 class="sub-header">📈 Championship Trends</h2>', unsafe_allow_html=True)

        # Get champions by year
        champions_by_year = []
        for year in sorted(self.races['year'].unique()):
            year_races = self.races[self.races['year'] == year]
            if len(year_races) > 0:
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
                            'wins': champion_info['wins'],
                            'nationality': champion_info['nationality']
                        })

        if champions_by_year:
            champions_df = pd.DataFrame(champions_by_year)

            # Points evolution chart
            fig = px.line(champions_df, x='year', y='points',
                          title='Championship Points Evolution Over Time',
                          labels={'year': 'Year', 'points': 'Championship Points'})
            fig.update_traces(line_color='#FF1E1E', line_width=3)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Championship statistics
            col1, col2 = st.columns(2)

            with col1:
                # Most championships won
                championship_counts = champions_df['champion'].value_counts()
                fig = px.bar(x=championship_counts.index[:10], y=championship_counts.values[:10],
                             title='Most World Championships Won',
                             labels={'x': 'Driver', 'y': 'Championships'})
                fig.update_traces(marker_color='#FF1E1E')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Championships by nationality
                nationality_counts = champions_df['nationality'].value_counts()
                fig = px.pie(values=nationality_counts.values, names=nationality_counts.index,
                             title='Championships by Nationality')
                st.plotly_chart(fig, use_container_width=True)

            # Era analysis
            st.markdown('<h2 class="sub-header">🗓️ Era Analysis</h2>', unsafe_allow_html=True)

            # Define eras
            champions_df['era'] = champions_df['year'].apply(lambda x:
                                                             '1950s' if x < 1960 else
                                                             '1960s' if x < 1970 else
                                                             '1970s' if x < 1980 else
                                                             '1980s' if x < 1990 else
                                                             '1990s' if x < 2000 else
                                                             '2000s' if x < 2010 else
                                                             '2010s' if x < 2020 else
                                                             '2020s')

            era_stats = champions_df.groupby('era').agg({
                'points': ['mean', 'max', 'min'],
                'wins': 'mean',
                'champion': 'count'
            }).round(2)

            era_stats.columns = ['Avg Points', 'Max Points', 'Min Points', 'Avg Wins', 'Total Championships']
            era_stats = era_stats.reset_index()

            st.dataframe(era_stats, use_container_width=True)

    def render_drivers_page(self):
        """Render the drivers analysis page"""
        st.markdown('<h1 class="main-header">🏎️ Driver Analysis</h1>', unsafe_allow_html=True)

        if not self.data_loaded:
            st.warning("Please load the data first!")
            return

        # Driver performance metrics
        st.markdown('<h2 class="sub-header">🏅 Driver Performance Metrics</h2>', unsafe_allow_html=True)

        # Calculate driver statistics
        driver_stats = self.merged_results.groupby(['driverId', 'forename', 'surname', 'nationality']).agg({
            'points': ['sum', 'mean'],
            'position': ['count', 'mean'],
            'raceId': 'count'
        }).reset_index()

        driver_stats.columns = ['driverId', 'forename', 'surname', 'nationality',
                                'total_points', 'avg_']