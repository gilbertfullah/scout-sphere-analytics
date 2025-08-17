# team_analytics.py
"""
Comprehensive Team Analytics System for Football/Soccer
Uses mplsoccer, statsbombpy, and other modern packages
Integrates with Supabase for data storage and retrieval
"""

import os
import json
import base64
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplsoccer import Pitch, VerticalPitch, Radar, PyPizza, FontManager
from mplsoccer.statsbomb import read_event
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.ndimage import gaussian_filter
from supabase import create_client, Client
from PIL import Image
import requests

# Configure matplotlib for better quality
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

class TeamAnalytics:
    """
    Comprehensive team analytics class for football/soccer data analysis
    """
    
    def __init__(self, supabase_url: str, supabase_key: str, team_id: str):
        """
        Initialize the analytics system
        
        Args:
            supabase_url: Your Supabase project URL
            supabase_key: Your Supabase anon key
            team_id: The ID of the team to analyze
        """
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.team_id = team_id
        self.team_data = None
        self.matches_data = None
        self.events_data = None
        self.players_data = None
        
    def fetch_team_data(self, days_range: int = 30) -> Dict:
        """
        Fetch team data from Supabase
        """
        try:
            # Fetch team info
            team_response = self.supabase.table('teams').select('*').eq('id', self.team_id).single().execute()
            self.team_data = team_response.data
            
            # Fetch matches
            start_date = (datetime.now() - timedelta(days=days_range)).isoformat()
            matches_response = self.supabase.table('matches').select('*').or_(
                f'home_team_id.eq.{self.team_id},away_team_id.eq.{self.team_id}'
            ).gte('match_date', start_date).execute()
            self.matches_data = pd.DataFrame(matches_response.data)
            
            # Fetch match events
            if not self.matches_data.empty:
                match_ids = self.matches_data['id'].tolist()
                events_response = self.supabase.table('match_events').select('*').in_(
                    'match_id', match_ids
                ).execute()
                self.events_data = pd.DataFrame(events_response.data)
            
            # Fetch players
            players_response = self.supabase.table('players').select('*').eq(
                'current_team_id', self.team_id
            ).execute()
            self.players_data = pd.DataFrame(players_response.data)
            
            return {
                'team': self.team_data,
                'matches': len(self.matches_data),
                'players': len(self.players_data)
            }
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def generate_heatmap(self, player_id: Optional[str] = None, save_path: Optional[str] = None) -> str:
        """
        Generate a heatmap of team or player activity
        
        Returns:
            Base64 encoded image string for display in Flutter
        """
        # Create a pitch
        pitch = Pitch(pitch_color='#22312b', line_color='white', linewidth=2)
        fig, ax = pitch.draw(figsize=(16, 10))
        
        # Filter events for the team
        if self.events_data is not None and not self.events_data.empty:
            team_events = self.events_data[self.events_data['team_id'] == self.team_id]
            
            if player_id:
                team_events = team_events[team_events['player_id'] == player_id]
            
            # Extract x, y coordinates (assuming you have these columns)
            # If not, generate sample data
            if 'x' in team_events.columns and 'y' in team_events.columns:
                x = team_events['x'].values
                y = team_events['y'].values
            else:
                # Generate sample data for demonstration
                np.random.seed(42)
                x = np.random.normal(60, 20, 500)
                y = np.random.normal(40, 15, 500)
                x = np.clip(x, 0, 120)
                y = np.clip(y, 0, 80)
            
            # Create heatmap
            bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(25, 18))
            bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], 1)
            
            # Plot heatmap
            pcm = pitch.heatmap(bin_statistic, ax=ax, cmap='hot', edgecolors='#22312b')
            
            # Add colorbar
            cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
            cbar.set_label('Activity Intensity', fontsize=12, color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Add title
        player_name = "Team" if not player_id else self._get_player_name(player_id)
        ax.set_title(f'{player_name} Activity Heatmap', fontsize=20, color='white', pad=20)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#22312b')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Save to file if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(image_base64))
        
        # Store in Supabase
        self._store_visualization('heatmap', image_base64)
        
        return image_base64
    
    def generate_shot_map(self, match_id: Optional[str] = None) -> str:
        """
        Generate a shot map showing all shots and their outcomes
        """
        pitch = VerticalPitch(pitch_color='#22312b', line_color='white', 
                              linewidth=2, half=True)
        fig, ax = pitch.draw(figsize=(12, 10))
        
        # Generate sample shot data (replace with real data)
        np.random.seed(42)
        shots_data = pd.DataFrame({
            'x': np.random.uniform(88, 120, 15),
            'y': np.random.uniform(18, 62, 15),
            'xG': np.random.uniform(0.01, 0.8, 15),
            'outcome': np.random.choice(['Goal', 'Saved', 'Missed', 'Blocked'], 15),
            'player': [f'Player {i%5+1}' for i in range(15)]
        })
        
        # Color map for outcomes
        colors = {'Goal': '#00ff00', 'Saved': '#ffff00', 
                 'Missed': '#ff0000', 'Blocked': '#ff8800'}
        
        for outcome in colors:
            outcome_shots = shots_data[shots_data['outcome'] == outcome]
            if not outcome_shots.empty:
                pitch.scatter(outcome_shots['x'], outcome_shots['y'], 
                            s=outcome_shots['xG']*500, 
                            c=colors[outcome], 
                            alpha=0.8, 
                            edgecolors='white',
                            linewidth=2,
                            ax=ax, 
                            label=outcome)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=12, framealpha=0.8)
        ax.set_title('Shot Map - xG Weighted', fontsize=18, color='white', pad=20)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#22312b')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        self._store_visualization('shot_map', image_base64)
        return image_base64
    
    def generate_pass_network(self, match_id: Optional[str] = None) -> str:
        """
        Generate a pass network showing connections between players
        """
        pitch = Pitch(pitch_color='#22312b', line_color='white', linewidth=2)
        fig, ax = pitch.draw(figsize=(16, 10))
        
        # Generate sample pass network data
        np.random.seed(42)
        positions = pd.DataFrame({
            'player': [f'Player {i+1}' for i in range(11)],
            'x': [10, 30, 25, 35, 30, 50, 60, 55, 80, 85, 75],
            'y': [40, 15, 30, 50, 65, 25, 40, 55, 20, 40, 60],
            'passes_completed': np.random.randint(20, 80, 11)
        })
        
        # Create pass connections
        connections = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if np.random.random() > 0.5:  # 50% chance of connection
                    connections.append({
                        'from': i,
                        'to': j,
                        'passes': np.random.randint(5, 30)
                    })
        
        # Draw connections
        for conn in connections:
            x = [positions.iloc[conn['from']]['x'], positions.iloc[conn['to']]['x']]
            y = [positions.iloc[conn['from']]['y'], positions.iloc[conn['to']]['y']]
            ax.plot(x, y, color='white', linewidth=conn['passes']/10, alpha=0.5)
        
        # Draw player positions
        pitch.scatter(positions['x'], positions['y'], 
                     s=positions['passes_completed']*10,
                     color='#00ff00', edgecolors='white', 
                     linewidth=2, ax=ax, zorder=5)
        
        # Add player numbers
        for idx, row in positions.iterrows():
            ax.text(row['x'], row['y'], str(idx+1), 
                   color='black', fontsize=10, ha='center', va='center',
                   weight='bold', zorder=6)
        
        ax.set_title('Pass Network', fontsize=18, color='white', pad=20)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#22312b')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        self._store_visualization('pass_network', image_base64)
        return image_base64
    
    def generate_goal_distribution(self) -> str:
        """
        Generate interactive goal distribution charts using Plotly
        """
        # Create sample data
        time_periods = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
        goals_by_time = [8, 12, 9, 14, 11, 5]
        
        goal_types = ['Open Play', 'Penalty', 'Free Kick', 'Header', 'Own Goal']
        goals_by_type = [35, 8, 6, 9, 1]
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Goals by Time Period', 'Goals by Type'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}]]
        )
        
        # Time period bar chart
        fig.add_trace(
            go.Bar(
                x=time_periods,
                y=goals_by_time,
                marker_color='lightgreen',
                name='Goals'
            ),
            row=1, col=1
        )
        
        # Goal type pie chart
        fig.add_trace(
            go.Pie(
                labels=goal_types,
                values=goals_by_type,
                hole=0.3,
                marker_colors=['#00ff00', '#ffff00', '#00ffff', '#ff00ff', '#ff0000']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Goal Distribution Analysis",
            showlegend=False,
            height=400,
            template='plotly_dark'
        )
        
        # Convert to JSON for Flutter WebView
        plot_json = fig.to_json()
        
        # Also save as static image
        buffer = BytesIO()
        fig.write_image(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        self._store_visualization('goal_distribution', image_base64, plot_json)
        return plot_json
    
    def generate_player_radar(self, player_id: str) -> str:
        """
        Generate a radar chart for player performance metrics
        """
        # Sample player stats
        params = ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical']
        values = [85, 78, 82, 90, 45, 76]
        
        # Create radar chart using mplsoccer
        radar = Radar(params, min_range=[0]*6, max_range=[100]*6)
        fig, ax = radar.setup_axis()
        rings_inner = radar.draw_circles(ax=ax, facecolor='#22312b', edgecolor='white')
        radar_poly, rings, vertices = radar.draw_radar_solid(
            values, ax=ax, kwargs={'facecolor': '#00ff00', 'alpha': 0.6}
        )
        
        range_labels = radar.draw_range_labels(ax=ax, fontsize=10, color='white')
        param_labels = radar.draw_param_labels(ax=ax, fontsize=12, color='white')
        
        ax.set_title(f'Player Performance Radar', fontsize=16, color='white', pad=20)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#22312b')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        self._store_visualization('player_radar', image_base64)
        return image_base64
    
    def generate_pizza_chart(self, player_id: str) -> str:
        """
        Generate a pizza chart for player percentile rankings
        """
        # Parameter names and values (percentiles)
        params = ['Goals', 'Assists', 'Pass %', 'Dribbles', 'Tackles', 'Interceptions']
        values = [75, 82, 68, 90, 45, 60]
        
        # Color mapping
        slice_colors = ['#00ff00' if v >= 70 else '#ffff00' if v >= 40 else '#ff0000' 
                       for v in values]
        text_colors = ['white'] * len(params)
        
        # Create pizza chart
        baker = PyPizza(
            params=params,
            background_color="#22312b",
            straight_line_color="white",
            straight_line_lw=1,
            last_circle_color="white",
            last_circle_lw=2.5,
            other_circle_lw=0,
            other_circle_color="white",
            inner_circle_size=20
        )
        
        fig, ax = baker.make_pizza(
            values,
            figsize=(10, 10),
            color_blank_space="same",
            slice_colors=slice_colors,
            value_colors=text_colors,
            value_bck_colors=slice_colors,
            blank_alpha=0.4,
            param_location=110,
            kwargs_slices=dict(edgecolor="white", zorder=2, linewidth=2),
            kwargs_params=dict(color="white", fontsize=12, va="center"),
            kwargs_values=dict(color="white", fontsize=12, zorder=3,
                             bbox=dict(edgecolor="white", facecolor="cornflowerblue",
                                     boxstyle="round,pad=0.2", lw=1))
        )
        
        ax.set_title('Player Percentile Rankings', fontsize=16, color='white', pad=20)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#22312b')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        self._store_visualization('pizza_chart', image_base64)
        return image_base64
    
    def generate_formation_analysis(self) -> str:
        """
        Generate formation analysis visualization
        """
        formations = ['4-3-3', '4-2-3-1', '3-5-2', '4-4-2']
        usage = [45, 30, 15, 10]
        wins = [10, 5, 2, 1]
        draws = [3, 2, 1, 1]
        losses = [2, 1, 1, 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#22312b')
        
        # Formation usage pie chart
        ax1.pie(usage, labels=formations, autopct='%1.1f%%', startangle=90,
               colors=['#00ff00', '#00ff88', '#88ff00', '#88ff88'])
        ax1.set_title('Formation Usage', fontsize=14, color='white')
        
        # Formation performance stacked bar
        x = np.arange(len(formations))
        width = 0.6
        
        ax2.bar(x, wins, width, label='Wins', color='#00ff00')
        ax2.bar(x, draws, width, bottom=wins, label='Draws', color='#ffff00')
        ax2.bar(x, losses, width, bottom=np.array(wins)+np.array(draws), 
               label='Losses', color='#ff0000')
        
        ax2.set_ylabel('Matches', color='white')
        ax2.set_title('Formation Performance', fontsize=14, color='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(formations)
        ax2.legend()
        ax2.tick_params(colors='white')
        
        # Style adjustments
        for ax in [ax1, ax2]:
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='#22312b')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        self._store_visualization('formation_analysis', image_base64)
        return image_base64
    
    def generate_performance_dashboard(self) -> Dict[str, str]:
        """
        Generate a complete performance dashboard with all visualizations
        """
        dashboard = {}
        
        print("Generating heatmap...")
        dashboard['heatmap'] = self.generate_heatmap()
        
        print("Generating shot map...")
        dashboard['shot_map'] = self.generate_shot_map()
        
        print("Generating pass network...")
        dashboard['pass_network'] = self.generate_pass_network()
        
        print("Generating goal distribution...")
        dashboard['goal_distribution'] = self.generate_goal_distribution()
        
        print("Generating formation analysis...")
        dashboard['formation_analysis'] = self.generate_formation_analysis()
        
        # Generate player-specific charts for top player
        if not self.players_data.empty:
            player_id = self.players_data.iloc[0]['id']
            print(f"Generating player radar...")
            dashboard['player_radar'] = self.generate_player_radar(player_id)
            
            print(f"Generating player pizza chart...")
            dashboard['player_pizza'] = self.generate_pizza_chart(player_id)
        
        return dashboard
    
    def _get_player_name(self, player_id: str) -> str:
        """Helper to get player name"""
        if self.players_data is not None and not self.players_data.empty:
            player = self.players_data[self.players_data['id'] == player_id]
            if not player.empty:
                return f"{player.iloc[0]['first_name']} {player.iloc[0]['last_name']}"
        return "Unknown Player"
    
    def _store_visualization(self, viz_type: str, image_base64: str, 
                            plot_json: Optional[str] = None) -> None:
        """
        Store visualization in Supabase for retrieval by Flutter app
        """
        try:
            data = {
                'team_id': self.team_id,
                'visualization_type': viz_type,
                'image_data': image_base64,
                'plot_json': plot_json,
                'created_at': datetime.now().isoformat()
            }
            
            # Check if visualization exists
            existing = self.supabase.table('team_visualizations').select('id').eq(
                'team_id', self.team_id
            ).eq('visualization_type', viz_type).execute()
            
            if existing.data:
                # Update existing
                self.supabase.table('team_visualizations').update(data).eq(
                    'id', existing.data[0]['id']
                ).execute()
            else:
                # Insert new
                self.supabase.table('team_visualizations').insert(data).execute()
                
        except Exception as e:
            print(f"Error storing visualization: {e}")


# Example usage script
def main():
    """
    Example usage of the TeamAnalytics class
    """
    # Initialize with your Supabase credentials
    SUPABASE_URL = "your-supabase-url"
    SUPABASE_KEY = "your-supabase-anon-key"
    TEAM_ID = "your-team-id"
    
    # Create analytics instance
    analytics = TeamAnalytics(SUPABASE_URL, SUPABASE_KEY, TEAM_ID)
    
    # Fetch team data
    print("Fetching team data...")
    team_info = analytics.fetch_team_data(days_range=30)
    print(f"Team data loaded: {team_info}")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    dashboard = analytics.generate_performance_dashboard()
    
    print("\nDashboard generated successfully!")
    print(f"Generated {len(dashboard)} visualizations")
    
    # Save visualizations locally for testing
    for viz_type, data in dashboard.items():
        if viz_type != 'goal_distribution':  # This returns JSON
            # Decode and save image
            image_data = base64.b64decode(data)
            with open(f"{viz_type}.png", "wb") as f:
                f.write(image_data)
            print(f"Saved {viz_type}.png")
    
    return dashboard


if __name__ == "__main__":
    # Run the example
    dashboard = main()