# main.py
import os
import io
import base64
import json
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch, PyPizza
import plotly.graph_objects as go
from supabase import create_client, Client

# --- Pydantic Models for Type Safety ---
class AnalyticsRequest(BaseModel):
    team_id: str
    days_range: int = 30 # Default to last 30 days

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Core Analytics Class ---
class TeamAnalyticsGenerator:
    def __init__(self, supabase_client: Client, team_id: str, days_range: int):
        self.supabase = supabase_client
        self.team_id = team_id
        self.days_range = days_range
        self.events_df = None

    def _fetch_event_data(self):
        """Fetches all raw event data for the team within the time range."""
        try:
            rpc_params = {'p_team_id': self.team_id, 'p_days_range': self.days_range}
            response = self.supabase.rpc('get_team_raw_events', params=rpc_params).execute()
            if not response.data:
                self.events_df = pd.DataFrame()
                return
            self.events_df = pd.DataFrame(response.data)
            # Ensure coordinates are numeric
            coord_cols = ['x', 'y', 'end_x', 'end_y']
            self.events_df[coord_cols] = self.events_df[coord_cols].apply(pd.to_numeric, errors='coerce')
        except Exception as e:
            print(f"Error fetching event data: {e}")
            self.events_df = pd.DataFrame()

    def _save_plot_to_base64(self, fig) -> str:
        """Saves a matplotlib figure to a base64 encoded string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), dpi=120)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return image_base64

    def generate_heatmap(self) -> str | None:
        """Generates a professional heatmap of all player actions."""
        if self.events_df.empty or 'x' not in self.events_df.columns: return None
        df = self.events_df.dropna(subset=['x', 'y'])
        if df.empty: return None
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E2A17', line_color='#c7d5cc', line_zorder=2)
        fig, ax = pitch.draw(figsize=(8, 12))
        fig.set_facecolor('#0E2A17')
        pitch.kdeplot(df.x, df.y, ax=ax, fill=True, levels=100, shade_lowest=True, cut=4, cmap='viridis')
        return self._save_plot_to_base64(fig)

    def generate_shot_map(self) -> str | None:
        """Generates a shot map, color-coding goals and misses."""
        df_shots = self.events_df[self.events_df.type_name == 'Shot'].copy()
        if df_shots.empty: return None

        pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E2A17', line_color='#c7d5cc')
        fig, ax = pitch.draw(figsize=(8, 12))
        fig.set_facecolor('#0E2A17')
        
        goals = df_shots[df_shots.outcome_name == 'Goal']
        misses = df_shots[df_shots.outcome_name != 'Goal']
        
        pitch.scatter(misses.x, misses.y, alpha=0.6, s=100, color="#ba4f45", ax=ax, label='Miss/Saved')
        pitch.scatter(goals.x, goals.y, alpha=0.8, s=200, color="#69c37b", ax=ax, label='Goal', marker='football')
        
        ax.legend(facecolor='#0E2A17', edgecolor='None', fontsize=12, labelcolor='white', loc='upper left')
        return self._save_plot_to_base64(fig)

    def generate_goal_distribution_plot(self) -> str | None:
        """Generates an interactive Plotly chart for goal distribution."""
        # This logic should ideally come from a pre-aggregated goals table for performance
        # For this example, we simulate it from events.
        df_goals = self.events_df[self.events_df.type_name == 'Goal'].copy()
        if df_goals.empty: return None
        
        # Goal Timing
        bins = [0, 15, 30, 45, 60, 75, 90, 120]
        labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '90+']
        df_goals['minute_bin'] = pd.cut(df_goals['minute'], bins=bins, labels=labels, right=False)
        timing_data = df_goals['minute_bin'].value_counts().reindex(labels, fill_value=0)
        
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(
            x=timing_data.index,
            y=timing_data.values,
            name='Goals by Time',
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title_text='Goal Distribution by Time',
            xaxis_title="Minute",
            yaxis_title="Number of Goals"
        )
        return fig.to_json()

    def run_all(self):
        """Fetches data and runs all generation methods."""
        self._fetch_event_data()
        
        results = {
            "heatmap": self.generate_heatmap(),
            "shot_map": self.generate_shot_map(),
            "goal_distribution": self.generate_goal_distribution_plot()
            # Add other generators here (pass_network, pizza charts, etc.)
        }
        return results

# --- API Endpoint ---
@app.post("/generate-all-analytics")
async def generate_all_analytics_endpoint(
    request_data: AnalyticsRequest,
    x_internal_secret: str = Header(...)
):
    INTERNAL_SECRET = os.environ.get("PYTHON_API_SECRET")
    if not INTERNAL_SECRET or x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
    supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        analytics_generator = TeamAnalyticsGenerator(
            supabase_client=supabase_admin,
            team_id=request_data.team_id,
            days_range=request_data.days_range
        )
        visualizations = analytics_generator.run_all()
        
        # Cache the results in Supabase
        supabase_admin.table('team_visualizations').upsert({
            'team_id': request_data.team_id,
            'visualizations': visualizations, # Store all results in one JSONB column
            'last_updated': 'now()'
        }, on_conflict='team_id').execute() # Use on_conflict for upsert
        
        return {"status": "success", "message": "Analytics generated and cached."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")