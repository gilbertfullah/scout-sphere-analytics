# main.py
import os
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io
from supabase import create_client, Client

# --- Pydantic Model for Request Body Validation ---
class AnalyticsRequest(BaseModel):
    team_id: str

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Main Analytics Logic (adapted from your previous script) ---
def generate_visualizations(supabase: Client, team_id: str):
    """
    Connects to Supabase, fetches event data, generates visualizations,
    and returns their public URLs.
    """
    print(f"Starting visualization generation for team: {team_id}")

    try:
        # 1. Fetch Raw Event Data
        rpc_params = {'p_team_id': team_id, 'p_match_limit': 5}
        response = supabase.rpc('get_team_event_data', params=rpc_params).execute()
        if not response.data:
            print(f"No event data for team {team_id}.")
            # Return empty URLs if no data
            return {
                "heatmap_url": None,
                "shotmap_url": None,
                "pass_network_url": None
            }
        
        df_events = pd.DataFrame(response.data)
        coord_cols = ['x', 'y', 'end_x', 'end_y']
        df_events[coord_cols] = df_events[coord_cols].apply(pd.to_numeric, errors='coerce')

    except Exception as e:
        print(f"Error fetching event data: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    # 2. Generate and Upload Visualizations
    heatmap_url = _generate_heatmap(df_events.dropna(subset=['x', 'y']), team_id, supabase)
    shotmap_url = _generate_shot_map(df_events.dropna(subset=['x', 'y']), team_id, supabase)
    # Add other visualization calls here if needed

    # 3. Return the URLs
    return {
        "heatmap_url": heatmap_url,
        "shotmap_url": shotmap_url,
        "pass_network_url": None # Placeholder
    }

def _save_and_upload_plot(fig, team_id: str, asset_name: str, supabase: Client) -> str | None:
    """Helper to save, upload, and return a plot's URL."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor(), dpi=150)
        buf.seek(0)
        image_bytes = buf.read()
        plt.close(fig)

        storage_path = f'analytics/{team_id}/{asset_name}.png'
        supabase.storage.from_('team_analytics_assets').upload(
            path=storage_path, file=image_bytes, file_options={"content-type": "image/png", "upsert": "true"}
        )
        print(f"Successfully uploaded {asset_name}")
        return supabase.storage.from_('team_analytics_assets').get_public_url(storage_path)
    except Exception as e:
        print(f"Error saving/uploading {asset_name}: {e}")
        plt.close(fig)
        return None

def _generate_heatmap(df: pd.DataFrame, team_id: str, supabase: Client):
    """Generates a heatmap of all player actions."""
    if df.empty: return None
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#0E2A17', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(6.8, 10.4))
    fig.set_facecolor('#0E2A17')
    pitch.kdeplot(df.x, df.y, ax=ax, shade=True, levels=100, shade_lowest=True, cut=4, cmap='viridis')
    return _save_and_upload_plot(fig, team_id, 'heatmap_latest', supabase)

def _generate_shot_map(df: pd.DataFrame, team_id: str, supabase: Client):
    """Generates a shot map, color-coding goals and misses."""
    df_shots = df[df.type_name == 'Shot'].copy()
    if df_shots.empty: return None
    # ... (implementation from previous answer)
    return _save_and_upload_plot(fig, team_id, 'shotmap_latest', supabase)


# --- API Endpoint ---
@app.post("/generate-analytics")
async def generate_analytics_endpoint(
    request_data: AnalyticsRequest,
    # This Header dependency provides security. The Edge Function must send this header.
    x_internal_secret: str = Header(...) 
):
    # 1. **Security Check**
    # Get the secret from environment variables set on Render.com
    INTERNAL_SECRET = os.environ.get("PYTHON_API_SECRET")
    if not INTERNAL_SECRET or x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid secret")

    # 2. **Initialize Supabase Admin Client**
    # Use environment variables for Supabase credentials
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase credentials not configured")
        
    supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # 3. **Run Analytics Logic**
    try:
        result_urls = generate_visualizations(supabase_admin, request_data.team_id)
        
        # 4. **Update the Cache Table**
        supabase_admin.table('team_analytics_cache').upsert({
            'team_id': request_data.team_id,
            'heatmap_url': result_urls.get('heatmap_url'),
            'shotmap_url': result_urls.get('shotmap_url'),
            'last_updated': 'now()'
        }).execute()
        
        return {"status": "success", "message": "Analytics generated and cached.", "urls": result_urls}
        
    except Exception as e:
        # Catch any errors during the process and return a detailed server error
        raise HTTPException(status_code=500, detail=str(e))