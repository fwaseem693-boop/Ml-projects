import gradio as gr
import joblib
import pandas as pd

# Model Load
model = joblib.load('pubg_rank_model.pkl')
feature_names = joblib.load('features_list.pkl')

print("✅ Model Loaded Successfully!")

def predict_rank(kills, damageDealt, walkDistance, rideDistance, weaponsAcquired, 
                 heals, boosts, headshotKills, longestKill, swimDistance, 
                 teamKills, roadKills, vehicleDestroys):

    total_distance = walkDistance + rideDistance
    kill_efficiency = kills / (damageDealt + 1)

    data = {
        'kills': kills,
        'damageDealt': damageDealt,
        'walkDistance': walkDistance,
        'rideDistance': rideDistance,
        'weaponsAcquired': weaponsAcquired,
        'heals': heals,
        'boosts': boosts,
        'headshotKills': headshotKills,
        'longestKill': longestKill,
        'swimDistance': swimDistance,
        'teamKills': teamKills,
        'roadKills': roadKills,
        'vehicleDestroys': vehicleDestroys,
        'total_distance': total_distance,
        'kill_efficiency': kill_efficiency
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(df)[0]
    win_perc = round(pred * 100, 2)

    if win_perc >= 85:
        msg = "🏆 **CHICKEN DINNER!** God Level Performance 🔥"
    elif win_perc >= 70:
        msg = "🔥 Bahut Strong Performance! Top ranks pakke."
    elif win_perc >= 50:
        msg = "✅ Accha Khela! Average se bohat better."
    else:
        msg = "💪 Practice karo, agla match mein better rank aayega."

    return f"**{win_perc}%** Win Place Percentage\n\n{msg}"


# Gradio Interface
with gr.Blocks(title="PUBG Rank Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 PUBG Win Place Predictor")
    gr.Markdown("### Apne Stats Daalo aur Rank Prediction dekho")

    with gr.Row():
        with gr.Column():
            kills = gr.Slider(0, 50, value=5, label="Kills", step=1)
            damageDealt = gr.Slider(0, 8000, value=1500, label="Damage Dealt", step=10)
            walkDistance = gr.Slider(0, 20000, value=3000, label="Walk Distance", step=50)
            rideDistance = gr.Slider(0, 20000, value=2000, label="Ride Distance", step=50)
            weaponsAcquired = gr.Slider(0, 50, value=8, label="Weapons Acquired", step=1)

        with gr.Column():
            heals = gr.Slider(0, 40, value=6, label="Heals", step=1)
            boosts = gr.Slider(0, 40, value=7, label="Boosts", step=1)
            headshotKills = gr.Slider(0, 20, value=2, label="Headshot Kills", step=1)
            longestKill = gr.Slider(0, 1000, value=200, label="Longest Kill", step=5)
            swimDistance = gr.Slider(0, 5000, value=0, label="Swim Distance", step=10)

    with gr.Row():
        teamKills = gr.Slider(0, 15, value=0, label="Team Kills", step=1)
        roadKills = gr.Slider(0, 10, value=0, label="Road Kills", step=1)
        vehicleDestroys = gr.Slider(0, 10, value=0, label="Vehicle Destroys", step=1)

    btn = gr.Button("🔥 Predict My Rank", variant="primary", size="large")
    result = gr.Markdown()

    btn.click(predict_rank, 
              inputs=[kills, damageDealt, walkDistance, rideDistance, weaponsAcquired,
                      heals, boosts, headshotKills, longestKill, swimDistance,
                      teamKills, roadKills, vehicleDestroys],
              outputs=result)

demo.launch()