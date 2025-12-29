import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ==============================================================================
# 1. DATA SETUP
# ==============================================================================
data = {
    'Model': [
        'CNN', 'Transformer', 'LSTM', 'CNN-BiLSTM\n(Proposed)', 
        'Bi-GRU', 'GRU', 'Bi-LSTM', 'XGBoost', 'CNN-BiLSTM (Attn)'
    ],
    'RMSE': [30.19, 23.18, 22.06, 20.25, 20.26, 20.62, 21.17, 21.47, 21.89]
}

df = pd.DataFrame(data)
df = df.sort_values(by='RMSE', ascending=False)

# ==============================================================================
# 2. PLOTTING SETUP
# ==============================================================================
sns.set_style("whitegrid")
plt.rcParams.update({'font.family': 'serif'})

# Keep figure large so there is plenty of space
plt.figure(figsize=(14, 8), dpi=300)

# Colors
colors = ['#2ca02c' if 'Proposed' in x else '#B0B0B0' for x in df['Model']]
bars = plt.bar(df['Model'], df['RMSE'], color=colors, edgecolor='black', width=0.6)

# ==============================================================================
# 3. ADD NUMBERS (SMALLER FONT)
# ==============================================================================
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2., 
        height + 0.5, 
        f'{height:.2f}', 
        ha='center', va='bottom', 
        fontsize=10,       # <--- REDUCED SIZE (was 12)
        fontweight='bold', 
        color='black'
    )

# ==============================================================================
# 4. LABELS & FORMATTING
# ==============================================================================
plt.title('Fig 3. Ablation Study: RMSE Comparison (Lower is Better)', fontweight='bold', pad=25)
plt.ylabel('RMSE Value', labelpad=15, fontsize=14)
plt.xlabel('Model Architecture', labelpad=15, fontsize=14)

plt.ylim(0, 35)

# X-Axis Labels (SMALLER FONT)
# fontsize=10 makes the names smaller
plt.xticks(rotation=45, ha='right', fontsize=10) # <--- REDUCED SIZE

sns.despine()

# ==============================================================================
# 5. MARGINS & SAVE
# ==============================================================================
plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.30)

plt.savefig('Fig3_SmallFonts.png', dpi=300, bbox_inches='tight')
print("✅ Figure 3 Saved with smaller fonts.")
plt.show()