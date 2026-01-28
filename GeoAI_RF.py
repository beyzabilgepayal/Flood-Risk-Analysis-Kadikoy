# %% [CELL 1] Libraries and settings

import pandas as pd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# This is an optional setting
import warnings
warnings.filterwarnings('ignore')

print("1. cell is done")

# %% [CELL 2] Machine Learning 


# 1. Load the data
df= pd.read_csv(r'C:\myz\Kadikoy_Final_Training_Data.csv')

# 2.Variables
X= df[['kadikoy_grass_filled', 'Kadikoy_TWI_Fiziksel', 'egim_radyan']] 
y= df['label']

# 3. Training
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Random Forest Model
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Showing the results
y_pred= model.predict(X_test)
acc= accuracy_score(y_test, y_pred)
print(f" Accuracy Ratio: %{acc*100:.1f}")

# %% [CELL 3] Graphics and reports

# A. Feature Importance Graph
plt.figure(figsize=(10, 6))
importances= model.feature_importances_
features= ['Elevation (DEM)', 'TWI', 'Slope']
feature_df= pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance')
plt.barh(feature_df['Feature'], feature_df['Importance'], color= ['#e06666', '#ffe599', '#f9cb9c'])
plt.title('Feature Importance Analysis')
plt.savefig('C:/myz/feature_importance_final.png', dpi=300)
plt.show()

# B. Confusion Matrix Graph
cm= confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot= True, fmt= 'd', cmap= 'Blues')
plt.title('Model Classification Performance')
plt.savefig('C:/myz/confusion_matrix_final.png', dpi=300)
plt.show()

# %% [CELL 4] Producing maps

try:
    with rasterio.open(r'C:\myz\dem_align.tif') as dem_src:
        dem =dem_src.read(1)
        meta =dem_src.meta

    with rasterio.open(r'C:\myz\twi_align.tif') as twi_src:
        twi =twi_src.read(1)

    with rasterio.open(r'C:\myz\slope_align.tif') as slope_src:
        slope =slope_src.read(1)

    flat_dem =dem.flatten()
    flat_twi =twi.flatten()
    flat_slope =slope.flatten()
    
    X_map_df =pd.DataFrame({
        'kadikoy_grass_filled': flat_dem,
        'Kadikoy_TWI_Fiziksel': flat_twi,
        'egim_radyan': flat_slope
    })
    
    # Cleaning
    X_map_df= X_map_df.fillna(0)

    # Prediction
    prediction_prob= model.predict_proba(X_map_df)[:, 1]
    flood_map= prediction_prob.reshape(dem.shape)

    meta.update(dtype= rasterio.float32, count=1)
    with rasterio.open(r'C:\myz\Kadikoy_Flood_Risk_Map.tif', 'w', **meta) as dst:
        dst.write(flood_map.astype(rasterio.float32), 1)
        

except Exception as e:
    print(f"Error creating map: {e}")

# %% [CELL 5] Coordinate entry
def risk_questioning(x_utm, y_utm):
    try:
        with rasterio.open(r'C:\myz\dem_align.tif') as src:
            row, col= src.index(x_utm, y_utm)
            val_dem= src.read(1)[row, col]
        
        with rasterio.open(r'C:\myz\twi_align.tif') as src:
            val_twi= src.read(1)[row, col]
            
        with rasterio.open(r'C:\myz\slope_align.tif') as src:
            val_slope= src.read(1)[row, col]

        query_data= pd.DataFrame([[val_dem, val_twi, val_slope]], 
                                  columns=['kadikoy_grass_filled', 'Kadikoy_TWI_Fiziksel', 'egim_radyan'])
        
        possibilty= model.predict_proba(query_data)[0, 1]
        result= "Risky" if possibilty > 0.5 else "Safe"
        
        print("-" * 30)
        print(f"coordinate: {x_utm}, {y_utm}")
        print("-" * 30)
        print(f"Elevation: {val_dem:.2f} m")
        print(f"TWI: {val_twi:.2f}")
        print(f"Slope: {val_slope:.2f}")
        print("-" * 30)
        print(f"Flood Risk: %{possibilty*100:.1f}")
        print(f"Result: {result}")
        print("-" * 30)

    except Exception as e:
        print("Error!")

# You can change the coordinate here:
risk_questioning(674000, 4540000)