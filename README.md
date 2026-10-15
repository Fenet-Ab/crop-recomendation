# Ethiopian Crop & Fertilizer Recommendation System

This project recommends **which crop group to grow** on an Ethiopian farm from soil and climate data, then recommends **how much Urea and DAP** to apply for a **specific crop** in that group.

It is a two-stage advisor:

1. **Crop suitability (machine learning)** — XGBoost classifies the farm into one of four crop groups.
2. **Fertilizer package (agronomic rules)** — EIAR / MoA nutrient targets are adjusted using the soil test (N, P, K, pH, organic carbon) and converted into Urea + DAP rates, bag counts, and split application timing.

---

## What the system does

A farmer (or extension worker) enters:

- soil nutrients: N, P, K, Zn, S (mg/kg)
- soil pH, organic carbon, moisture, color, soil type
- climate: temperature, relative humidity, rainfall index, altitude
- region (Oromia, Amhara, Tigray, SNNPRS, Afar)

**Button 1 — Recommend Crop**

The model returns the best crop **group**, confidence, top-3 alternatives, and the individual crops in the group.

| Predicted group | Crops in the group |
|-----------------|--------------------|
| Major_Cereals | Teff, Maize, Wheat, Barley |
| Cereals | Dagussa, Sorghum |
| Pulses | Bean, Pea |
| Specialty | Niger seed, Potato, Red Pepper, Fallow |

**Button 2 — Recommend Fertilizer (Urea + DAP)**

Pick a specific crop (for example Wheat or Teff). The system does **not** guess fertilizer with the crop model. It uses crop-specific nutrient requirements and the same soil numbers:

- **DAP** (18% N, 46% P₂O₅) supplies all phosphorus and some starter nitrogen.
- **Urea** (46% N) supplies the remaining nitrogen.
- Output: kg/ha, 50 kg bags/ha, DAP:Urea proportion, and when to apply (sowing, tillering, etc.).
- Legumes (Bean, Pea) get mostly DAP and little or no Urea, because they fix nitrogen.

---

## How the crop model was trained

Notebooks `01` → `05` are the training pipeline. Run them **in order** from the `notebooks/` folder.

### 1. Data (`notebooks/01_data_exploration.ipynb`)

Source file: `data/Crop_recommendation_ethiopia_real.csv` (~3,867 farms).

Each row is a farm with:

- lab soil: N, P, K, pH, Zn, S, soil color
- NASA MERRA-2 climate: seasonal temperature (T2M), specific humidity (QV2M), rainfall (PRECTOTCORR), surface pressure (PS), soil wetness (GWETTOP)
- original crop grown: Teff, Maize, Wheat, Barley, Bean, Pea, Sorghum, Dagussa, Niger seed, Potato, Red Pepper, Fallow

### 2. Feature engineering (`notebooks/02_feature_engineering.ipynb`)

Raw columns are turned into model features:

| Feature | How it is built |
|---------|-----------------|
| N, P, K, Zn, S, ph | Soil lab values. **N is multiplied by 1000** so it is mg/kg like P and K. |
| temperature | Mean of seasonal T2M min/max |
| humidity | Relative humidity **%**, from QV2M + temperature + pressure (not raw humidity × 1000) |
| rainfall | Sum of seasonal PRECTOTCORR |
| altitude_m | Barometric formula from surface pressure. **PS is treated as kPa** (`× 10` to hPa), so altitude is ~1,500–2,600 m, not ~33 km |
| soil_moisture | GWETTOP (0–1) |
| soilcolor_* | One-hot dummies (black, brown, gray, red, other) kept in the feature list |

Processed table is saved as `data/processed/engineered.csv`.

### 3. Labels, scaling, split (`notebooks/04_preprocessing_and_split.ipynb`)

Rare crops are merged so the classifier has usable classes (see table above). Rare groups can be oversampled with light noise.

Then:

- `LabelEncoder` turns group names into integers → `models/label_encoder_merged.pkl`
- `StandardScaler` standardizes numeric features → `models/scaler_merged.pkl`
- stratified split: **70% train / 15% validation / 15% test**

### 4. Training (`notebooks/05_model_training.ipynb`)

Several classifiers were compared (logistic regression, decision tree, random forest, XGBoost). **XGBoost** won on validation F1.

Because classes are imbalanced (Major_Cereals is much larger than Pulses):

- **SMOTE** oversamples minority classes **inside the training folds only**
- pipeline: `SMOTE → XGBoost`
- hyperparameters tuned with `RandomizedSearchCV` and stratified k-fold, scoring **F1-macro**

The chosen pipeline is saved as:

```text
models/best_crop_model.pkl
```

Held-out test performance (from the training notebook): about **80% accuracy**, **~0.68 macro F1**. Pulses remain the hardest class because there are fewer samples.

---

## How a prediction works at runtime

When you click **Recommend Crop**:

1. The web form sends soil + climate JSON to `POST /predict`.
2. FastAPI builds one feature row in the **same order** the scaler was trained on.
3. `scaler.transform(...)` standardizes the row.
4. `model.predict_proba(...)` returns a probability for each crop group.
5. The group with the highest probability is the recommendation, plus top-3 alternatives.
6. For every crop in that group, a Urea + DAP package is pre-computed from the soil test.

When you click **Recommend Fertilizer**:

1. The form sends the **specific crop name** plus soil N, P, K, pH, SOC to `POST /recommend-crop-fertilizer`.
2. `src/fertilizer_service.py` looks up EIAR-style N and P targets for that crop (Maize needs more N than Teff; beans need P, not full Urea).
3. Targets are scaled down if soil N or P is already high, or up if soil is deficient. Acid pH triggers a lime warning.
4. Elemental N and P are converted:

```text
DAP  kg/ha  =  P_needed / 0.201          (DAP is 20.1% elemental P)
N from DAP  =  DAP_kg × 0.18
Urea kg/ha  =  (N_needed − N from DAP) / 0.46
bags        =  kg / 50
```

5. Urea is split across growth stages (for example wheat: 50% at sowing with all DAP, 30% at tillering, 20% at stem elongation).

The crop model and the fertilizer engine are **separate**. Changing fertilizer rates does not change the crop class; the crop class only tells you **which crop’s nutrient table** to use.

---

## Project layout

```text
crop_recommendation/
├── data/                          # local datasets (not pushed to GitHub)
│   ├── Crop_recommendation_ethiopia_real.csv
│   └── processed/                 # engineered features and train/val/test splits
├── models/                        # local .pkl files (not pushed to GitHub)
│   ├── best_crop_model.pkl        # XGBoost + SMOTE
│   ├── scaler_merged.pkl
│   └── label_encoder_merged.pkl
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_visualizations.ipynb
│   ├── 04_preprocessing_and_split.ipynb
│   └── 05_model_training.ipynb
├── src/
│   ├── app.py                     # FastAPI: crop + fertilizer APIs
│   └── fertilizer_service.py      # Urea + DAP agronomic engine
├── static/index.html              # web UI
└── requirements.txt
```

`venv/`, `data/`, and `models/` are gitignored. Keep trained `.pkl` files on your machine (or copy them into `models/` after training).

---

## How to run

### 1. Environment

```bash
cd "/home/fenet/Downloads/Telegram Desktop/crop_recommendation (3)"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If the virtual environment already exists:

```bash
source venv/bin/activate
```

### 2. Data and models

You need:

- `data/Crop_recommendation_ethiopia_real.csv` (to retrain)
- `models/best_crop_model.pkl`
- `models/scaler_merged.pkl`
- `models/label_encoder_merged.pkl`

To retrain from scratch, open Jupyter from the project venv and run notebooks **01 → 02 → 04 → 05** (03 is visualization only).

```bash
jupyter notebook notebooks/
```

### 3. Start the web app

Always start from the **project root** (so `static/` and `models/` resolve correctly):

```bash
source venv/bin/activate
# Local dev:
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

# On Render (or other PaaS) set the service start command to use the
# platform-provided $PORT. You can use the included helper script:
bash start.sh
```

Open: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

That redirects to the advisor UI.

### 4. Use the UI

1. Choose region (or a quick preset: Oromia, Amhara, Tigray, SNNPRS).
2. Fill soil and climate, or keep the preset values.
3. Click **Recommend Crop**.
4. Click a crop chip (Wheat, Teff, …) or **Recommend Fertilizer (Urea + DAP)**.
5. Read DAP vs Urea kg/ha, 50 kg bags, proportion, and split timing.
6. In the **SMS panel**, select a farmer (Amharic or Afan Oromo), preview the message, then send.

SMS uses the **same crop + DAP/Urea numbers** as the fertilizer result. The farmer’s `language_pref` (`am` / `om`) chooses the template. Broadcast rebuilds a new message per farmer so each person gets their own language.

Put these in `.env` (never commit the real values):

```
SMS_API_TOKEN=your_smsethiopia_key
DATABASE_URL=postgresql://postgres:...@...pooler.supabase.com:6543/postgres
```

Optional one-time seed:

```bash
python -m src.seed
```

### 5. Useful API routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Redirects to the UI |
| GET | `/health` | Confirms models loaded |
| POST | `/predict` | Crop group + fertilizer packages for crops in that group |
| POST | `/recommend-crop-fertilizer` | Urea + DAP for one named crop and soil test |
| GET | `/docs` | Interactive Swagger documentation |
| GET | `/farmers/list` | Registered farmers (name, phone, language) |
| POST | `/farmers/` | Register a farmer |
| POST | `/farmers/preview-sms` | Localized SMS text from crop + DAP/Urea |
| POST | `/farmers/send-sms` | Send via SMSEthiopia and log in Supabase |
| GET | `/farmers/sms-logs` | Recent delivery log |

---

## Notes

- Fertilizer rates are **decision support**, not a legal prescription. Local extension advice, rainfall, and variety still matter.
- Do not commit `venv/` or large `.pkl` / `.csv` files to GitHub (100 MB file limit).
- If `/predict` reports a feature-count mismatch, re-run notebooks 02 → 04 → 05 so the scaler and model match the current feature list.
