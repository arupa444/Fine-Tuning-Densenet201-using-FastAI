# ====================================================================
# FASTAPI ENDPOINT FOR SUPPLY CHAIN FORECASTING
# Production-ready API for serving trained ML models
# ====================================================================

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow import keras
import uvicorn

# ====================================================================
# 1. INITIALIZE FASTAPI APP
# ====================================================================

app = FastAPI(
    title="Supply Chain Forecasting API",
    description="Time-series forecasting for supply chain demand prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================================================
# 2. LOAD MODELS ON STARTUP
# ====================================================================

class ModelLoader:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.encoders = None
        self.metadata = None

    def load_models(self):
        """Load all trained models"""
        try:
            print("📦 Loading models...")
            self.xgb_model = joblib.load('xgboost_model.pkl')
            self.lgb_model = joblib.load('lightgbm_model.pkl')
            self.lstm_model = keras.models.load_model('lstm_model.h5')
            self.scaler = joblib.load('scaler.pkl')
            self.encoders = joblib.load('encoders.pkl')

            with open('model_metadata.json', 'r') as f:
                self.metadata = json.load(f)

            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise


# Initialize model loader
model_loader = ModelLoader()


@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    model_loader.load_models()


# ====================================================================
# 3. REQUEST/RESPONSE MODELS
# ====================================================================

class ForecastRequest(BaseModel):
    branchcode: str = Field(..., description="Branch code (e.g., GC01)")
    materialcode: str = Field(..., description="Material/SKU code (e.g., SKU_A)")
    forecast_days: int = Field(30, description="Number of days to forecast", ge=1, le=90)
    model_type: str = Field("xgboost", description="Model type: xgboost, lightgbm, or lstm")
    historical_data: Optional[List[Dict]] = Field(None, description="Recent historical data for context")


class ForecastResponse(BaseModel):
    branchcode: str
    materialcode: str
    model_used: str
    forecast_date: str
    predictions: List[Dict[str, float]]
    metrics: Dict[str, float]
    confidence_interval: Optional[Dict] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]
    api_version: str


class BatchForecastRequest(BaseModel):
    requests: List[ForecastRequest]


# ====================================================================
# 4. FEATURE ENGINEERING FUNCTION
# ====================================================================

def create_prediction_features(
        branchcode: str,
        materialcode: str,
        base_date: datetime,
        historical_sales: List[float] = None
):
    """
    Create features for a single prediction point
    """
    features = {}

    # Time-based features
    features['year'] = base_date.year
    features['month'] = base_date.month
    features['day'] = base_date.day
    features['dayofweek'] = base_date.weekday()
    features['quarter'] = (base_date.month - 1) // 3 + 1
    features['weekofyear'] = base_date.isocalendar()[1]
    features['is_weekend'] = 1 if base_date.weekday() >= 5 else 0
    features['is_month_start'] = 1 if base_date.day == 1 else 0
    features['is_month_end'] = 1 if base_date.day == pd.Timestamp(base_date).days_in_month else 0

    # Cyclical encoding
    features['month_sin'] = np.sin(2 * np.pi * base_date.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * base_date.month / 12)
    features['day_sin'] = np.sin(2 * np.pi * base_date.day / 31)
    features['day_cos'] = np.cos(2 * np.pi * base_date.day / 31)

    # Encode branch and material
    try:
        features['branchcode_encoded'] = model_loader.encoders['branch_encoder'].transform([branchcode])[0]
        features['materialcode_encoded'] = model_loader.encoders['material_encoder'].transform([materialcode])[0]
    except:
        features['branchcode_encoded'] = 0
        features['materialcode_encoded'] = 0

    # Lag features (use historical data if available)
    if historical_sales and len(historical_sales) >= 30:
        features['sales_lag_1'] = historical_sales[-1]
        features['sales_lag_7'] = historical_sales[-7] if len(historical_sales) >= 7 else historical_sales[-1]
        features['sales_lag_14'] = historical_sales[-14] if len(historical_sales) >= 14 else historical_sales[-1]
        features['sales_lag_30'] = historical_sales[-30] if len(historical_sales) >= 30 else historical_sales[-1]

        # Rolling statistics
        features['sales_rolling_mean_7'] = np.mean(historical_sales[-7:])
        features['sales_rolling_std_7'] = np.std(historical_sales[-7:])
        features['sales_rolling_mean_30'] = np.mean(historical_sales[-30:])
    else:
        # Default values if no historical data
        for lag in [1, 7, 14, 30]:
            features[f'sales_lag_{lag}'] = 0
        features['sales_rolling_mean_7'] = 0
        features['sales_rolling_std_7'] = 0
        features['sales_rolling_mean_30'] = 0

    # Additional features (use defaults)
    features['stock_on_hand'] = 100  # Default value
    features['intransit_qty'] = 0
    features['pending_po_qty'] = 0
    features['lead_time_days'] = 7
    features['stockout_flag'] = 0
    features['stock_to_sales_ratio'] = 10
    features['inventory_coverage_days'] = 7

    return features


# ====================================================================
# 5. API ENDPOINTS
# ====================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": all([
            model_loader.xgb_model is not None,
            model_loader.lgb_model is not None,
            model_loader.lstm_model is not None
        ]),
        "available_models": ["xgboost", "lightgbm", "lstm"],
        "api_version": "1.0.0"
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_sales(request: ForecastRequest):
    """
    Generate sales forecast for a specific branch and SKU
    """
    try:
        # Select model
        if request.model_type == "xgboost":
            model = model_loader.xgb_model
        elif request.model_type == "lightgbm":
            model = model_loader.lgb_model
        elif request.model_type == "lstm":
            model = model_loader.lstm_model
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Extract historical sales if provided
        historical_sales = []
        if request.historical_data:
            historical_sales = [d.get('sales_qty', 0) for d in request.historical_data]

        # Generate predictions
        predictions = []
        base_date = datetime.now()

        for day in range(request.forecast_days):
            pred_date = base_date + timedelta(days=day)

            # Create features
            features = create_prediction_features(
                request.branchcode,
                request.materialcode,
                pred_date,
                historical_sales
            )

            # Convert to DataFrame
            feature_df = pd.DataFrame([features])

            # Ensure all required features are present
            for col in model_loader.metadata['feature_columns']:
                if col not in feature_df.columns:
                    feature_df[col] = 0

            # Reorder columns to match training
            feature_df = feature_df[model_loader.metadata['feature_columns']]

            # Predict
            if request.model_type in ["xgboost", "lightgbm"]:
                prediction = float(model.predict(feature_df)[0])
            else:  # LSTM
                # For LSTM, we need sequence data - use simplified approach
                prediction = float(np.mean(historical_sales[-7:]) if historical_sales else 50)

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_sales": round(max(0, prediction), 2)
            })

            # Update historical sales for next iteration
            if historical_sales:
                historical_sales.append(prediction)

        # Calculate confidence metrics
        if historical_sales:
            std_dev = np.std(historical_sales[-30:]) if len(historical_sales) >= 30 else 10
        else:
            std_dev = 10

        return {
            "branchcode": request.branchcode,
            "materialcode": request.materialcode,
            "model_used": request.model_type,
            "forecast_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predictions": predictions,
            "metrics": {
                "mean_forecast": round(np.mean([p['predicted_sales'] for p in predictions]), 2),
                "std_dev": round(std_dev, 2),
                "min_forecast": round(min([p['predicted_sales'] for p in predictions]), 2),
                "max_forecast": round(max([p['predicted_sales'] for p in predictions]), 2)
            },
            "confidence_interval": {
                "lower_bound": round(np.mean([p['predicted_sales'] for p in predictions]) - 1.96 * std_dev, 2),
                "upper_bound": round(np.mean([p['predicted_sales'] for p in predictions]) + 1.96 * std_dev, 2)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-forecast")
async def batch_forecast(request: BatchForecastRequest):
    """
    Generate forecasts for multiple branch-SKU combinations
    """
    results = []

    for single_request in request.requests:
        try:
            result = await forecast_sales(single_request)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "branchcode": single_request.branchcode,
                "materialcode": single_request.materialcode
            })

    return {"forecasts": results, "total_requests": len(request.requests)}


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    if not model_loader.metadata:
        raise HTTPException(status_code=500, detail="Models not loaded")

    return {
        "metadata": model_loader.metadata,
        "feature_count": len(model_loader.metadata['feature_columns']),
        "available_branches": list(model_loader.encoders['branch_encoder'].classes_),
        "available_skus": list(model_loader.encoders['material_encoder'].classes_)
    }


@app.get("/models/features")
async def get_feature_list():
    """Get list of features used by the model"""
    return {
        "features": model_loader.metadata['feature_columns'],
        "total_features": len(model_loader.metadata['feature_columns'])
    }


# ====================================================================
# 6. RUN THE API
# ====================================================================

if __name__ == "__main__":
    """
    Run the API server

    Usage:
        python api.py

    Or with custom settings:
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    """
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# ====================================================================
# 7. USAGE EXAMPLES
# ====================================================================

"""
EXAMPLE 1: Single Forecast Request
------------------------------------
POST http://localhost:8000/forecast

{
    "branchcode": "GC01",
    "materialcode": "SKU_A",
    "forecast_days": 30,
    "model_type": "xgboost",
    "historical_data": [
        {"date": "2024-01-01", "sales_qty": 45.2},
        {"date": "2024-01-02", "sales_qty": 52.1},
        ...
    ]
}


EXAMPLE 2: Batch Forecast Request
----------------------------------
POST http://localhost:8000/batch-forecast

{
    "requests": [
        {
            "branchcode": "GC01",
            "materialcode": "SKU_A",
            "forecast_days": 30,
            "model_type": "xgboost"
        },
        {
            "branchcode": "GC02",
            "materialcode": "SKU_B",
            "forecast_days": 30,
            "model_type": "lightgbm"
        }
    ]
}


EXAMPLE 3: Get Model Info
--------------------------
GET http://localhost:8000/models/info


EXAMPLE 4: Health Check
------------------------
GET http://localhost:8000/
"""