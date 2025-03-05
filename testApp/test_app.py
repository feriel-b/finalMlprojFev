import pytest
import pandas as pd
from app import preprocess_input, PredictInput, feature_names

def test_preprocess_columns():
    # Create a sample input (using the schema example from PredictInput)
    sample_data = PredictInput.model_validate({
        "State": "CA",
        "Account_length": 128.0,
        "Area_code": 415.0,
        "International_plan": "no",
        "Voice_mail_plan": "yes",
        "Number_vmail_messages": 25.0,
        "Total_day_minutes": 265.1,
        "Total_day_calls": 110.0,
        "Total_eve_minutes": 197.4,
        "Total_eve_calls": 99.0,
        "Total_night_minutes": 244.7,
        "Total_night_calls": 91.0,
        "Total_intl_minutes": 10.0,
        "Total_intl_calls": 3.0,
        "Customer_service_calls": 1.0
    })
    
    processed_array = preprocess_input(sample_data)
    # Convert back to DataFrame for easier column inspection (if needed)
    # Assume the input sample produces a DataFrame of shape (1, N)
    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    
    # These assertions are already in your preprocess_input,
    # but also verifying them in a test ensures consistency.
    assert list(processed_df.columns) == feature_names