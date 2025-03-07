from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# The URL where your FastAPI app is running
FASTAPI_URL = "http://localhost:8000/predict"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    form_data = {}
    if request.method == 'POST':
        try:
            # For debugging - print raw form data
            print("Raw form data:", dict(request.form))
            
            # Gather the input fields from the form
            form_data = {
                "State": request.form.get("State"),
                "Account length": float(request.form.get("Account_length")),
                "Area code": float(request.form.get("Area_code")),
                "International plan": request.form.get("International_plan"),
                "Voice mail plan": request.form.get("Voice_mail_plan"),
                "Number vmail messages": float(request.form.get("Number_vmail_messages")),
                "Total day minutes": float(request.form.get("Total_day_minutes")),
                "Total day calls": float(request.form.get("Total_day_calls")),
                "Total eve minutes": float(request.form.get("Total_eve_minutes")),
                "Total eve calls": float(request.form.get("Total_eve_calls")),
                "Total night minutes": float(request.form.get("Total_night_minutes")),
                "Total night calls": float(request.form.get("Total_night_calls")),
                "Total intl minutes": float(request.form.get("Total_intl_minutes")),
                "Total intl calls": float(request.form.get("Total_intl_calls")),
                "Customer service calls": float(request.form.get("Customer_service_calls"))
            }

            # For debugging: print the data being sent
            print("Sending data to FastAPI:", form_data)

            # Send a POST request to the FastAPI prediction endpoint
            response = requests.post(FASTAPI_URL, json=form_data)
            
            # For debugging: print the response
            print("Response status code:", response.status_code)
            print("Response content:", response.text)
            
            response.raise_for_status()
            result = response.json()
        except ValueError as ve:
            # Handle form data conversion errors
            result = {"error": f"Invalid form data: {str(ve)}"}
        except requests.exceptions.RequestException as e:
            # Handle connection and HTTP errors
            result = {"error": f"API request error: {str(e)}"}
            print(f"API request error details: {e}")
        except Exception as e:
            result = {"error": f"An error occurred: {str(e)}"}
            print(f"General error details: {e}")
    
    return render_template('index.html', result=result, form=form_data)

if __name__ == '__main__':
    # Run the Flask app on port 5001 (for example)
    app.run(debug=True, port=5001)
