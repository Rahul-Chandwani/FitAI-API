from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and tokenizer
MODEL_PATH = os.getenv('MODEL_PATH', './fitness_ai_model_enhanced')
print(f"Loading model from {MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using fallback responses...")
    tokenizer = None
    model = None

def generate_response(prompt, max_length=512):
    """Generate response using the trained model"""
    if model is None or tokenizer is None:
        return generate_fallback_response(prompt)
    
    try:
        formatted_prompt = f"""### Instruction:
{prompt}

### Response:
"""
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response part
        response = response.split("### Response:")[-1].strip()
        
        # Try to parse as JSON if it looks like JSON
        if response.startswith('{') or response.startswith('['):
            try:
                return json.loads(response)
            except:
                pass
        
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt):
    """Fallback responses when model is not available"""
    prompt_lower = prompt.lower()
    
    if 'workout' in prompt_lower:
        return {
            "workout": {
                "id": "fallback_workout",
                "name": "Basic Workout",
                "duration": "45 min",
                "total_calories": 400,
                "exercises": [
                    {
                        "id": "ex_001",
                        "name": "Push-ups",
                        "sets": 3,
                        "reps": "12",
                        "unit": "reps",
                        "tutorial": "Standard push-up form",
                        "form_tips": ["Keep back straight", "Full range of motion"],
                        "level": "intermediate"
                    }
                ]
            }
        }
    elif 'diet' in prompt_lower or 'meal' in prompt_lower:
        return {
            "meals": [
                {
                    "name": "Breakfast",
                    "type": "Breakfast",
                    "food": "Oatmeal with fruits",
                    "calories": 400,
                    "protein": 15,
                    "carbs": 60,
                    "fats": 10
                }
            ]
        }
    else:
        return {
            "response": "I'm here to help with your fitness journey. Please try again or contact support."
        }

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "Fitness AI API is running",
        "model_loaded": model is not None,
        "endpoints": {
            "/generate_plan": "POST - Generate workout/diet/training plans or chat responses"
        }
    })

@app.route('/generate_plan', methods=['POST'])
def generate_plan():
    """Main endpoint for all AI generation"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        request_type = data.get('type')
        
        if not request_type:
            return jsonify({"error": "Type field is required"}), 400
        
        # Handle different request types
        if request_type == 'workout':
            return handle_workout_request(data)
        elif request_type == 'diet':
            return handle_diet_request(data)
        elif request_type == 'training_plan':
            return handle_training_plan_request(data)
        elif request_type == 'chat':
            return handle_chat_request(data)
        else:
            return jsonify({"error": f"Unknown type: {request_type}"}), 400
            
    except Exception as e:
        print(f"Error in generate_plan: {e}")
        return jsonify({"error": str(e)}), 500

def handle_workout_request(data):
    """Handle workout generation request"""
    preferences = data.get('preferences', {})
    
    muscles = preferences.get('target_muscles', ['Full Body'])
    duration = preferences.get('duration_minutes', 45)
    level = preferences.get('experience_level', 'intermediate')
    location = preferences.get('location', 'gym')
    
    prompt = f"""Generate a {level} {', '.join(muscles)} workout for {location} lasting {duration} minutes.
Target muscles: {', '.join(muscles)}
Experience level: {level}
Location: {location}
Duration: {duration} minutes"""
    
    response = generate_response(prompt)
    
    # Ensure response is in correct format
    if isinstance(response, dict) and 'workout' in response:
        return jsonify(response)
    elif isinstance(response, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(response)
            return jsonify(parsed)
        except:
            pass
    
    # Return fallback
    return jsonify(generate_fallback_response(prompt))

def handle_diet_request(data):
    """Handle diet generation request"""
    preferences = data.get('preferences', {})
    
    diet_pref = preferences.get('diet_preference', 'Balanced')
    aim = preferences.get('diet_aim', 'High Protein')
    calories = preferences.get('target_calories', 2000)
    
    prompt = f"""Generate a {diet_pref} diet plan for {aim} with {calories} calories.
Dietary preference: {diet_pref}
Goal: {aim}
Target calories: {calories}"""
    
    response = generate_response(prompt)
    
    # Ensure response is in correct format
    if isinstance(response, dict) and 'meals' in response:
        return jsonify(response)
    elif isinstance(response, str):
        try:
            parsed = json.loads(response)
            return jsonify(parsed)
        except:
            pass
    
    return jsonify(generate_fallback_response(prompt))

def handle_training_plan_request(data):
    """Handle comprehensive training plan request"""
    user_profile = data.get('user_profile', {})
    preferences = data.get('preferences', {})
    
    goal = user_profile.get('goal', 'general_fitness')
    duration = preferences.get('duration_weeks', 8)
    days_per_week = preferences.get('days_per_week', 5)
    
    prompt = f"""Generate a {duration}-week training plan for {goal}.
Goal: {goal}
Duration: {duration} weeks
Training days: {days_per_week} per week"""
    
    response = generate_response(prompt, max_length=1024)
    
    # For training plans, we might need to construct a more complex response
    if isinstance(response, dict) and 'plan' in response:
        return jsonify(response)
    
    # Return a basic structure
    return jsonify({
        "plan": {
            "plan_id": f"plan_{goal}_{duration}w",
            "duration": duration,
            "target_macros": {
                "protein": 150,
                "carbs": 200,
                "fats": 67
            },
            "weeks": []  # Would be populated by model
        }
    })

def handle_chat_request(data):
    """Handle chat/conversation request"""
    message = data.get('message', '')
    context = data.get('context', {})
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    # Add context to prompt if available
    context_str = ""
    if context:
        user_profile = context.get('user_profile', {})
        if user_profile:
            context_str = f"\nUser context: Goal - {user_profile.get('goal', 'fitness')}"
    
    prompt = f"{message}{context_str}"
    
    response = generate_response(prompt, max_length=300)
    
    # Ensure response is a string for chat
    if isinstance(response, dict):
        response = json.dumps(response)
    
    return jsonify({"response": str(response)})

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
