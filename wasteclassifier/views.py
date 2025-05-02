from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import os
import base64
from openai import OpenAI

def index(request):
    """Renders the main page."""
    return render(request, 'index.html')

@csrf_exempt # Use csrf_exempt for simplicity, consider proper CSRF for production
def analyze_waste(request):
    """Receives image data, analyzes using OpenAI, and returns classification."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data_url = data.get('image_data')

            if not image_data_url:
                return HttpResponseBadRequest("Missing image data")

            # Extract base64 data
            # format: "data:image/png;base64,iVBORw0KGgo..."
            try:
                header, encoded = image_data_url.split(",", 1)
                # You might want to validate the header (e.g., header == 'data:image/png;base64')
                # image_data = base64.b64decode(encoded)
            except ValueError:
                return HttpResponseBadRequest("Invalid image data format")

            # --- OpenAI API Call --- 
            try:
                client = OpenAI(
                    base_url="https://models.github.ai/inference",
                    api_key=os.environ["GITHUB_PAT"],
)

                prompt_text = ( 
                    "Analyze the object in the image and classify it as waste. "
                    "First, determine if it is 'Recyclable' or 'Non-Recyclable'. "
                    "If Recyclable, briefly explain how it can be recycled (2-3 lines). "
                    "If Non-Recyclable, determine if it is suitable for 'Energy Generation' or 'Coprocessing'. "
                    "Provide the main classification (Recyclable/Non-Recyclable) and the specific detail." 
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini", # Using o4-mini as requested
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url,
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=4000,
                )

                analysis_result = response.choices[0].message.content

                # Basic parsing (you might want more robust parsing)
                lines = analysis_result.strip().split('\n', 1)
                classification = lines[0]
                details = lines[1] if len(lines) > 1 else "No further details provided."

                return JsonResponse({'classification': classification, 'details': details})

            except Exception as e:
                print(f"OpenAI API error: {e}") # Log the error server-side
                return JsonResponse({'error': f'Error analyzing image: {str(e)}'}, status=500)
            # --- End OpenAI API Call ---

        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON data")
        except Exception as e:
            print(f"Error in analyze_waste view: {e}") # Log unexpected errors
            return JsonResponse({'error': 'An unexpected server error occurred.'}, status=500)

    else:
        return HttpResponseBadRequest("Invalid request method. Only POST is allowed.")
