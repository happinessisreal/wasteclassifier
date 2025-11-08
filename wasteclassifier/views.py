from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import os
import base64
import google.generativeai as genai  # Import Google library
import mimetypes  # To determine image mime type


def index(request):
    """Renders the main page."""
    return render(request, "index.html")


@csrf_exempt  # Use csrf_exempt for simplicity, consider proper CSRF for production
def analyze_waste(request):
    """Receives image data, analyzes using Gemini, and returns classification."""
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            image_data_url = data.get("image_data")

            if not image_data_url:
                return HttpResponseBadRequest("Missing image data")

            # Extract image data and mime type from data URL
            try:
                # format: "data:image/png;base64,iVBORw0KGgo..."
                header, encoded = image_data_url.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]  # e.g., "image/png"
                image_data_bytes = base64.b64decode(encoded)
            except (ValueError, IndexError):
                return HttpResponseBadRequest("Invalid image data format")

            # --- Google Gemini API Call ---
            try:
                # Configure the Gemini client
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                genai.configure(api_key=api_key)

                # Prepare the image part for the prompt
                image_part = {"mime_type": mime_type, "data": image_data_bytes}

                # Prepare the text part for the prompt
                prompt_text = (
                    "Analyze the object in the image to detect plastic. "
                    "First, determine if it is **'Plastic'** or **'Not Plastic'**. "
                    "If 'Plastic', identify its **Resin Identification Code** (e.g., #1 PET, #2 HDPE, #5 PP). "
                    "Then, based on the code, determine its general recyclability: **'Widely Recyclable'** (typically #1, #2) or **'Check Locally'** (typically #3, #4, #5, #6, #7). "
                    "Provide the main classification (Plastic/Not Plastic) and the specific details (Resin Code, Recyclability)."
                )

                # Select the Gemini model (use a vision-capable model)
                # gemini-1.5-flash is a good choice for speed and capability
                model = genai.GenerativeModel("gemini-2.0-flash")

                # Send prompt with image and text
                response = model.generate_content([prompt_text, image_part])

                # Basic parsing (Gemini response structure might differ slightly)
                analysis_result = response.text
                lines = analysis_result.strip().split("\n", 1)
                classification = lines[0]
                details = lines[1] if len(lines) > 1 else "No further details provided."

                return JsonResponse(
                    {"classification": classification, "details": details}
                )

            except ValueError as ve:  # Catch missing API key error
                print(f"Configuration error: {ve}")
                return JsonResponse({"error": str(ve)}, status=500)
            except Exception as e:
                print(f"Google Gemini API error: {e}")  # Log the error server-side
                # Add more specific error handling for Gemini if needed
                return JsonResponse(
                    {"error": f"Error analyzing image: {str(e)}"}, status=500
                )
            # --- End Google Gemini API Call ---

        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON data")
        except Exception as e:
            print(f"Error in analyze_waste view: {e}")  # Log unexpected errors
            return JsonResponse(
                {"error": "An unexpected server error occurred."}, status=500
            )

    else:
        return HttpResponseBadRequest("Invalid request method. Only POST is allowed.")
