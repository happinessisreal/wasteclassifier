document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const capturedImage = document.getElementById('captured-image');
    const captureButton = document.getElementById('capture-btn');
    const classificationResult = document.getElementById('classification');
    const detailsResult = document.getElementById('details');
    const resultsDiv = document.getElementById('results');

    let stream = null;

    // Access the webcam
    async function initWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            console.log("Webcam access granted.");
            // Ensure video is visible initially
            video.style.display = 'block';
            capturedImage.style.display = 'none';
        } catch (err) {
            console.error("Error accessing webcam: ", err);
            classificationResult.textContent = 'Error accessing webcam. Please ensure permissions are granted.';
        }
    }

    // Capture image from webcam
    captureButton.addEventListener('click', () => {
        if (!stream) {
            console.error("Webcam stream not available.");
            classificationResult.textContent = 'Webcam not available.';
            return;
        }

        // Draw the current video frame onto the canvas
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Get the image data from the canvas
        const imageDataUrl = canvas.toDataURL('image/png');

        // Display the captured image (optional)
        capturedImage.src = imageDataUrl;
        capturedImage.style.display = 'block';
        video.style.display = 'none'; // Hide video feed after capture

        console.log("Image captured. Sending to backend...");
        classificationResult.textContent = 'Analyzing... Please wait.';
        detailsResult.textContent = '';

        // Send image data to the backend
        sendImageToBackend(imageDataUrl);
    });

    // Function to send image data to the backend
    async function sendImageToBackend(imageDataUrl) {
        try {
            const response = await fetch('/analyze/', { // Use the correct endpoint
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 'X-CSRFToken': getCookie('csrftoken') // Uncomment if CSRF protection is enabled on the view
                },
                body: JSON.stringify({ image_data: imageDataUrl })
            });

            if (!response.ok) {
                // Handle HTTP errors (e.g., 400, 500)
                const errorData = await response.json().catch(() => ({ error: 'Unknown server error' })); // Try to parse error, fallback
                throw new Error(`Server error: ${response.status} ${response.statusText}. ${errorData.error || ''}`);
            }

            const data = await response.json();

            if (data.error) {
                // Handle application-level errors returned in JSON
                throw new Error(data.error);
            }

            // Display results
            classificationResult.textContent = `Classification: ${data.classification}`;
            detailsResult.textContent = `Details: ${data.details}`;

        } catch (error) {
            console.error('Error sending image or processing response:', error);
            classificationResult.textContent = 'Analysis failed.';
            detailsResult.textContent = `Error: ${error.message}`;
        } finally {
            // Optionally re-enable the capture button or reset the view
            // video.style.display = 'block'; // Show video again
            // capturedImage.style.display = 'none';
        }
    }

    // Helper function to get CSRF token (needed for POST requests in Django)
    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Initialize webcam on page load
    initWebcam();
});
