"""
Flask web application for emotion detection.

This server provides a web interface for the emotion detection application,
allowing users to analyze text for emotions through a web browser.
"""

from flask import Flask, render_template, request
from EmotionDetection import emotion_detector, format_emotion_output

app = Flask("Emotion Detector")


@app.route("/emotionDetector")
def emotion_detector_route():
    """
    Handle emotion detection requests from the web interface.

    Gets text from query parameter 'textToAnalyze', processes it through
    the emotion detection function, and returns formatted output.

    Returns
    -------
    str
        Formatted emotion analysis result as HTML string.
    """
    text_to_analyze = request.args.get("textToAnalyze")

    if not text_to_analyze:
        # Blank input from user
        return "Invalid text! Please try again!"

    # Get emotion detection result
    response = emotion_detector(text_to_analyze)

    # Task 7: handle case where dominant_emotion is None
    if response.get("dominant_emotion") is None:
        return "Invalid text! Please try again!"

    # Format the output
    formatted_response = format_emotion_output(response)

    return formatted_response


@app.route("/")
def render_index_page():
    """
    Render the main index page.

    Returns
    -------
    str
        Rendered HTML template for the emotion detection interface.
    """
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
