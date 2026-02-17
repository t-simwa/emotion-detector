"""
Emotion detection client using the Watson NLP-based Skills Network API.

This module exposes a single function, `emotion_detector`, which sends
text to the remote emotion-detection service and returns the parsed result
as a Python dictionary.
"""

from typing import Any, Dict

import requests


def emotion_detector(text_to_analyze: str) -> Dict[str, Any]:
    """
    Detect emotions present in the provided text.

    This function sends the given text to the Skills Network hosted
    Watson NLP emotion analysis endpoint and returns a dictionary with
    emotion scores and the dominant emotion.

    Parameters
    ----------
    text_to_analyze : str
        The input text that should be analyzed for emotions.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing emotion scores (anger, disgust, fear,
        joy, sadness) and the dominant emotion (under the key
        'dominant_emotion').
    """
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(url, json=payload, headers=headers, timeout=10)

    # Task 7: handle blank input (status code 400)
    if response.status_code == 400:
        # Return a dictionary with the same keys but all values set to None
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response.raise_for_status()

    # Parse the response
    response_data = response.json()

    # Extract emotion predictions from the response
    # The response structure may vary, but typically contains emotion predictions
    emotions = response_data.get("emotionPredictions", [])

    if emotions and len(emotions) > 0:
        # Get the first emotion prediction
        emotion_prediction = emotions[0].get("emotion", {})

        # Extract individual emotion scores
        anger = emotion_prediction.get("anger", 0)
        disgust = emotion_prediction.get("disgust", 0)
        fear = emotion_prediction.get("fear", 0)
        joy = emotion_prediction.get("joy", 0)
        sadness = emotion_prediction.get("sadness", 0)

        # Determine dominant emotion
        emotion_scores = {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
        }
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        return {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "dominant_emotion": dominant_emotion,
        }

    # Fallback if response structure is different
    return response_data


def format_emotion_output(emotion_dict: Dict[str, Any]) -> str:
    """
    Format the emotion detection output into a readable string.

    Parameters
    ----------
    emotion_dict : Dict[str, Any]
        A dictionary containing emotion scores (anger, disgust, fear,
        joy, sadness) and the dominant emotion.

    Returns
    -------
    str
        A formatted string displaying all emotion scores and the dominant emotion.
    """
    anger = emotion_dict.get("anger")
    disgust = emotion_dict.get("disgust")
    fear = emotion_dict.get("fear")
    joy = emotion_dict.get("joy")
    sadness = emotion_dict.get("sadness")
    dominant_emotion = emotion_dict.get("dominant_emotion")

    formatted_output = (
        f"For the given statement, the system response is "
        f"'anger': {anger}, 'disgust': {disgust}, 'fear': {fear}, "
        f"'joy': {joy} and 'sadness': {sadness}. "
        f"The dominant emotion is {dominant_emotion}."
    )

    return formatted_output


if __name__ == "__main__":
    SAMPLE_TEXT = "I am so happy to work on this exciting project!"
    emotion_result = emotion_detector(SAMPLE_TEXT)
    print(emotion_result)
    print("\nFormatted output:")
    print(format_emotion_output(emotion_result))

