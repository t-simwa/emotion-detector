"""
Unit tests for the emotion_detector function.

These tests call the live Watson NLP Skills Network API and verify
that the detected dominant emotion matches expectations for a few
representative input sentences.
"""

import unittest

from emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Unit tests for the emotion_detector function."""

    def test_joy_sentence(self):
        """Text expressing happiness should have dominant emotion 'joy'."""
        text = "I am glad this happened"
        result = emotion_detector(text)
        self.assertEqual(result.get("dominant_emotion"), "joy")

    def test_anger_sentence(self):
        """Text expressing anger should have dominant emotion 'anger'."""
        text = "I am really mad about this"
        result = emotion_detector(text)
        self.assertEqual(result.get("dominant_emotion"), "anger")

    def test_sadness_sentence(self):
        """Text expressing sadness should have dominant emotion 'sadness'."""
        text = "I feel sad and disappointed"
        result = emotion_detector(text)
        self.assertEqual(result.get("dominant_emotion"), "sadness")

    def test_fear_sentence(self):
        """Text expressing fear should have dominant emotion 'fear'."""
        text = "I am scared about the future"
        result = emotion_detector(text)
        self.assertEqual(result.get("dominant_emotion"), "fear")


if __name__ == "__main__":
    unittest.main()

