# ---------------------------------------------------------------
# Decision class testing
# Author: Rodrigo Alfredo Mendoza
# Last modified: 1/05/2024
# ---------------------------------------------------------------

from model.plagiarism_type import identify_unordered_sentences, identify_voice_change, identify_insert_replace, identify_time_change
from unittest import TestCase

class TestDecisionPlagiarisPercentage(TestCase):

    def setUp(self):
        ...

    def test_unordered(self):
        self.assertTrue(
            identify_unordered_sentences(
                'The sun rose slowly over the horizon, casting a warm glow across the sleepy town. Birds chirped cheerfully, welcoming the new day with their melodious songs. People gradually emerged from their homes, ready to embrace the opportunities that lay ahead.',
                'People gradually emerged from their homes, ready to embrace the opportunities that lay ahead. Birds chirped cheerfully, welcoming the new day with their melodious songs. The sun rose slowly over the horizon, casting a warm glow across the sleepy town.'
            )
        )

    def test_voice(self):
        self.assertTrue(
            identify_voice_change(
                "The chef prepared a delicious meal for the guests.",
                "A delicious meal was prepared for the guests by the chef."
            )
        )

    def test_insert_replace(self):
        self.assertTrue(
            'The city streets bustled with activity as people hurried to their destinations. Car horns blared and pedestrians weaved through the crowded sidewalks. Bright neon signs illuminated the night sky, adding to the vibrant energy of the urban landscape.',
            'The city streets bustled with activity as people hurried to their destinations, eager to reach their homes before the storm hit. Car horns blared and pedestrians weaved through the crowded sidewalks. Bright neon signs illuminated the night sky, adding to the vibrant energy of the urban landscape while reminding everyone of the bustling nightlife.'
        )

    def test_time(self):
        self.assertTrue(
            'The students are studying diligently for their exams next week.',
            'The students studied diligently for their exams last week.'
        )