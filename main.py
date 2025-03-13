import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import os


OUTPUT_DIR = "out"
DIGIT_MODEL_PATH = "digit_model.keras"

FOREGROUND = "white"
BACKGROUND = "black"


def draw_number(number):
    img = Image.new("L", (200, 200), BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Font configurations to try in order of preference
    font_configs = [
        ("/System/Library/Fonts/Noteworthy.ttc", 150),  # Mac font
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150),  # Linux font
        ("arial.ttf", 150),  # Windows font
    ]

    font = None
    for font_path, font_size in font_configs:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (IOError, OSError):
            continue

    if font is None:
        font = ImageFont.load_default()

    # Get text size for centering
    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate center position
    x = (200 - text_width) // 2
    y = (200 - text_height) // 2 - (200 * 0.3)  # Shift up by % of height

    # Draw the number centered
    draw.text((x, y), text, fill=FOREGROUND, font=font)

    return img


class Test:
    model: tf.keras.models.Model = None
    image: Image.Image = None

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.model = tf.keras.models.load_model(DIGIT_MODEL_PATH)

    def given_number(self, n):
        self.image = draw_number(n)

    def when_number_is_bitten(self):
        bitten = self.image.copy()
        draw = ImageDraw.Draw(bitten)

        width, height = self.image.size

        # bite position
        center_x = width * 0.35
        center_y = height * 0.50

        # bite radius
        radius = 30

        # take the bite
        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ],
            fill=BACKGROUND,
        )

        self.image = bitten

    def then_the_number_is(self, expected_number):
        # Preprocess image for the model
        img_array = np.array(self.image.resize((28, 28)))
        img_array = img_array.reshape(1, 28, 28, 1) / 255.0

        # Get prediction
        [prediction] = self.model.predict(img_array)
        predicted_number = np.argmax(prediction)
        confidence = prediction[predicted_number]

        assert (
            predicted_number == expected_number
        ), f"Expected {expected_number}, got {predicted_number} with confidence {confidence}"

    def snap(self, filename):
        """Save the current image to disk"""
        self.image.save(os.path.join(OUTPUT_DIR, filename))


def test_number_is_number(test: Test, n):
    test.given_number(n)
    test.snap(f"test_number_is_number_{n}.png")

    test.then_the_number_is(n)


def test_biting_8_makes_3(test: Test):
    """
    Scenario: Biting 8 makes 3
    Given number is 8
    When number is bitten
    Then number is 3
    """
    test.given_number(8)
    test.snap("original_8.png")

    test.when_number_is_bitten()
    test.snap("bitten_8.png")

    test.then_the_number_is(3)


if __name__ == "__main__":
    test = Test()

    for i in range(10):
        test_number_is_number(test, i)

    test_biting_8_makes_3(test)
