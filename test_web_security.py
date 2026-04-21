import io
import os
import unittest
import uuid
import shutil

from PIL import Image
from werkzeug.datastructures import FileStorage

from web_security import (
    ValidationError,
    image_timestamp_from_upload,
    parse_finite_float,
    save_verified_image,
    validate_timestamp,
)


def build_image_upload(image_format, filename):
    image = Image.new("RGB", (8, 8), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    return FileStorage(stream=buffer, filename=filename)


class WebSecurityTests(unittest.TestCase):
    def test_validate_timestamp_accepts_digits(self):
        self.assertEqual(validate_timestamp("1765801316099"), "1765801316099")

    def test_validate_timestamp_rejects_invalid_values(self):
        with self.assertRaises(ValidationError):
            validate_timestamp("../bad")

    def test_parse_finite_float_rejects_non_finite(self):
        with self.assertRaises(ValidationError):
            parse_finite_float("inf", "fx", min_value=0.0)

    def test_image_timestamp_from_upload_requires_jpeg_extension(self):
        upload = build_image_upload("PNG", "1765801316099.png")
        with self.assertRaises(ValidationError):
            image_timestamp_from_upload(upload)

    def test_save_verified_image_normalizes_png_to_jpeg(self):
        upload = build_image_upload("PNG", "sample.png")

        tmpdir = os.path.join(os.getcwd(), f"tmp_test_{uuid.uuid4().hex}")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            target_path = os.path.join(tmpdir, "saved.jpg")
            save_verified_image(upload, target_path)

            with Image.open(target_path) as saved:
                self.assertEqual(saved.format, "JPEG")
                self.assertEqual(saved.size, (8, 8))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
