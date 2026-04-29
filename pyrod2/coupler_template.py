import base64
import cv2
import numpy as np
from rect import Rect

class CouplerTemplate:
    def __init__(self, frame_index, rect, template):
        self.frame_index = frame_index
        self.rect = rect
        self.template = template

    def copy(self):
        return CouplerTemplate(self.frame_index, self.rect.copy(), self.template)

    def to_json(self):
        # Encode the template into a Base64 representation of a PNG grayscale image.
        success, buffer = cv2.imencode(".png", self.template)
        assert success
        b64str = base64.b64encode(buffer).decode("utf-8")
        return {
            "f": self.frame_index,
            "r": self.rect.to_json(),
            "t": f"data:image/png;base64,{b64str}",
        }

    @staticmethod
    def from_json(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[CouplerTemplate.from_json] params must be a dict")

        required_keys = {"f", "r", "t"}
        if not required_keys.issubset(params.keys()):
            raise ValueError(f"[CouplerTemplate.from_json] params must contain keys: {required_keys}")

        b64str = params["t"]
        # remove base64 URI header
        if "," in b64str:
            b64str = b64str.split(",")[1]
        buffer = base64.b64decode(b64str)
        arr = np.frombuffer(buffer, np.uint8)
        template = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        return CouplerTemplate(
            int(params["f"]),
            Rect.from_json(params["r"]),
            template,
        )
