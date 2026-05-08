import numpy as np
from point import Point


class RodResult:
    def __init__(self, frame_index, initial_center, y_top, y_bottom, poly_c, poly_w):
        self.frame_index = frame_index
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.initial_center = initial_center
        self.poly_c = poly_c
        self.poly_w = poly_w

    def to_json(self):
        return {
            "f": self.frame_index,
            "yt": self.y_top,
            "yb": self.y_bottom,
            "ic": self.initial_center,
            "pc": RodResult.poly_to_json(self.poly_c),
            "pw": RodResult.poly_to_json(self.poly_w),
        }

    @staticmethod
    def from_json(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[CouplerResult.from_json] params must be a dict")

        keys = params.keys()
        required_keys = {"f", "yt", "yb", "ic", "pc", "pw"}
        if not required_keys.issubset(keys):
            raise ValueError(f"[CouplerResult.from_json] params must contain keys: {required_keys}, but was: {keys}")

        return RodResult(
            frame_index = int(params["f"]),
            initial_center = int(params["ic"]),
            y_top = int(params["yt"]),
            y_bottom = int(params["yb"]),
            poly_c = RodResult.poly_from_json(params["pc"]),
            poly_w = RodResult.poly_from_json(params["pw"]),
        )

    def __repr__(self):
        return f"RodResult( [{self.frame_index:04d}] Y {self.y_top}-{self.y_bottom}, center {self.initial_center}, {self.poly_c}, W {self.poly_w} )"

    @staticmethod
    def poly_to_json(poly):
        return {
            "c": poly.coef.tolist(),      # Convert numpy array to list
            "d": poly.domain.tolist(),
            "w": poly.window.tolist()
        }

    @staticmethod
    def poly_from_json(json):
        return np.polynomial.Polynomial(
            coef   = json["c"],
            domain = json["d"],
            window = json["w"]
        )

