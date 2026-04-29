
class Point:
    """A Point with integer or float coordinates."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def copy(self):
        return Point(self.x, self.y)

    def __repr__(self):
        return f"Point( {self.x:.3f} x {self.y:.3f} )"

    def to_json(self):
        return {
            "x": self.x,
            "y": self.y,
        }

    @staticmethod
    def from_json(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[Point.from_json] params must be a dict")

        required_keys = {"x", "y"}
        if not required_keys.issubset(params.keys()):
            raise ValueError(f"[Point.from_json] params must contain keys: {required_keys}")

        for key in required_keys:
            if not isinstance(params[key], (int, float)):
                raise ValueError(f"[Point.from_json] params['{key}'] must be int or float")

        return Point(
            params["x"],
            params["y"],
        )
