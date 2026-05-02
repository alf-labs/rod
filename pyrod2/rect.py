from point import Point

class Rect:
    """A rectangle with integer coordinates."""
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def is_empty(self):
        return self.w <= 0 or self.h <= 0

    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)

    def move_to(self, x, y):
        self.x = x
        self.y = y

    def move_by(self, dx, dy):
        self.x += dx
        self.y += dy

    def recenter_to(self, cx, cy):
        ocx = int(self.x + self.w // 2)
        ocy = int(self.y + self.h // 2)
        dx = cx - ocx
        dy = cy - ocy
        self.x += dx
        self.y += dy

    def scale_by(self, sx, sy):
        if sx != 1.0:
            cx = self.x + self.w // 2
            nw = self.w = int(self.w * sx)
            self.x = cx - nw // 2
        if sy != 1.0:
            cy = self.y + self.h // 2
            nh = self.h = int(self.h * sy)
            self.y = cy - nh // 2

    def center(self):
        return int(self.x + self.w // 2), int(self.y + self.h // 2)

    def centerPoint(self):
        return Point(int(self.x + self.w // 2), int(self.y + self.h // 2))

    def __repr__(self):
        return f"Rect( {self.x:.3f} x {self.y:.3f} + {self.w:.3f} w x {self.h:.3f} h )"

    def to_json(self):
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
        }

    @staticmethod
    def from_json(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[Rect.from_json] params must be a dict")

        required_keys = {"x", "y", "w", "h"}
        if not required_keys.issubset(params.keys()):
            raise ValueError(f"[Rect.from_json] params must contain keys: {required_keys}")

        for key in required_keys:
            if not isinstance(params[key], (int, )):
                raise ValueError(f"[Rect.from_json] params['{key}'] must be int")

        return Rect(
            params["x"],
            params["y"],
            params["w"],
            params["h"],
        )
