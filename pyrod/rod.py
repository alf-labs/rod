
class Rod:
    def __init__(self, left, right, score, frame=0):
        self.left = left
        self.right = right
        self.score = score
        self.frame = frame

    def merge(self, other_rod, weight=0.75):
        wa = weight
        wb = 1 - weight
        self.left = self.left * wa + other_rod.left * wb
        self.right = self.right * wa + other_rod.right * wb
        self.score = other_rod.score
        self.frame = other_rod.frame

    def center(self):
        return (self.left + self.right) / 2

    def width(self):
        return self.right - self.left

    def __repr__(self):
        return f"Rod( {self.left:.3f} -> {self.right:.3f} ; width {self.width():.3f} ; score {self.score:.3f} )"

    def dupAtFrame(self, frame):
        return Rod(self.left, self.right, self.score, frame)

    def dupInterpolateTo(self, frame, toRod):
        delta_f = toRod.frame - self.frame
        if delta_f == 0:    # should not happen
            return self.dupAtFrame(frame)
        pb = (frame - self.frame) / delta_f
        pa = 1 - pb
        return Rod(
            self.left  * pa + toRod.left * pb,
            self.right * pa + toRod.right * pb,
            round(self.score * pa + toRod.score * pb, 2),
            frame)

    def toJson(self):
        return {
            "l": self.left,
            "r": self.right,
            "s": self.score,
            "f": self.frame,
        }

    @staticmethod
    def fromJson(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[Rod.fromJson] params must be a dict")

        required_keys = {"l", "r", "s", "f"}
        if not required_keys.issubset(params.keys()):
            raise ValueError(f"[Rod.fromJson] params must contain keys: {required_keys}")

        for key in required_keys:
            if not isinstance(params[key], (int, float)):
                raise ValueError(f"[Rod.fromJson] params['{key}'] must be int or float")

        return Rod(
            params["l"],
            params["r"],
            params["s"],
            params["f"],
        )
