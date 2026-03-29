
class Rod:
    def __init__(self, left, right, score, frame=0, tunnel_metric=1.0):
        self.left = left
        self.right = right
        self.score = score
        self.frame = frame
        self.tunnel_metric = tunnel_metric

    def merge(self, other_rod, weight=0.75):
        wa = weight
        wb = 1 - weight
        self.left = self.left * wa + other_rod.left * wb
        self.right = self.right * wa + other_rod.right * wb
        self.score = other_rod.score
        self.frame = other_rod.frame
        self.tunnel_metric = other_rod.tunnel_metric

    def apply_scale(self, scale):
        if scale != 1:
            self.left = self.left * scale
            self.right = self.right * scale

    def center(self):
        return (self.left + self.right) / 2

    def width(self):
        return self.right - self.left

    def iou(self, rod2):
        """Computes IoU (Intersection over Union) between this and the other rod"""
        rod1 = self
        intersection = max(0, min(rod1.right, rod2.right) - max(rod1.left, rod2.left))
        union = (rod1.right - rod1.left) + (rod2.right - rod2.left) - intersection
        return intersection / union if union > 0 else 0

    def __repr__(self):
        return f"Rod( {self.left:.3f} -> {self.right:.3f} ; width {self.width():.3f} ; score {self.score:.3f} ; tunnel {self.tunnel_metric:.3f} )"

    def isTunnel(self):
        return self.tunnel_metric < 1.0

    def dupAtFrame(self, frame, tunnel_metric=None):
        if tunnel_metric is None:
            tunnel_metric = self.tunnel_metric
        return Rod(self.left, self.right, self.score, frame, tunnel_metric)

    def dupInterpolateTo(self, frame, toRod, tunnel_metric=None):
        if tunnel_metric is None:
            tunnel_metric = self.tunnel_metric
        delta_f = toRod.frame - self.frame
        if delta_f == 0:    # should not happen
            return self.dupAtFrame(frame)
        pb = (frame - self.frame) / delta_f
        pa = 1 - pb
        return Rod(
            self.left  * pa + toRod.left * pb,
            self.right * pa + toRod.right * pb,
            round(self.score * pa + toRod.score * pb, 2),
            frame,
            tunnel_metric)

    def toJson(self):
        return {
            "l": self.left,
            "r": self.right,
            "s": self.score,
            "f": self.frame,
            "t": self.tunnel_metric,
        }

    @staticmethod
    def fromJson(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[Rod.fromJson] params must be a dict")

        required_keys = {"l", "r", "s", "f", "t"}
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
            params["t"],
        )
