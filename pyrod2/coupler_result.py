from point import Point

class CouplerResult:
    def __init__(self, frame_index, center, quality, coupler_ref):
        # couple_ref is the CouplerTemplate.frame_index value.
        self.frame_index = frame_index
        self.center = center
        self.quality = quality
        self.coupler_ref = coupler_ref

    def to_json(self):
        return {
            "f": self.frame_index,
            "r": self.coupler_ref,
            "q": self.quality,
            "c": self.center.to_json(),
        }

    @staticmethod
    def from_json(params):
        # Validate params is a dict with expected fields and types
        if not isinstance(params, dict):
            raise ValueError("[CouplerResult.from_json] params must be a dict")

        keys = params.keys()
        required_keys = {"f", "r", "q", "c"}
        if not required_keys.issubset(keys):
            raise ValueError(f"[CouplerResult.from_json] params must contain keys: {required_keys}, but was: {keys}")

        return CouplerResult(
            frame_index = int(params["f"]),
            center = Point.from_json(params["c"]),
            quality = float(params["q"]),
            coupler_ref = int(params["r"]),
        )

    def __repr__(self):
        return f"CouplerResult( [{self.frame_index:.3f}] {self.center}, qual {self.quality:.3f}, ref {self.coupler_ref:4d} )"
