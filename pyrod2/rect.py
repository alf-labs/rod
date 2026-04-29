
class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def is_empty(self):
        return self.w <= 0 or self.h <= 0

    def copy(self):
        return Rect(self.x, self.y, self.w, self.h)

    def center(self):
        return int(self.x + self.w // 2), int(self.y + self.h // 2)

    def __repr__(self):
        return f"Rect( {self.x:.3f}x{self.y:.3f} + {self.w:.3f} w x{self.h:.3f} h )"
