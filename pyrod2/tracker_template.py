from rect import Rect

class TrackerTemplate:
    def __init__(self, frame_index, rect, template):
        self.frame_index = frame_index
        self.rect = rect
        self.template = template

    def copy(self):
        return TrackerTemplate(self.frame_index, self.rect.copy(), self.template)
