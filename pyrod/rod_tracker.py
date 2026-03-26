import numpy as np

class RodTrack:
    def __init__(self, rod, track_id):
        self.id = track_id
        self.rod = rod
        self.hits = 1
        self.misses = 0
        self.scores = [ rod.score ]
        self.active = False # Becomes true after reaching hit threshold
        self._last_temporal_score = 0

    def temporal_score(self):
        """0 for 1 hit, closer to scores' average over time."""
        temp = np.mean(self.scores) * np.log(self.hits)
        self._last_temporal_score = temp
        return temp

    def __repr__(self):
        return f"RodTrack( {self.id:3d} : hits {self.hits:3d}, miss {self.misses:3d}, temporal score {self._last_temporal_score:.3f} -> {self.rod} )"


class TemporalRodTracker:
    def __init__(self, iou_threshold=0.4, min_hits=10, max_misses=3):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.max_misses = max_misses

    def _compute_iou(self, rod1, rod2):
        # Compute IoU between two intervals
        intersection = max(0, min(rod1.right, rod2.right) - max(rod1.left, rod2.left))
        union = (rod1.right - rod1.left) + (rod2.right - rod2.left) - intersection
        return intersection / union if union > 0 else 0

    def update(self, candidates):
        matched_indices = set()

        # 1. Try to match new candidates to existing tracks
        for track in self.tracks:
            best_iou = 0
            best_cand_idx = -1

            for i, rod in enumerate(candidates):
                if i in matched_indices:
                    continue
                iou = self._compute_iou(track.rod, rod)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_cand_idx = i

            if best_cand_idx != -1:
                # Update existing track (Simple EMA for smoothing)
                rod = candidates[best_cand_idx]
                track.rod.merge(rod)
                track.hits += 1
                track.misses = 0
                track.scores.append(rod.score)
                matched_indices.add(best_cand_idx)
            else:
                track.misses += 1

        # 2. Create new tracks for unmatched candidates
        for i, cand in enumerate(candidates):
            if i not in matched_indices:
                self.tracks.append(RodTrack(cand, self.next_id))
                self.next_id += 1

        # 3. Prune dead tracks and filter for output
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        # Return only 'stable' tracks
        stable_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        stable_tracks.sort(
            key=lambda t: t.temporal_score(),
            reverse=True
        )
        return stable_tracks
