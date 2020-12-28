from .frame_set import TflFrame
import numpy as np
from .calculate_distance import SFM
from .detect_lights import find_lights
from.run_in_CNN.predict_tfl import load_model, run_candidates_in_filter


class TflManager:

    def __init__(self, principal_point, focal_length, model):
        self.pp = principal_point
        self.focal = focal_length
        self.prev_frame = None
        self.model = load_model(model)

    def run(self, frame, frame_data):
        if self.prev_frame is None:
            self.prev_frame = TflFrame(frame, np.eye(4))
            self.find_tfl_candidates(self.prev_frame)
            self.run_in_CNN(self.prev_frame)
            # TODO: return value, visualize
        else:
            EM = frame_data['egomotion_' + str(int(self.prev_frame.frame_path[27:-16])) + '-' + str(int(frame[27:-16]))]
            frame = TflFrame(frame, EM)
            self.find_tfl_candidates(frame)
            self.run_in_CNN(frame)
            self.find_tfl_dist(self.prev_frame, frame)
            self.prev_frame = frame
            # TODO: return value, visualize

    def find_tfl_candidates(self, frame):  # -> candidates, auxiliary (in frame class)
        frame.candidates, frame.aux = find_lights.find_tfl_lights(frame)

    def run_in_CNN(self, frame):  # -> traffic_lights, auxiliary (in frame class)
        run_candidates_in_filter(frame, self.model)

    def find_tfl_dist(self, prev_frame, curr_frame):  # ->
        SFM.calc_TFL_dist(prev_frame, curr_frame, self.focal, self.pp)
