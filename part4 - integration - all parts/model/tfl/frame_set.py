import matplotlib.pyplot as plt


class Frame:
    def __init__(self, path):
        self.frame = plt.imread(path)
        self.frame_path = path


class TflFrame(Frame):
    def __init__(self, frame, EM):
        super().__init__(frame)
        self.traffic_lights = []
        self.candidates = []
        self.aux = []
        self.EM_mat = EM
        self.tfl_dist = []
