import pickle
from ..model.tfl.TFL_man import TflManager


class Controller:
    # initializes everything the controller needs to use
    def __init__(self, data_path):
        with open(data_path, 'r+') as file:
            contents = file.read().split('\n')  # this is pls file
            with open(contents[0], 'rb') as pkl:
                self.frames_data = pickle.load(pkl)
            self.frame_paths = contents[1:]
            model_path = '../data/model.h5'
            self.tfl_manager = TflManager(self.frames_data['principle_point'], self.frames_data['flx'], model_path)

    # runs the program
    def run(self):
        for frame in self.frame_paths:
            self.tfl_manager.run(frame, self.frames_data)


def main():
    controller = Controller("/data/data.pls")
    controller.run()


if __name__ == "__main__":
    main()

