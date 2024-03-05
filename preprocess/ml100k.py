
class ML100k():

    def __init__(self) -> None:
        pass

    def load_data(self, path: str) -> None:

        with open(path, 'r') as f:
            data = f.readlines()
            data = [line.strip().split('\t') for line in data]
            data = [[int(i) for i in line] for line in data]

        self.data = data
        self.num_users = max([line[0] for line in data])
        self.num_items = max([line[1] for line in data])
        self.num_ratings = len(data)


