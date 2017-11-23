import pickle

class MyClass():
    def __init__(self):
        self.info = OtherClass(option=1)

    def pickle(self):
        f = file('test_file', 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def unpickle(self):
	with file('test_file', 'rb') as f:
		return pickle.load(f)

class OtherClass():
    def __init__(self, option):
        self.property = option * 2
