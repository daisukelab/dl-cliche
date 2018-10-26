# Based on https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
import tables
import numpy as np

class BigH5Array():
    def __init__(self, filename, shape=None, atom=tables.Float32Atom()):
        self.filename = filename
        self.shape = shape
        self.atom = atom
    def open_for_write(self):
        self.f = tables.open_file(self.filename, mode='w')
        self.array_c = self.f.create_carray(self.f.root, 'carray', self.atom, self.shape)
    def open_for_write_expandable(self):
        self.f = tables.open_file(self.filename, mode='w')
        self.array_e = self.f.create_earray(self.f.root, 'data', self.atom, [0] + list(self.shape[1:]))
    def open_for_read(self):
        self.f = tables.open_file(self.filename, mode='r')
    def data(self): # for expandable
        # bigarray.data()[1:10,2:20]
        return self.f.root.data
    def append(self, row_data): # for expandable
        self.array_e.append(row_data)
    def __call__(self): # for random access
        return self.f.root.carray
    def close(self):
        self.f.close()

def big_h5_load(filename):
    bigfile = BigH5Array(filename)
    bigfile.open_for_read()
    bigarray = np.array(bigfile())
    bigfile.close()
    return bigarray
