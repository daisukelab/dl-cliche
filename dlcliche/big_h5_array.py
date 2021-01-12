# Based on https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
import tables
import numpy as np
import h5py


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


class H5VarLenStorage:
    """Variable length HDF5 database storage helper class.

    Wrapped access to write/read variable length data are provided.
    Multi dimentional (ndim > 2) is automatically reshaped to 2 dimentions
    when storing with `put()` and unfolded with `get()`.

    Requirement:
    - Data shall have variable length in the last dimention of its shape.

    Attributes:
        f (h5py.File): HDF5 file object for direct access.
    """

    def __init__(self, file_name, mode='r', verbose=False):
        self.f = h5py.File(file_name, mode)
        self.count = {}
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()

    @staticmethod
    def _is_str(data):
        return type(data) == str or type(data) != np.ndarray

    def set_dataset(self, key, num_items, example):
        if self._is_str(example):
            dt = h5py.string_dtype()
            shape = (num_items,)
            attr_shape = []
        else:
            dt = h5py.vlen_dtype(example.dtype)
            shape = (num_items, np.prod(example.shape[:-1]))
            attr_shape = list(example.shape[:-1])
        if self.verbose:
            print(f'key={key} stores data with shape={shape + (-1,)}')
        self.f.create_dataset(key, shape=shape, dtype=dt)
        self.f[key].attrs['shape'] = attr_shape
        self.count[key] = 0

    def set_attr(self, key, data):
        self.f.attrs[key] =  data

    def shape(self, key):
        return self.f[key].attrs['shape']

    def put(self, key, data):
        if not self._is_str(data):
            shape = data.shape[:-1]
            assert np.all(self.shape(key) == shape), f'putting variable shape　{shape}　is not compatible with definition {self.shape(key)}.'
            data = data.reshape((np.prod(shape), -1))
        self.f[key][self.count[key]] = data
        self.count[key] += 1

    def close(self):
        self.f.close()

    def attr(self, key):
        return self.f.attrs[key]

    def get(self, key, index):
        var = self.f[key][index]
        if self._is_str(var):
            return var
        var = np.array(list(self.f[key][index]))
        shape = list(self.shape(key)) + [-1]
        return var.reshape(shape)

    def __repr__(self):
        format_string = self.__class__.__name__ + '\n'
        format_string += '\n'.join([f'  [{k}] shape={self.shape(k)} count={self.count[k]}' for k in self.count])
        return format_string
