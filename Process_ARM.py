import xarray as xr
import matplotlib.pyplot as plt
import pathlib

class ProcessARM:
    def __init__(self, path) -> None:
        self.files = [str(f) for f in path.glob('*')]  # get files from path
        self.files.sort()
    
    def read_files(self):
        ds=xr.open_mfdataset(self.files)

