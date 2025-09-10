import itertools, math, pickle, requests, os
from copy import copy
import numpy as np
from pathlib import Path
from typing import Tuple
from pprint import pprint
import logging

def confLogger(logger, level):
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []
    formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

log = logging.getLogger(__name__)

# confLogger(log, logging.DEBUG)
confLogger(log, logging.INFO)

class MockConfig:

    def __init__(self, cfg: dict, num_warps=1, num_stages=1):
        self.cfg = cfg
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __repr__(self):
        return f"<Config {self.cfg}, warps={self.num_warps}, stages={self.num_stages}>"


class KernelOracle:
    # feel free to subclass by defining these class variables only:
    common_sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # GPU constraints
    # the idea here is: we should have a subclass per gfx
    # alternative: we can pass the device properties in the ctor
    n_max_grid = 65536
    threads_per_warp = 64
    min_allowed_warps = 1 # per block
    max_allowed_warps = math.inf # per block # .. eh we already define min & max warps in num_warps_try
    num_warps_try = [1, 2, 4, 8]
    num_stages_try = [1, 2, 4]
    # parameters space ~ < 9000
    # model weights address:
    GITHUB_URL = 'https://github.com/AmdSampsa/kernelOracleWeights/raw/refs/heads/master/mi350_playground.pkl'
    LOCAL_MODEL_PATH = "/tmp/mi350_playground.pkl"
    configClass = MockConfig

    def __init__(self, size_hints: dict, kernel_name: str, path_to_model = None, fetch_model = True):
        self.size_hints = size_hints
        self.original_size_hints = size_hints.copy()
        """
        kernels full decorators has this structure

            ::

                size_hints={'y': 33554432, 'x': 16},
                filename=__file__,
                triton_meta={'signature':..,  'device': DeviceProperties(type='hip', ..., max_threads_per_multi_processor=2048, warp_size=64), 
                    'constants': {}, 'configs': }
                inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', ..}

        we use explicitly here just size_hints and kernel_name, but we could use more..

        """
        self.kernel_name = kernel_name
        self.nametag = 0 # a single number describing the flavour of this kernel. loaded from model metadata
        if (path_to_model is None) and fetch_model: # no path defined, user wants to use remote model data
            if not os.path.exists(self.LOCAL_MODEL_PATH):
                print('Model not found locally. Fetching from GitHub...')
                if self.fetch_model_from_github():
                    self.path = self.LOCAL_MODEL_PATH
                else: 
                    self.path = None
            else:
                print("Found local model file", self.LOCAL_MODEL_PATH)
                self.path = self.LOCAL_MODEL_PATH
        elif path_to_model: # user wants to use a local copy defined by him/herself
            self.path = path_to_model
        else: # no local model, dont fetch remote model (debugging mode)
            self.path = None
        
        if self.path is not None:
            self.__load_model_with_metadata__() # populates self.model and self.meta
            self.nametag = self.generate_filetag(self.kernel_name)
        else:
            self.model = None # indicates we could not load the model or user just want to debug
            self.meta = {}

        # for the ML model we want all dimensions
        for dim in ['x', 'y', 'z']:
            if dim not in size_hints:
                self.size_hints[dim] = 1

    @classmethod
    def installPkgsIf(cls):
        try:
            import xgboost, sklearn, tabulate, pandas
        except ModuleNotFoundError as e:
            log.error("Modules missing, will install some: '{}'", e)
            os.system("pip install --no-input pandas scikit-learn xgboost tabulate")
        else:
            log.debug("all packages installed allright")

    @classmethod 
    def canDo(cls, size_hints: dict, kernel_name: str):
        """Check if we can use this optimizer for the kernel in question
        """
        if "poi" not in kernel_name:
            return False # this is only for POI kernels
        # Define the allowed keys
        allowed_keys = {"x", "y", "z"} # only for xyz/block
        # Check if all keys in the dictionary are within the allowed keys
        if set(size_hints.keys()).issubset(allowed_keys):
            return True
        else:
            return False

    @classmethod
    def tab_transform_to_powers_of_two(cls, df):
        # Create a copy of the DataFrame to avoid modifying the original one
        transformed_df = df.copy()

        # List of columns to transform
        columns_to_transform = ['XBLOCK', 'YBLOCK', 'ZBLOCK', 'xnumel', 'ynumel', 'znumel']
        
        # Transform each specified column to powers of 2
        for col in columns_to_transform:
            transformed_df[col] = 2 ** transformed_df[col].round()
            # transformed_df[col] is np.float64.
        
        return transformed_df


    def __bool__(self):
        return self.model is not None


    def __str__(self):
        st = f"<{self.__class__.__name__} / {self.size_hints} / {self.kernel_name} -> {self.nametag} / path: {str(self.path)}"
        if self.model is None:
            st += " / DEFUNCT"
        st += ">"
        return st


    # Function to download the model file from GitHub
    def fetch_model_from_github(self):
        """Fetch model binary from GitHub and save to local path."""
        # os.makedirs(os.path.dirname(self.LOCAL_MODEL_PATH), exist_ok=True)  # Ensure directories exist
        try:
            response = requests.get(self.GITHUB_URL)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

            with open(self.LOCAL_MODEL_PATH, 'wb') as f:
                f.write(response.content)
            log.debug(f'Model downloaded and saved to {self.LOCAL_MODEL_PATH}')
            return True
        
        except requests.exceptions.HTTPError as http_err:
            log.error(f'HTTP error occurred: {http_err}')  # Log HTTP errors
        except requests.exceptions.ConnectionError as conn_err:
            log.error(f'Connection error occurred: {conn_err}')  # Log connection errors
        except requests.exceptions.Timeout as timeout_err:
            log.error(f'Timeout error occurred: {timeout_err}')  # Log timeout errors
        except Exception as err:
            log.error(f'An error occurred: {err}')  # Log any other errors

    def __load_model_with_metadata__(self):
        try:
            with open(self.path, 'rb') as file:
                loaded_model_with_metadata = pickle.load(file)
            # Access the model and metadata
            self.model = loaded_model_with_metadata["model"]
            self.meta = loaded_model_with_metadata["meta"]
        except Exception as e:
            log.error(f'Could not load the model, reason: {e}')
            self.model = None
            self.meta = {}


    def generate_filetag(self, filename):
        """kernel type to number mapping
        """
        mapper = self.meta["filename2tag"]
        for key in mapper:
            if key in filename:
                return mapper[key]
        return 0

    @classmethod
    def validate_config(cls, block_sizes: Tuple[int, ...], numel_sizes, num_warps: int) -> bool:
        """Basic validation of a config
        
        block_sizes: (xblock, yblock, zblock)
        numel_sizes: (xnumel, ynumel, znumel)
        """
        # TODO: not really sure how to limit total number of threads
        # .. as triton doesn't deal with threads at all, but just with the number of parallelization
        log.debug('validate config %s %s', block_sizes, num_warps)
        min_allowed_threads  = cls.min_allowed_warps*cls.threads_per_warp
        max_allowed_threads = cls.max_allowed_warps*cls.threads_per_warp
        total_threads = 1 # or in fact, total number of elements in a block
        for block_size in block_sizes:
            total_threads *= block_size # blocksize = 1 has no effect..
        # basic sanity check: blocksize <= numel
        for block_size, numel_size in zip(block_sizes, numel_sizes):
            if block_size > numel_size:
                log.debug('block_size > numel_size')
                return False
        # Basic thread count checks
        if total_threads > max_allowed_threads:
            log.debug('max threads exceeded')
            return False
        if total_threads < min_allowed_threads:
            log.debug('min threads underflow')
            return False
        # Warp alignment
        if total_threads % num_warps != 0:
            log.debug('warps not aligned')
            return False
        # Work per warp
        # there's num_warps warps per block.  do the elements/threads distribute equally into each warp?
        work_per_warp = total_threads // num_warps
        if work_per_warp < cls.threads_per_warp:
            log.debug('bad work distribution among warps')
            return False
        # Grid size checks
        for block_size, numel_size in zip(block_sizes, numel_sizes):
            grid_size = (numel_size + block_size - 1) // block_size # (1+1-1)//1 = 1 -> i.e. numel=1 & block_size=1 -> no effect
            if grid_size > cls.n_max_grid:
                log.debug('max grid size exceeded: numel_size %s, block_size %s, grid_size %s', numel_size, block_size, grid_size)
                return False
        log.debug('config OK')
        return True


    @classmethod
    def genConfigs(cls, xnumel=1, ynumel=1, znumel=1, nametag=0):
        """Generate a set of configurations and return them as pandas dataframe
        """
        import pandas as pd
        numel_sizes = (xnumel, ynumel, znumel)
        # normalize
        xnumel_log2 = np.log2(xnumel)
        ynumel_log2 = np.log2(ynumel)
        znumel_log2 = np.log2(znumel)
        data = []
        # Loop through the different values for XBLOCK, YBLOCK, num_warps, and num_stages
        for XBLOCK in cls.common_sizes:
            for YBLOCK in cls.common_sizes:
                for ZBLOCK in cls.common_sizes:
                    for num_warps in cls.num_warps_try:
                        for num_stages in cls.num_stages_try:
                            if cls.validate_config(((XBLOCK, YBLOCK, ZBLOCK)), numel_sizes, num_warps):
                                # Append the current configuration as a dictionary to the data list
                                data.append({
                                    'name': nametag,
                                    'XBLOCK': np.log2(XBLOCK),    
                                    'YBLOCK': np.log2(YBLOCK),    
                                    'ZBLOCK': np.log2(ZBLOCK),    
                                    'num_warps': num_warps,  
                                    'num_stages': num_stages, 
                                    'xnumel': xnumel_log2,    
                                    'ynumel': ynumel_log2,   
                                    'znumel': znumel_log2     
                                })
        # Convert the list of dictionaries into a DataFrame
        return pd.DataFrame(data)

    
    def rankConfigs(self, nmax):
        """Generate configs and rank them using the ML regressor.  Return the best configs.
        """
        input_df = self.genConfigs(
            xnumel=self.size_hints["x"],
            ynumel=self.size_hints["y"],
            znumel=self.size_hints["z"],
            nametag=self.nametag
        )
        # run prediction on all configurations:
        if self.model is None:
            log.error("Model not defined, returning empty list of configs")
            return []
        predictions = self.model.predict(input_df)
        # add predictions to the table
        comb = input_df.copy()
        comb["Y"] = predictions
        # sort: smaller the better
        sorted_ = comb.sort_values(by='Y', ascending=True)
        final=self.tab_transform_to_powers_of_two(sorted_)
        configs = []
        for i, row  in enumerate(final.iterrows()):
            data = row[1]
            # print(">>", data)
            cfg = {}
            if "x" in self.original_size_hints: # NOTE: according to the original size_hints, not self.size_hints (thats modified)
                cfg["XBLOCK"] = int(data["XBLOCK"])
            if "y" in self.original_size_hints:
                cfg["YBLOCK"] = int(data["YBLOCK"])
            if "z" in self.original_size_hints:
                cfg["ZBLOCK"] = int(data["ZBLOCK"])
            Config = self.configClass
            configs.append(Config(cfg, num_warps=int(data["num_warps"]), num_stages=int(data["num_stages"])))
            if i>=nmax: # some reasonable cutoff..
                break
        return configs


    def train(self, csv_file: Path):
        """Just an example how this goes.. probably your not supposed
        to train using this class
        """
        from tabulate import tabulate
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        import pandas as pd

        df = pd.read_csv(csv_file)
        """

        the csv file (created with another tool lilbrary), looks like this:
        
        ::

            |    |   name |   XBLOCK |   YBLOCK |   ZBLOCK |   num_warps |   num_stages |   xnumel |   ynumel |   znumel |         Y |
            |----|--------|----------|----------|----------|-------------|--------------|----------|----------|----------|-----------|
            |  0 |      1 |        1 |        9 |        0 |           1 |            1 |        1 |       25 |        0 | 0.099476  |
            |  1 |      1 |        1 |        9 |        0 |           1 |            2 |        1 |       25 |        0 | 0.099396  |
            |  2 |      1 |        1 |        9 |        0 |           1 |            4 |        1 |       25 |        0 | 0.0991965 |
            |  3 |      1 |        1 |        9 |        0 |           1 |            8 |        1 |       25 |        0 | 0.099316  |

        XBLOCK, YBLOCK, ZBLOCK and xnumel, ynumel and znumel are in the units of powers of two
        Y is execution time (ms)
        """
        """
        markdown_output = tabulate(df.head(4), headers='keys', tablefmt='github')
        # Print the pretty Markdown output
        print(markdown_output)
        """
        # Define features and target variable
        X = df.drop(columns=['Y'])
        y = df['Y']
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the XGBoost regressor
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)


def test1():
    """config generation sanity test
    """
    confLogger(log, logging.DEBUG)
    ko = KernelOracle({'y': 33554432, 'x': 16}, kernel_name='triton_poi_fused_clone_14', path_to_model=None)
    print(ko)
    ko.rankConfigs()

def test2():
    """model download & caching
    """
    confLogger(log, logging.DEBUG)
    ko = KernelOracle({'y': 33554432, 'x': 16}, kernel_name='triton_poi_fused_clone_14', path_to_model=None)
    if ko:
        print("ok!")
    else:
        print("def")
    print(ko)

def test3():
    """test local model loading
    """
    confLogger(log, logging.DEBUG)
    ko = KernelOracle({'y': 33554432, 'x': 16}, 
            kernel_name='triton_poi_fused_clone_14', 
            path_to_model="/root/shared/notebook/mi350_playground.pkl", 
            fetch_model=False)
    if ko:
        print("ok!")
    else:
        print("def")
    print(ko)

def test4():
    """test local model loading and inference
    """
    confLogger(log, logging.INFO)
    ko = KernelOracle({'y': 33554432, 'x': 16}, 
            kernel_name='triton_poi_fused_clone_14', 
            path_to_model="/root/shared/notebook/mi350_playground.pkl", 
            fetch_model=False)
    if ko:
        confs = ko.rankConfigs()
        for conf in confs:
            print(conf)

if __name__ == "__main__":
    # very basic testing of this module
    test1()
    # test2()
    # test3()
    # test4()
