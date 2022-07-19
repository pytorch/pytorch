_pandas = None
_WITH_PANDAS = None


def _try_import_pandas() -> bool:
    try:
        import pandas  # type: ignore[import]
        global _pandas
        _pandas = pandas
        return True
    except ImportError:
        return False


# pandas used only for prototyping, will be shortly replaced with TorchArrow
def _with_pandas() -> bool:
    global _WITH_PANDAS
    if _WITH_PANDAS is None:
        _WITH_PANDAS = _try_import_pandas()
    return _WITH_PANDAS


class PandasWrapper:
    @classmethod
    def create_dataframe(cls, data, columns):
        if not _with_pandas():
            raise Exception("DataFrames prototype requires pandas to function")
        return _pandas.DataFrame(data, columns=columns)  # type: ignore[union-attr]

    @classmethod
    def is_dataframe(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.frame.DataFrame)  # type: ignore[union-attr]

    @classmethod
    def is_column(cls, data):
        if not _with_pandas():
            return False
        return isinstance(data, _pandas.core.series.Series)  # type: ignore[union-attr]

    @classmethod
    def iterate(cls, data):
        if not _with_pandas():
            raise Exception("DataFrames prototype requires pandas to function")
        for d in data.itertuples(index=False):
            yield d

    @classmethod
    def concat(cls, buffer):
        if not _with_pandas():
            raise Exception("DataFrames prototype requires pandas to function")
        return _pandas.concat(buffer)  # type: ignore[union-attr]

    @classmethod
    def get_item(cls, data, idx):
        if not _with_pandas():
            raise Exception("DataFrames prototype requires pandas to function")
        return data[idx: idx + 1]

    @classmethod
    def get_len(cls, df):
        if not _with_pandas():
            raise Exception("DataFrames prototype requires pandas to function")
        return len(df.index)

    @classmethod
    def get_columns(cls, df):
        if not _with_pandas():
            raise Exception("DataFrames prototype requires pandas to function")
        return list(df.columns.values.tolist())


# When you build own implementation just override it with dataframe_wrapper.set_df_wrapper(new_wrapper_class)
default_wrapper = PandasWrapper


def get_df_wrapper():
    return default_wrapper


def set_df_wrapper(wrapper):
    global default_wrapper
    default_wrapper = wrapper


def create_dataframe(data, columns=None):
    wrapper = get_df_wrapper()
    return wrapper.create_dataframe(data, columns)


def is_dataframe(data):
    wrapper = get_df_wrapper()
    return wrapper.is_dataframe(data)


def get_columns(data):
    wrapper = get_df_wrapper()
    return wrapper.get_columns(data)


def is_column(data):
    wrapper = get_df_wrapper()
    return wrapper.is_column(data)


def concat(buffer):
    wrapper = get_df_wrapper()
    return wrapper.concat(buffer)


def iterate(data):
    wrapper = get_df_wrapper()
    return wrapper.iterate(data)


def get_item(data, idx):
    wrapper = get_df_wrapper()
    return wrapper.get_item(data, idx)


def get_len(df):
    wrapper = get_df_wrapper()
    return wrapper.get_len(df)
