import torcharrow as ta

try:
    import pandas  # type: ignore[import]

    # pandas used only for prototyping, will be shortly replaced with TorchArrow
    WITH_PANDAS = True
except ImportError:
    WITH_PANDAS = False


class PandasWrapper:
    @classmethod
    def create_dataframe(cls, data, **kwargs):
        if not WITH_PANDAS:
            Exception("DataFrames prototype requires pandas to function")
        return pandas.DataFrame(data, **kwargs)

    @classmethod
    def is_dataframe_object(cls, data):
        if not WITH_PANDAS:
            Exception("DataFrames prototype requires pandas to function")
        return isinstance(data, pandas.core.frame.DataFrame)

    @classmethod
    def is_series_object(cls, data):
        if not WITH_PANDAS:
            Exception("DataFrames prototype requires pandas to function")
        return isinstance(data, pandas.core.series.Series)

    @classmethod
    def concat(cls, buffer, **kwargs):
        if not WITH_PANDAS:
            Exception("DataFrames prototype requires pandas to function")
        return pandas.concat(buffer, **kwargs)

    @classmethod
    def get_len(cls, df):
        if not WITH_PANDAS:
            Exception("DataFrames prototype requires pandas to function")
        return len(df.index)


# When you build own implementation just override it with dataframe_wrapper.set_df_wrapper(new_wrapper_class)
default_wrapper = PandasWrapper


def get_df_wrapper():
    return default_wrapper


def set_df_wrapper(wrapper):
    default_wrapper = wrapper


def create_dataframe(data, **kwargs):
    wrapper = get_df_wrapper()
    wrapper.create_dataframe(data, **kwargs)


def is_dataframe_object(data):
    wrapper = get_df_wrapper()
    wrapper.is_dataframe_object()


def is_series_object(data):
    wrapper = get_df_wrapper()
    wrapper.is_series_object()


def concat(buffer):
    wrapper = get_df_wrapper()
    wrapper.concat(buffer)


def get_len(df):
    wrapper = get_df_wrapper()
    wrapper.get_len(df)
