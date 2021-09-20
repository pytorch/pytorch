try:
    import pandas  # type: ignore[import]
    # pandas used only for prototyping, will be shortly replaced with TorchArrow
    WITH_PANDAS = True
except ImportError:
    WITH_PANDAS = False


class PandasWrapper():
    @classmethod
    def concat(cls, buffer):
        if not WITH_PANDAS:
            Exception('DataFrames prototype requires pandas to function')
        return pandas.concat(buffer)

    @classmethod
    def get_len(cls, df):
        if not WITH_PANDAS:
            Exception('DataFrames prototype requires pandas to function')
        return len(df.index)


default_wrapper = PandasWrapper

# When you build own implementation just override it with dataframe_wrapper.set_df_wrapper(new_wrapper_class)


def get_df_wrapper():
    return default_wrapper


def set_df_wrapper(wrapper):
    default_wrapper = wrapper


def concat(buffer):
    wrapper = get_df_wrapper()
    wrapper.concat(buffer)


def get_len(df):
    wrapper = get_df_wrapper()
    wrapper.conget_len(df)
