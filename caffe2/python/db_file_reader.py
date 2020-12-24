## @package db_file_reader
# Module caffe2.python.db_file_reader





from caffe2.python import core, scope, workspace, _import_c_extension as C
from caffe2.python.dataio import Reader
from caffe2.python.dataset import Dataset
from caffe2.python.schema import from_column_list

import os


class DBFileReader(Reader):

    default_name_suffix = 'db_file_reader'

    """Reader reads from a DB file.

    Example usage:
    db_file_reader = DBFileReader(db_path='/tmp/cache.db', db_type='LevelDB')

    Args:
        db_path: str.
        db_type: str. DB type of file. A db_type is registed by
            `REGISTER_CAFFE2_DB(<db_type>, <DB Class>)`.
        name: str or None. Name of DBFileReader.
            Optional name to prepend to blobs that will store the data.
            Default to '<db_name>_<default_name_suffix>'.
        batch_size: int.
            How many examples are read for each time the read_net is run.
        loop_over: bool.
            If True given, will go through examples in random order endlessly.
        field_names: List[str]. If the schema.field_names() should not in
            alphabetic order, it must be specified.
            Otherwise, schema will be automatically restored with
            schema.field_names() sorted in alphabetic order.
    """
    def __init__(
        self,
        db_path,
        db_type,
        name=None,
        batch_size=100,
        loop_over=False,
        field_names=None,
    ):
        assert db_path is not None, "db_path can't be None."
        assert db_type in C.registered_dbs(), \
            "db_type [{db_type}] is not available. \n" \
            "Choose one of these: {registered_dbs}.".format(
                db_type=db_type,
                registered_dbs=C.registered_dbs(),
        )

        self.db_path = os.path.expanduser(db_path)
        self.db_type = db_type
        self.name = name or '{db_name}_{default_name_suffix}'.format(
            db_name=self._extract_db_name_from_db_path(),
            default_name_suffix=self.default_name_suffix,
        )
        self.batch_size = batch_size
        self.loop_over = loop_over

        # Before self._init_reader_schema(...),
        # self.db_path and self.db_type are required to be set.
        super(DBFileReader, self).__init__(self._init_reader_schema(field_names))
        self.ds = Dataset(self._schema, self.name + '_dataset')
        self.ds_reader = None

    def _init_name(self, name):
        return name or self._extract_db_name_from_db_path(
        ) + '_db_file_reader'

    def _init_reader_schema(self, field_names=None):
        """Restore a reader schema from the DB file.

        If `field_names` given, restore scheme according to it.

        Overwise, loade blobs from the DB file into the workspace,
        and restore schema from these blob names.
        It is also assumed that:
        1). Each field of the schema have corresponding blobs
            stored in the DB file.
        2). Each blob loaded from the DB file corresponds to
            a field of the schema.
        3). field_names in the original schema are in alphabetic order,
            since blob names loaded to the workspace from the DB file
            will be in alphabetic order.

        Load a set of blobs from a DB file. From names of these blobs,
        restore the DB file schema using `from_column_list(...)`.

        Returns:
            schema: schema.Struct. Used in Reader.__init__(...).
        """
        if field_names:
            return from_column_list(field_names)

        if self.db_type == "log_file_db":
            assert os.path.exists(self.db_path), \
                'db_path [{db_path}] does not exist'.format(db_path=self.db_path)
        with core.NameScope(self.name):
            # blob_prefix is for avoiding name conflict in workspace
            blob_prefix = scope.CurrentNameScope()
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'Load',
                [],
                [],
                absolute_path=True,
                db=self.db_path,
                db_type=self.db_type,
                load_all=True,
                add_prefix=blob_prefix,
            )
        )
        col_names = [
            blob_name[len(blob_prefix):] for blob_name in workspace.Blobs()
            if blob_name.startswith(blob_prefix)
        ]
        schema = from_column_list(col_names)
        return schema

    def setup_ex(self, init_net, finish_net):
        """From the Dataset, create a _DatasetReader and setup a init_net.

        Make sure the _init_field_blobs_as_empty(...) is only called once.

        Because the underlying NewRecord(...) creats blobs by calling
        NextScopedBlob(...), so that references to previously-initiated
        empty blobs will be lost, causing accessibility issue.
        """
        if self.ds_reader:
            self.ds_reader.setup_ex(init_net, finish_net)
        else:
            self._init_field_blobs_as_empty(init_net)
            self._feed_field_blobs_from_db_file(init_net)
            self.ds_reader = self.ds.random_reader(
                init_net,
                batch_size=self.batch_size,
                loop_over=self.loop_over,
            )
            self.ds_reader.sort_and_shuffle(init_net)
            self.ds_reader.computeoffset(init_net)

    def read(self, read_net):
        assert self.ds_reader, 'setup_ex must be called first'
        return self.ds_reader.read(read_net)

    def _init_field_blobs_as_empty(self, init_net):
        """Initialize dataset field blobs by creating an empty record"""
        with core.NameScope(self.name):
            self.ds.init_empty(init_net)

    def _feed_field_blobs_from_db_file(self, net):
        """Load from the DB file at db_path and feed dataset field blobs"""
        if self.db_type == "log_file_db":
            assert os.path.exists(self.db_path), \
                'db_path [{db_path}] does not exist'.format(db_path=self.db_path)
        net.Load(
            [],
            self.ds.get_blobs(),
            db=self.db_path,
            db_type=self.db_type,
            absolute_path=True,
            source_blob_names=self.ds.field_names(),
        )

    def _extract_db_name_from_db_path(self):
        """Extract DB name from DB path

            E.g. given self.db_path=`/tmp/sample.db`, or
            self.db_path = `dper_test_data/cached_reader/sample.db`
            it returns `sample`.

            Returns:
                db_name: str.
        """
        return os.path.basename(self.db_path).rsplit('.', 1)[0]
