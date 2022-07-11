



from caffe2.python.normalizer_context import UseNormalizer, NormalizerContext
from caffe2.python.normalizer import BatchNormalizer
from caffe2.python.layer_test_util import LayersTestCase


class TestNormalizerContext(LayersTestCase):
    def test_normalizer_context(self):
        bn = BatchNormalizer(momentum=0.1)
        with UseNormalizer({'BATCH': bn}):
            normalizer = NormalizerContext.current().get_normalizer('BATCH')
            self.assertEquals(bn, normalizer)
