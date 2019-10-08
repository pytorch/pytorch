.. java:import:: android.graphics Bitmap

.. java:import:: android.graphics ImageFormat

.. java:import:: android.media Image

.. java:import:: org.pytorch Tensor

.. java:import:: java.nio ByteBuffer

.. java:import:: java.util Locale

TensorImageUtils
================

.. java:package:: org.pytorch.torchvision
   :noindex:

.. java:type:: public final class TensorImageUtils

   Contains utility functions for \ :java:ref:`org.pytorch.Tensor`\  creation from \ :java:ref:`android.graphics.Bitmap`\  or \ :java:ref:`android.media.Image`\  source.

Fields
------
TORCHVISION_NORM_MEAN_RGB
^^^^^^^^^^^^^^^^^^^^^^^^^

.. java:field:: public static float[] TORCHVISION_NORM_MEAN_RGB
   :outertype: TensorImageUtils

TORCHVISION_NORM_STD_RGB
^^^^^^^^^^^^^^^^^^^^^^^^

.. java:field:: public static float[] TORCHVISION_NORM_STD_RGB
   :outertype: TensorImageUtils

Methods
-------
bitmapToFloat32Tensor
^^^^^^^^^^^^^^^^^^^^^

.. java:method:: public static Tensor bitmapToFloat32Tensor(Bitmap bitmap, float[] normMeanRGB, float[] normStdRGB)
   :outertype: TensorImageUtils

   Creates new \ :java:ref:`org.pytorch.Tensor`\  from full \ :java:ref:`android.graphics.Bitmap`\ , normalized with specified in parameters mean and std.

   :param normMeanRGB: means for RGB channels normalization, length must equal 3, RGB order
   :param normStdRGB: standard deviation for RGB channels normalization, length must equal 3, RGB order

bitmapToFloat32Tensor
^^^^^^^^^^^^^^^^^^^^^

.. java:method:: public static Tensor bitmapToFloat32Tensor(Bitmap bitmap, int x, int y, int width, int height, float[] normMeanRGB, float[] normStdRGB)
   :outertype: TensorImageUtils

   Creates new \ :java:ref:`org.pytorch.Tensor`\  from specified area of \ :java:ref:`android.graphics.Bitmap`\ , normalized with specified in parameters mean and std.

   :param bitmap: \ :java:ref:`android.graphics.Bitmap`\  as a source for Tensor data
   :param x: x coordinate of top left corner of bitmap's area
   :param y: y coordinate of top left corner of bitmap's area
   :param width: width of bitmap's area
   :param height: height of bitmap's area
   :param normMeanRGB: means for RGB channels normalization, length must equal 3, RGB order
   :param normStdRGB: standard deviation for RGB channels normalization, length must equal 3, RGB order

imageYUV420CenterCropToFloat32Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. java:method:: public static Tensor imageYUV420CenterCropToFloat32Tensor(Image image, int rotateCWDegrees, int tensorWidth, int tensorHeight, float[] normMeanRGB, float[] normStdRGB)
   :outertype: TensorImageUtils

   Creates new \ :java:ref:`org.pytorch.Tensor`\  from specified area of \ :java:ref:`android.media.Image`\ , doing optional rotation, scaling (nearest) and center cropping.

   :param image: \ :java:ref:`android.media.Image`\  as a source for Tensor data
   :param rotateCWDegrees: Clockwise angle through which the input image needs to be rotated to be upright. Range of valid values: 0, 90, 180, 270
   :param tensorWidth: return tensor width, must be positive
   :param tensorHeight: return tensor height, must be positive
   :param normMeanRGB: means for RGB channels normalization, length must equal 3, RGB order
   :param normStdRGB: standard deviation for RGB channels normalization, length must equal 3, RGB order
