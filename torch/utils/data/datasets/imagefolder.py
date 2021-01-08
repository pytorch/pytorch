import io
import os
import time
import tarfile
from pathlib import Path

from torch.utils.data.datasets.common import get_file_pathnames_from_root


def is_img_ext(ext: str):
    return ext.lower() in [".png", ".jpg", ".jpeg", ".img", ".image", ".pbm", ".pgm", ".ppm"]

class ImageFolder:
    r""" :class:`ImageFolder`

    This is a class to do pre-processing for an image folder
    args:
        root: the root of the image files

    """

    def __init__(self, root: str = '.'):
        self.root : str = root


    def to_tar(self, tar_pathname: str, create_label: bool = True):

        # always compress for now
        tarstream = tarfile.open(tar_pathname, mode="w:gz")

        for pathname in get_file_pathnames_from_root(self.root, ''):
            filename = os.path.basename(pathname)
            splits = os.path.splitext(filename)
            basename = splits[0]
            ext = splits[1]

            # do not allow any non image file exist in the folder
            if not is_img_ext(ext):
                try:
                    tarstream.close()
                    os.remove(tar_pathname)
                except Exception as e:
                    pass
                raise TypeError("Image folder {} should only contain image file, but got non-image file {}".format(
                    self.root, pathname))

            # no encoding at the moment, store the raw image binary into tar file
            tarstream.add(pathname, arcname=filename)

            if create_label:
                category_id = os.path.basename(os.path.normpath(self.root))
                bio = io.BytesIO()
                bio.write(str.encode('{{"category_id": "{}"}}'.format(category_id)))
                path_info = Path(pathname)
                tinfo = tarfile.TarInfo(basename + ".json")
                tinfo.size = bio.tell()
                bio.seek(0)
                tinfo.mtime = time.time()
                tinfo.uname = path_info.owner()
                tinfo.gname = path_info.group()
                tinfo.mode = path_info.stat().st_mode & 0o0777
                tarstream.addfile(tinfo, bio)

        tarstream.close()
