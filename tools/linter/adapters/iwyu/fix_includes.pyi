from typing import Optional, List, Any
import io

class IWYUOutputRecord:
    filename: str

    def Merge(self, other: IWYUOutputRecord) -> None: ...
    def HasContentfulChanges(self) -> bool: ...

class IWYUOutputParser:
    def ParseOneRecord(self, iwyu_output: "io.StringIO", flags: Any) -> Optional[IWYUOutputRecord]: ...

class FileInfo:
    encoding: str

    @staticmethod
    def parse(filename: str) -> "FileInfo": ...

class LineInfo:
    line: str


class FixIncludesError(Exception):
    pass

def ParseOneFile(f: List[str], iwyu_record: IWYUOutputRecord) -> List[LineInfo]: ...
def NormalizeFilePath(basedir: Optional[str], filename: str) -> str: ...
def FixFileLines(iwyu_record: IWYUOutputRecord, file_lines: List[LineInfo], flags: Any, fileinfo: FileInfo) -> List[str]: ...
