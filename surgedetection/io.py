from pathlib import Path
import re
import tarfile

def list_tar_filepaths(tar_filepath: Path, pattern: str = r".*\.tif", prepend_vsitar: bool = True) -> list[str]:
    """
        List all files matching a pattern in a tarfile (.tar or .tar.gz)
        
        :param tar_filepath: The path to the tarfile
        :param pattern: A pattern to match for the files. Defaults to all tiffs
        :param prepend_vsitar: If "/vsitar/" should be prepended to each string. Allows for direct GDAL use
        
        :returns: A list of filenames matching the pattern.
    """
    tar_filepath = Path(tar_filepath).absolute()
    
    filepaths: list[str] = []
    
    with tarfile.open(tar_filepath) as tar:
        for tar_file in filter(
            lambda name: re.match(pattern, name) is not None, tar.getnames()
        ):
            filepaths.append(f"{'/vsitar/' if prepend_vsitar else ''}{tar_filepath}/{tar_file}")
            
    return filepaths
