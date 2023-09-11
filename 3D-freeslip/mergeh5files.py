import subprocess
from dedalus.tools import post
import pathlib

set_paths = list(pathlib.Path("diagnostics").glob("diagnostics_s*.h5"))
post.merge_sets("diagnostics/diagnostics_0.h5", set_paths, cleanup=False)
