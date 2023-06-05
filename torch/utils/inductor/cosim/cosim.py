#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbg_log", help="Output log with TORCH_COMPILE_DEBUG=1")
    parser.add_argument("--dbg_code_src_path", help="Debug code path with TORCH_COMPILE_DEBUG=1")
    parser.add_argument("--dbg_code_dst_path", help="Copy the debug code to the dst path and rename it by graph idex")
    parser.add_argument("--dbg_code_dst_single_readable_py", help="Compose all the readable graph as a single graph for comparasion")
    args = parser.parse_args()
    return args

def find_dir(path, sub_file_str):
  for sub_item in os.scandir(path):
      if sub_item.is_dir():
          for inner_sub_item in os.scandir(sub_item):
            if inner_sub_item.is_dir() and inner_sub_item.name.lower() == sub_file_str.lower():
              return sub_item
  return ""


# The function assumes the sub items of the path is as follows:
# path
#   |-xxxx.debug
#   |-xxxx.py
def rename_sub_dir_file_by_graph_idx(path, graph_idx):
  for sub_item in os.scandir(path):
      if sub_item.is_dir():
        shutil.move(sub_item, os.path.join(path, graph_idx + ".debug"))
      else:
        assert(sub_item.is_file())
        shutil.move(sub_item, os.path.join(path, graph_idx + ".py"))


def refine_dbg_code_src_path(args):
  compile_graph_flag = "torchinductor done compiling FORWARDS graph"
  output_code_flag = "debug trace:"

  dbg_log = Path(args.dbg_log)
  dbg_code_src_path = Path(args.dbg_code_src_path)
  dbg_code_dst_path = Path(args.dbg_code_dst_path)
  dbg_code_dst_single_readable_py = Path(args.dbg_code_dst_single_readable_py)

  # Create the directories to create readable.py
  if os.path.exists(dbg_code_dst_path):
    shutil.rmtree(dbg_code_dst_path)
  if not os.path.exists(dbg_code_dst_single_readable_py.parent):
    dbg_code_dst_single_readable_py.parent.mkdir(parents=True)

  with open(dbg_log) as f, open(dbg_code_dst_single_readable_py, "a") as all_readable_py:
    lines = f.readlines()
    for i in range(0, len(lines)):
      line = lines[i]
      if compile_graph_flag in line:
        # Parse log file to get graph index
        line_items = line.split(compile_graph_flag)
        assert(len(line_items) == 2)
        graph_idx = str(int(line_items[1]))

        # Parse debug dir
        next_line = lines[i + 1]
        assert (output_code_flag in next_line)
        line_items = next_line.split(output_code_flag)
        assert(len(line_items) == 2)
        output_code_name = line_items[1].strip()
        assert(output_code_name.endswith("debug"))

        output_code_name = output_code_name.split('/')
        current_graph_dbg_path = output_code_name[-1]
        current_graph_gen_code_path = find_dir(dbg_code_src_path, current_graph_dbg_path)
        assert(current_graph_gen_code_path != "")
        assert(os.path.exists(current_graph_gen_code_path))

        print("Gen code of graph {} is placed at {}".format(graph_idx, current_graph_gen_code_path))

        output_path_with_graph_idx = os.path.join(dbg_code_dst_path, graph_idx)
        shutil.copytree(current_graph_gen_code_path, output_path_with_graph_idx, dirs_exist_ok=True)
        # Recursively enumerate sub dir and files and rename them to graph_idx
        rename_sub_dir_file_by_graph_idx(output_path_with_graph_idx, graph_idx)
        print("Done copying the gened code of graph {} to {}".format(graph_idx, output_path_with_graph_idx))

        with open(os.path.join(output_path_with_graph_idx, graph_idx + ".debug", "fx_graph_readable.py")) as graph_readable:
          all_readable_py.writelines("GRAPH_INDEX:{}\n".format(graph_idx))
          content = graph_readable.readlines()
          all_readable_py.writelines(content)
          all_readable_py.writelines("\n")


if __name__ == "__main__":
    args = parse_args()
    assert args.dbg_log
    assert args.dbg_code_src_path
    assert args.dbg_code_dst_path
    assert args.dbg_code_dst_single_readable_py
    refine_dbg_code_src_path(args)
