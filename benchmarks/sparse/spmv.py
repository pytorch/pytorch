import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpMV")

    parser.add_argument("--format", default='gcs', type=str)
    parser.add_argument("--m", default='1000', type=int)
    parser.add_argument("--nnz_ratio", default='0.1', type=float)
    parser.add_argument("--outfile", default='stdout', type=str)

    args = parser.parse_args()

    if args.outfile == 'stdout':
        outfile = sys.stdout
    elif args.outfile == 'stderr':
        outfile = sys.stderr
    else:
        outfile = args.outfile

    print(args, file=outfile)
        
