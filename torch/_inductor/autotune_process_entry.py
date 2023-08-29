from torch._inductor.autotune_process import TuningProcess

# Entry point for the subprocess supporting the TuningProcess's benchmark operation.
if __name__ == "__main__":
    TuningProcess.process_main()
