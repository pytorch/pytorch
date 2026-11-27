"""Makes WITNESS_TSV, the fourth input of gen_torch_members.py: per sample of
a member of the table of torch's C callees, what the engine's witness of the
dispatcher saw under one call of the callee on that sample.

    witness_torch_members.py all SWEEP_DIR TABLE DEVICE JOBS OUT_DIR
    witness_torch_members.py merge OUT_DIR WITNESS_TSV
    witness_torch_members.py one SWEEP_DIR DEVICE OUT_DIR MEMBER...

SWEEP_DIR is the directory of the sweep whose table and samples the
generator reads (its modules `sweep` and `measure` are imported from there,
and its cache of samples is read). TABLE is torch_members.tsv: the members
are its lines without a form. DEVICE is cpu or cuda. OUT_DIR must be
absolute.

`all` runs `one` for every member, JOBS processes at a time, six members per
process; a member whose process died or hung is run again alone, twice at
most. `one` makes every sample the sweep has for a member once (without the
derived samples that pass `out=`), in the sample's own ambient setting, and
calls the callee on it under torch._rust._witness. One line per sample goes
to OUT_DIR/<member>.<device>.jsonl. `merge` writes the table: member, device,
id, returned, read, top, opaque, why. A call that raises is not witnessed
(`returned` 0, the other columns `-`).

    read    an operator that hands a value out of a tensor ran under the call
    top     one of them was a top-level operator
    opaque  the witness could not classify what ran

The table of the members is then made again with it:
    witness_torch_members.py all SWEEP_DIR TABLE cpu 12 OUT_DIR
    witness_torch_members.py all SWEEP_DIR TABLE cuda 6 OUT_DIR
    witness_torch_members.py merge OUT_DIR WITNESS_TSV
    gen_torch_members.py SWEEP_TSV NO_OUT_TSV SAMPLES_DIR WITNESS_TSV
"""

import glob
import json
import os
import signal
import subprocess
import sys
import traceback


def one(sweep_dir, device, out_dir, members):
    sys.path.insert(0, sweep_dir)
    os.chdir(sweep_dir)
    import measure
    import sweep

    import torch

    torch.set_num_threads(1)
    for member in members:
        path = os.path.join(out_dir, f"{member}.{device}.jsonl")
        try:
            callee = sweep.resolve(member)
            samples, _ = sweep.collect(member, callee, 80, device)
        except Exception as e:
            with open(path, "w") as f:
                f.write(json.dumps({"tool_error": f"{type(e).__name__}: {e}"}) + "\n")
            continue
        with open(path, "w") as f:
            for sample, make in samples:
                if sample.startswith("out:"):
                    continue
                record = {"id": sample}
                signal.alarm(60)
                try:
                    operands = make()
                    args, kwargs = operands[0], operands[1]
                    seen = None
                    with measure.setting(operands):
                        torch.manual_seed(measure.SEED)
                        try:
                            _, seen = torch._rust._witness(callee, *args, **kwargs)
                            record["returned"] = True
                        except Exception as e:
                            record["returned"] = False
                            record["raised"] = f"{type(e).__name__}: {str(e)[:120]}"
                    if seen is not None:
                        opaque = bool(seen["opaque"])
                        record.update(
                            read=bool(seen["read_value"]) and not opaque,
                            top=bool(seen["read_value_top"]) and not opaque,
                            opaque=opaque,
                            why=str(seen["why"])[:160],
                            ops=list(seen["ops"])[:8],
                        )
                except sweep.Skip as e:
                    record["skipped"] = str(e)
                except Exception as e:
                    record["tool_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    record["trace"] = traceback.format_exc()[-400:]
                signal.alarm(0)
                f.write(json.dumps(record) + "\n")
                f.flush()
            f.write(json.dumps({"done": len(samples)}) + "\n")


def finished(out_dir, member, device):
    path = os.path.join(out_dir, f"{member}.{device}.jsonl")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return any(
            '"done"' in line or ('"tool_error"' in line and '"id"' not in line)
            for line in f
        )


def run_all(sweep_dir, table, device, jobs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    members = []
    with open(table) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not line.startswith("#") and len(parts) > 2 and parts[1] == "-":
                members.append(parts[0])
    env = dict(os.environ)
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    started = 0

    def run(batches):
        nonlocal started
        running = []
        while batches or running:
            while batches and len(running) < jobs:
                path = os.path.join(out_dir, f"log.{device}.{started}.txt")
                cmd = ["timeout", "900", sys.executable, os.path.abspath(__file__)]
                cmd += ["one", sweep_dir, device, out_dir, *batches.pop(0)]
                with open(path, "w") as log:
                    running.append(
                        subprocess.Popen(
                            cmd, env=env, stdout=log, stderr=subprocess.STDOUT
                        )
                    )
                started += 1
            running[0].wait()
            running = [p for p in running if p.poll() is None]

    todo = [m for m in members if not finished(out_dir, m, device)]
    run([todo[i : i + 6] for i in range(0, len(todo), 6)])
    for _ in range(2):
        run([[m] for m in members if not finished(out_dir, m, device)])
    left = [m for m in members if not finished(out_dir, m, device)]
    print(f"{device}: {len(members)} members, not finished: {left}")


def merge(out_dir, out):
    lines = ["member\tdevice\tid\treturned\tread\ttop\topaque\twhy"]
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        member, device = os.path.basename(path)[: -len(".jsonl")].rsplit(".", 1)
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if "id" not in record or not ("read" in record or "raised" in record):
                    continue

                def said(k, record=record):
                    return "-" if k not in record else str(int(record[k]))

                why = record.get("why", "").replace("\t", " ").replace("\n", " ")
                flags = (said(k) for k in ("returned", "read", "top", "opaque"))
                lines.append(
                    "\t".join((member, device, record["id"], *flags, why or "-"))
                )
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(len(lines) - 1, "samples")


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "all":
        run_all(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
    else:
        one(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
