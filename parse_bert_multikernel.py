runtime = None
results = []
local = []
for r in open("log_bert_pytorch_multikernel.out"):
    r = r.strip()
    print(r)
    if r.startswith("runtime input shapes"):
        if local:
            results.append(local)
        runtime = r.split("shapes ")[1]
        local = []
    elif r.startswith("picked kernel"):
        try:
            index = int(r.split("picked kernel ")[1].split(" ")[0])
            choice = r.split("  :  ")[1].split(" for shape ")[0]
            key = r.split("for shape ")[1]
            local.append((index, choice, key))
        except:
            pass

breakpoint()
for r in results:
    print([x[0] for x in r])
