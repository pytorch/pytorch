import os
import shutil
import glob
import subprocess
import argparse
import sys
import prettytable
import json

### Setup ###

def clear_mkdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

clear_mkdir('modules')
clear_mkdir('targets')
clear_mkdir('targets/generated')

target_choices = [
    'spirv',         # SPIRV directly
    'spirv-glsl',    # SPIRV through synthesized GLSL
    'dxil',          # DXIL with HLSL and DXC
    'dxil-embedded'  # DXIL with precompiled modules
]

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='spirv', choices=target_choices)
parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--output', type=str, default='benchmarks.json')

args = parser.parse_args(sys.argv[1:])

slangc = '..\\..\\build\\Release\\bin\\slangc.exe'
target = args.target
samples = args.samples

if target == 'spirv':
    target = 'spirv -emit-spirv-directly'
    target_ext = 'spirv'
    embed = False
elif target == 'spirv-glsl':
    target = 'spirv -emit-spirv-via-glsl'
    target_ext = 'spirv'
    embed = False
elif target == 'dxil-embedded':
    target_ext = 'dxil'
    embed = True
elif target == 'dxil':
    target_ext = 'dxil'
    embed = False

print(f'slangc:  {slangc}')
print(f'target:  {target}')
print(f'samples: {samples}\n')

### Utility ###

def parse(results):
    results = results.split('\n')
    results = [ r for r in results if r.startswith('[*]') ]
    results = [ r.split() for r in results ]
    profile = {}
    for r in results:
        profile[r[1]] = float(r[-1][:-2])
    return profile

timings = {}
def run(command, key):
    profile = {}
    for i in range(samples):
        try:
            results = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True).decode('utf-8')
        except subprocess.CalledProcessError as exc:
            print(f"[Error] Failed to run command: {command}")
            print(exc.output.decode('utf-8'))
            return  # Return without adding to timings

        p = parse(results)
        if len(profile) == 0:
            profile = p
        else:
            for k, v in p.items():
                profile.setdefault(k, 0)
                profile[k] += v

    # Only add to timings if we have data
    if profile:
        for k in profile:
            profile[k] /= samples
        timings[key] = profile
    else:
        print(f"[Warning] No timing data collected for {key}")

def compile_cmd(file, output, stage=None, entry=None, emit=False):
    cmd = f'{slangc} -report-perf-benchmark {file}'

    if stage:
        cmd += f' -stage {stage}'
        if entry:
            cmd += f' -entry {entry}'
        else:
            cmd += f' -entry {stage}'

    if emit:
        cmd += f' -target {target_ext}'
        output += '.' + target_ext
        if target == 'dxil-embedded':
            cmd += ' -profile lib_6_6'
    elif embed:
        cmd += ' -embed-dxil'

    cmd += f' -o {output}'

    return cmd

### Monolithic compilation ###

hit = 'hit.slang'

cmd = compile_cmd(hit, f'targets/dxr-ch-mono', stage='closesthit', entry='MdlRadianceClosestHitProgram', emit=True)
run(cmd, f'full/{target_ext}/mono/closesthit')
print(f'[I] compiled shadow (monolithic)')

cmd = compile_cmd(hit, f'targets/dxr-ah-mono', stage='anyhit', entry='MdlRadianceAnyHitProgram', emit=True)
run(cmd, f'full/{target_ext}/mono/anyhit')
print(f'[I] compiled shadow (monolithic)')

cmd = compile_cmd(hit, f'targets/dxr-sh-mono', stage='anyhit', entry='MdlShadowAnyHitProgram', emit=True)
run(cmd, f'full/{target_ext}/mono/shadow')
print(f'[I] compiled shadow (monolithic)')

### Module precompilation ###

modules = []

for file in glob.glob(f'*.slang'):
    if not file.endswith('hit.slang'):
        basename = os.path.basename(file)
        run(compile_cmd(file, f'modules/{basename}-module'), 'module/' + file)
        print(f'[I] compiled {file}.')

### Module whole compilation ###

cmd = compile_cmd(hit, f'targets/dxr-ch-modules', stage='closesthit', entry='MdlRadianceClosestHitProgram', emit=True)
run(cmd, f'full/{target_ext}/module/closesthit')
print(f'[I] compiled closesthit (module)')

cmd = compile_cmd(hit, f'targets/dxr-ah-modules', stage='anyhit', entry='MdlRadianceAnyHitProgram', emit=True)
run(cmd, f'full/{target_ext}/module/anyhit')
print(f'[I] compiled anyhit (module)')

cmd = compile_cmd(hit, f'targets/dxr-sh-modules', stage='anyhit', entry='MdlShadowAnyHitProgram', emit=True)
run(cmd, f'full/{target_ext}/module/shadow')
print(f'[I] compiled shadow (module)')

# Module precompilation time
precompilation_time = 0
for k in timings:
    if k.startswith('module'):
        precompilation_time += timings[k]['compileInner']

timings[f'full/{target_ext}/precompilation'] = { 'compileInner': precompilation_time }

# Output to benchmark file
json_data = []
for k, v in timings.items():
    if not k.startswith('full'):
        continue

    name = k.split('/')[1:]
    name = ' : '.join(reversed(name))

    data = {
        'name': name,
        'value': v['compileInner'],
        'unit': 'milliseconds'
    }

    json_data.append(data)

# TODO: append target to benchmark file name
with open(args.output, 'w') as file:
    json.dump(json_data, file, indent=4)

# Generate readable Markdown as well
print(4 * '\n')
print('# Slang MDL benchmark results\n')
print('## Module precompilation time\n')
precomp_key = f'full/{target_ext}/precompilation'
if precomp_key in timings:
    print(f'Total: **{timings[precomp_key]["compileInner"]} ms**\n')
else:
    print("No precompilation data available\n")

print('## Module compilation for entry points\n')

entries = [ 'Closest Hit', 'Any Hit', 'Shadow' ]
prefixes = [ 'closesthit', 'anyhit', 'shadow' ]

table = prettytable.PrettyTable()
table.set_style(prettytable.MARKDOWN)
table.field_names = [ 'Entry', 'Total' ]

total = 0
for entry, prefix in zip(entries, prefixes):
    row = [ entry ]
    key = f'full/{target_ext}/module/{prefix}'
    if key in timings:
        db = timings[key]
        spCompile = db.get('compileInner', 0)
        row.append(f'{spCompile:.3f}ms')
        table.add_row(row)
        total += spCompile
    else:
        row.append('Failed')
        table.add_row(row)
        print(f"[Warning] Compilation failed for module/{prefix}")

print(f'Total: **{total} ms**\n')
print(table, end='\n\n')

print('## Monolithic compilation for entry points\n')

table = prettytable.PrettyTable()
table.set_style(prettytable.MARKDOWN)
table.field_names = [ 'Entry', 'Total' ]

total = 0
for entry, prefix in zip(entries, prefixes):
    row = [ entry ]
    key = f'full/{target_ext}/mono/{prefix}'
    if key in timings:
        db = timings[key]
        spCompile = db.get('compileInner', 0)
        row.append(f'{spCompile:.3f}ms')
        table.add_row(row)
        total += spCompile
    else:
        row.append('Failed')
        table.add_row(row)
        print(f"[Warning] Compilation failed for mono/{prefix}")

print(f'Total: **{total} ms**\n')
print(table, end='\n\n')